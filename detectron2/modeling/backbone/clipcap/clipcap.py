import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import numpy as np

import clip


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec:  # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(
                    TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class ClipCaptionModel(nn.Module):

    def pad_captions(self, tokens):
        tokens = torch.tensor(tokens)
        self.max_seq_len = 40
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def inference(self, prefix, use_beam_search=False):
        self.clip_project.eval()
        self.eval()
        self.gpt.eval()
        prefix_embed = self.clip_project(prefix).reshape(1, self.prefix_length, -1).detach()
        if use_beam_search:
            generated_text_prefix = generate_beam1(self, self.tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(self, self.tokenizer, embed=prefix_embed)
        return generated_text_prefix

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 1024,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.Transformer):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.activation = {}
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                  clip_length, num_layers)


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser


# TODO
def pad_tokens(self, item: int):
    tokens = self.captions_tokens[item]
    padding = self.max_seq_len - tokens.shape[0]
    if padding > 0:
        tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        self.captions_tokens[item] = tokens
    elif padding < 0:
        tokens = tokens[:self.max_seq_len]
        self.captions_tokens[item] = tokens
    mask = tokens.ge(0)  # mask is zero where we out of sequence
    tokens[~mask] = 0
    mask = mask.float()
    mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
    return tokens, mask




def train(GT, prefix, model: ClipCaptionModel, args,
          lr: float = 2e-5, warmup_steps: int = 500, output_prefix: str = ""):
    device = torch.device('cuda:0')
    batch_size = args.bs
    epochs = args.epochs

    # model = model.to(device)
    model.train()
    # optimizer = AdamW(model.parameters(), lr=lr)
    # train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    # )
    # save_config(args)
    captions_tokens = []
    caption2embedding = prefix
    for gt in GT:
        captions_tokens.append(torch.tensor(model.tokenizer.encode(gt), dtype=torch.int64))
        max_seq_len = max(max_seq_len, captions_tokens[-1].shape[0])
    all_len = torch.tensor([len(captions_tokens[i]) for i in range(len(len(captions_tokens)))]).float()
    max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
    return model


def pseudo_labeling_loss(prefix_teacher, prefix_student, model: ClipCaptionModel, prefix_length=10, isregion=False):
    gpt_embedding_size = model.gpt.transformer.wte.weight.shape[1]
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(prefix_student.device)
    # embed_student = model.clip_project(prefix_student).view(-1, prefix_length, gpt_embedding_size)

    filter_value = -float("Inf")
    model.eval()
    entry_length = 40
    temperature = 1.0
    top_p = 0.8
    stop_token: str = '.'
    stop_token_index = model.tokenizer.encode(stop_token)[0]
    losses = []
    tokens = None
    generated_list = []
    embedding_list = []
    model = model.to(prefix_teacher.device)
    with torch.no_grad():
        embed_teacher = model.clip_project(prefix_teacher).view(-1, prefix_length, gpt_embedding_size).detach()
        for entry_idx in range(len(embed_teacher)):
            generated_teacher = embed_teacher[entry_idx].unsqueeze(0)
            entry_loss = []
            tokens = None
            for i in range(entry_length):
                outputs_teacher = model.gpt(inputs_embeds=generated_teacher)
                logits_teacher = outputs_teacher.logits.detach()

                logits_teacher = logits_teacher[:, -1, :] / (temperature if temperature > 0 else 1.0)
                with torch.no_grad():
                    sorted_logits, sorted_indices = torch.sort(logits_teacher, descending=True)
                    cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                        ..., :-1
                                                        ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits_teacher[:, indices_to_remove] = filter_value
                    next_token_teacher = torch.argmax(logits_teacher, -1).unsqueeze(0)
                    next_token_embed_teacher = model.gpt.transformer.wte(next_token_teacher)
                    if tokens is None:
                        tokens = next_token_teacher
                    else:
                        tokens = torch.cat((tokens, next_token_teacher), dim=1)
                    generated_teacher = torch.cat((generated_teacher, next_token_embed_teacher), dim=1)

                    if stop_token_index == next_token_teacher.item():
                        break
            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = model.tokenizer.decode(output_list)
            embedding_list.append(output_list)
            generated_list.append(output_text)
    if isregion:
        masks = []
        final_emb = []
        for i in range(len(embedding_list)):
            emb,mask = model.pad_captions(embedding_list[i])
            masks.append(mask)
            final_emb.append(emb)
        masks = torch.stack(masks,0).to(prefix_student.device)
        final_emb = torch.stack(final_emb,0).to(prefix_student.device)
        embedding_list = final_emb
        # print(embedding_list)
        teacher_tokens = embedding_list
        # teacher_tokens = torch.tensor(embedding_list).to(prefix_student.device)
        outputs = model(teacher_tokens, prefix_student,mask=masks)
    else:
        teacher_tokens = torch.tensor(embedding_list).to(prefix_student.device)
        outputs = model(teacher_tokens, prefix_student)
    logits = outputs.logits[:, prefix_length - 1: -1]
    # print("hereeeeee")
    # print(logits.shape)
    # print(tokens.shape)
    loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), teacher_tokens.flatten(), ignore_index=0)

    return loss, None  # , generated_list


def unsupervised_loss(prefix_teacher, prefix_student, model: ClipCaptionModel, prefix_length=10):
    gpt_embedding_size = model.gpt.transformer.wte.weight.shape[1]

    embed_teacher = model.clip_project(prefix_teacher).view(-1, prefix_length, gpt_embedding_size).detach()

    embed_student = model.clip_project(prefix_student).view(-1, prefix_length, gpt_embedding_size)

    filter_value = -float("Inf")
    model.eval()
    entry_length = 67
    temperature = 1.0
    top_p = 0.8
    stop_token: str = '.'
    stop_token_index = model.tokenizer.encode(stop_token)[0]
    losses = []
    tokens = None
    generated_list = []
    for p in model.parameters():
        p.requires_grad = False
    for entry_idx in range(len(embed_teacher)):
        generated_teacher = embed_teacher[entry_idx].unsqueeze(0)
        generated_student = embed_student[entry_idx].unsqueeze(0)
        entry_loss = []
        tokens = None
        for i in range(entry_length):
            # print(i)

            outputs_teacher = model.gpt(inputs_embeds=generated_teacher)
            outputs_student = model.gpt(inputs_embeds=generated_student)
            logits_teacher = outputs_teacher.logits.detach()
            logits_student = outputs_student.logits

            logits_teacher = logits_teacher[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits_student = logits_student[:, -1, :] / (temperature if temperature > 0 else 1.0)
            # print(logits_student == logits_teacher)

            loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(logits_student, dim=-1),
                                              torch.nn.functional.log_softmax(logits_teacher.detach(), dim=-1).detach(),
                                              reduction='batchmean', log_target=True)
            # loss = torch.nn.functional.cross_entropy(logits_student, torch.softmax(logits_teacher,1).detach())
            # print(loss)
            entry_loss.append(loss)

            with torch.no_grad():
                sorted_logits, sorted_indices = torch.sort(logits_teacher, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits_teacher[:, indices_to_remove] = filter_value
                next_token_teacher = torch.argmax(logits_teacher, -1).unsqueeze(0)
                next_token_embed_teacher = model.gpt.transformer.wte(next_token_teacher)
                # if tokens is None:
                #     tokens = next_token_teacher
                # else:
                #     tokens = torch.cat((tokens, next_token_teacher), dim=1)
                generated_teacher = torch.cat((generated_teacher, next_token_embed_teacher), dim=1)

                # student
                sorted_logits, sorted_indices = torch.sort(logits_student, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits_student[:, indices_to_remove] = filter_value
                next_token_student = torch.argmax(logits_student, -1).unsqueeze(0)
                next_token_embed_student = model.gpt.transformer.wte(next_token_student)

                generated_student = torch.cat((generated_student, next_token_embed_student), dim=1)
                if stop_token_index == next_token_teacher.item():
                    break
        losses.append(sum(entry_loss) / len(entry_loss))
        # output_list = list(tokens.squeeze().cpu().numpy())
        # output_text = model.tokenizer.decode(output_list)
        # generated_list.append(output_text)
    return sum(losses) / len(losses), None  # , generated_list


def unsupervised_feature_loss(prefix_teacher, prefix_student, model: ClipCaptionModel, prefix_length=10):
    gpt_embedding_size = model.gpt.transformer.wte.weight.shape[1]

    embed_teacher = model.clip_project(prefix_teacher).view(-1, prefix_length, gpt_embedding_size).detach()

    embed_student = model.clip_project(prefix_student).view(-1, prefix_length, gpt_embedding_size)

    filter_value = -float("Inf")
    model.eval()
    entry_length = 67
    temperature = 1.0
    top_p = 0.8
    stop_token: str = '.'
    stop_token_index = model.tokenizer.encode(stop_token)[0]
    losses = []
    tokens = None
    generated_list = []
    for p in model.parameters():
        p.requires_grad = False
    for entry_idx in range(len(embed_teacher)):
        generated_teacher = embed_teacher[entry_idx].unsqueeze(0)
        generated_student = embed_student[entry_idx].unsqueeze(0)
        entry_loss = []
        tokens = None
        for i in range(entry_length):
            # print(i)
            teacher_features = model.gpt(inputs_embeds=generated_teacher).logits().detach()
            student_features = model.gpt(inputs_embeds=generated_student).logits()

            logits_teacher = model.lm_head(teacher_features).detach()
            logits_student = model.lm_head(student_features).detach()

            logits_teacher = logits_teacher[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits_student = logits_student[:, -1, :] / (temperature if temperature > 0 else 1.0)
            # print(logits_student == logits_teacher)
            loss_fn = nn.MSELoss()
            loss = loss_fn(teacher_features.detach(), student_features)
            # loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(logits_student, dim=-1),
            #                                   torch.nn.functional.log_softmax(logits_teacher.detach(), dim=-1).detach(),
            #                                   reduction='batchmean', log_target=True)
            # loss = torch.nn.functional.cross_entropy(logits_student, torch.softmax(logits_teacher,1).detach())
            # print(loss)
            entry_loss.append(loss)

            with torch.no_grad():
                sorted_logits, sorted_indices = torch.sort(logits_teacher, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits_teacher[:, indices_to_remove] = filter_value
                next_token_teacher = torch.argmax(logits_teacher, -1).unsqueeze(0)
                next_token_embed_teacher = model.gpt.transformer.wte(next_token_teacher)
                # if tokens is None:
                #     tokens = next_token_teacher
                # else:
                #     tokens = torch.cat((tokens, next_token_teacher), dim=1)
                generated_teacher = torch.cat((generated_teacher, next_token_embed_teacher), dim=1)

                # student
                sorted_logits, sorted_indices = torch.sort(logits_student, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits_student[:, indices_to_remove] = filter_value
                next_token_student = torch.argmax(logits_student, -1).unsqueeze(0)
                next_token_embed_student = model.gpt.transformer.wte(next_token_student)

                generated_student = torch.cat((generated_student, next_token_embed_student), dim=1)
                if stop_token_index == next_token_teacher.item():
                    break
        losses.append(sum(entry_loss) / len(entry_loss))
        # output_list = list(tokens.squeeze().cpu().numpy())
        # output_text = model.tokenizer.decode(output_list)
        # generated_list.append(output_text)
    return sum(losses) / len(losses), None  # , generated_list


def generate_feature_caption(prefix, model: ClipCaptionModel, prefix_length=10):
    gpt_embedding_size = model.gpt.transformer.wte.weight.shape[1]

    embed = model.clip_project(prefix).view(-1, prefix_length, gpt_embedding_size)

    filter_value = -float("Inf")
    model.eval()
    entry_length = 67
    temperature = 1.0
    top_p = 0.8
    stop_token: str = '.'
    stop_token_index = model.tokenizer.encode(stop_token)[0]
    losses = []
    tokens = None
    generated_list = []
    break_flag = False
    final_features = None
    res = []
    for p in model.parameters():
        p.requires_grad = False
    for entry_idx in range(len(embed)):
        generated = embed[entry_idx].unsqueeze(0)
        tokens = None
        break_flag = False
        for i in range(entry_length):
            # print(i)
            features = model.gpt(inputs_embeds=generated).logits
            final_features = features
            logits = model.lm_head(features)

            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                ..., :-1
                                                ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value
            next_token = torch.argmax(logits, -1).unsqueeze(0)
            next_token_embed = model.gpt.transformer.wte(next_token)

            generated = torch.cat((generated, next_token_embed), dim=1)

            if stop_token_index == next_token.item():
                res.append(features[:, -1, :])
                final_features = None
                break
        if final_features is not None:
            res.append(final_features[:, -1, :])

    return res


def generate_first_feature_caption(prefix, model: ClipCaptionModel, prefix_length=10):
    gpt_embedding_size = model.gpt.transformer.wte.weight.shape[1]

    embed = model.clip_project(prefix).view(-1, prefix_length, gpt_embedding_size)

    filter_value = -float("Inf")
    model.eval()
    entry_length = 67
    temperature = 1.0
    top_p = 0.8
    stop_token: str = '.'
    stop_token_index = model.tokenizer.encode(stop_token)[0]
    losses = []
    tokens = None
    generated_list = []
    break_flag = False
    out_features = None
    res = []

    for entry_idx in range(len(embed)):
        generated = embed[entry_idx].unsqueeze(0)
        tokens = None
        break_flag = False
        out_features = None

        for i in range(entry_length):

            # print(i)
            features = model.gpt(inputs_embeds=generated)

            out_features = model.activation['first_layer']

            logits = features.logits

            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                ..., :-1
                                                ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value
            next_token = torch.argmax(logits, -1).unsqueeze(0)
            next_token_embed = model.gpt.transformer.wte(next_token)
            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)
            generated = torch.cat((generated, next_token_embed), dim=1)

            if stop_token_index == next_token.item():
                res.append(out_features[:, -1, :])
                out_features = None
                break

        output_list = list(tokens.squeeze().cpu().numpy())
        output_text = model.tokenizer.decode(output_list)
        generated_list.append(output_text)
        if out_features is not None:
            res.append(out_features[:, -1, :])

    return res, generated_list


def v2l(prefix, model: ClipCaptionModel):
    prefix_length = 40
    gpt_embedding_size = 768
    embed = model(prefix).view(-1, prefix_length, gpt_embedding_size)[:, -1, :]
    # embed = model.clip_project(prefix).view(-1, prefix_length, gpt_embedding_size)
    return embed.view(embed.shape[0], -1)


# def prompt_consistency_loss(images):
#     classes = ['aeroplane', 'bird', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'cat',
#                'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant',
#                'sheep', 'sofa', 'train', 'tv/monitor']
#
#     prompts = ['image of ' + c for c in classes]
#
#     clip_model, clip_preprocess = clip.load("RN50", device='cpu')
#
#         self.clip_model.visual = None
def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in trange(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                # if tokens is None:
                #     tokens = next_token
                # else:
                #     tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]
