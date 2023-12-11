import torch


def RegionCLIP2CLIP(regionCLIP_weights_path):
    regionCLIP_weights = torch.load(regionCLIP_weights_path, 'cpu')['model']

    clip_weights = {}

    for key in regionCLIP_weights.keys():
        if 'backbone' in key and 'teacher' not in key and 'offline' not in key:
            print("key is ", key)
            clip_weights[key.lstrip('backbone')[1:]] = regionCLIP_weights[key]

    return clip_weights
