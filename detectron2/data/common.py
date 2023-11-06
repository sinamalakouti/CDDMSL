# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import itertools
import logging
import numpy as np
import pickle
import random
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

from detectron2.utils.serialize import PicklableWrapper

__all__ = ["MapDataset", "DatasetFromList", "AspectRatioGroupedDataset", "ToIterableDataset"]


class MapDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.

    Args:
        dataset: a dataset where map function is applied.
        map_func: a callable which maps the element in dataset. map_func is
            responsible for error handling, when error happens, it needs to
            return None so the MapDataset will randomly use other
            elements from the dataset.
    """

    def __init__(self, dataset, map_func):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )


class DatasetFromList(data.Dataset):
    """
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(self, lst: list, copy: bool = True, serialize: bool = True):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool): whether to hold memory using serialized objects, when
                enabled, data loader workers can use shared RAM from master
                process instead of making a copy.
        """
        self._lst = lst
        self._copy = copy
        self._serialize = serialize

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            logger = logging.getLogger(__name__)
            logger.info(
                "Serializing {} elements to byte tensors and concatenating them all ...".format(
                    len(self._lst)
                )
            )
            self._lst = [_serialize(x) for x in self._lst]
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)
            logger.info("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024 ** 2))

    def __len__(self):
        if self._serialize:
            return len(self._addr)
        else:
            return len(self._lst)

    def __getitem__(self, idx):
        if self._serialize:
            start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
            end_addr = self._addr[idx].item()
            bytes = memoryview(self._lst[start_addr:end_addr])
            return pickle.loads(bytes)
        elif self._copy:
            return copy.deepcopy(self._lst[idx])
        else:
            return self._lst[idx]


class ToIterableDataset(data.IterableDataset):
    """
    Convert an old indices-based (also called map-style) dataset
    to an iterable-style dataset.
    """

    def __init__(self, dataset, sampler):
        """
        Args:
            dataset (torch.utils.data.Dataset): an old-style dataset with ``__getitem__``
            sampler (torch.utils.data.sampler.Sampler): a cheap iterable that produces indices
                to be applied on ``dataset``.
        """
        assert not isinstance(dataset, data.IterableDataset), dataset
        assert isinstance(sampler, Sampler), sampler
        self.dataset = dataset
        self.sampler = sampler

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None or worker_info.num_workers == 1:
            for idx in self.sampler:
                yield self.dataset[idx]
        else:
            # With map-style dataset, `DataLoader(dataset, sampler)` runs the
            # sampler in main process only. But `DataLoader(ToIterableDataset(dataset, sampler))`
            # will run sampler in every of the N worker and only keep 1/N of the ids on each
            # worker. The assumption is that sampler is cheap to iterate and it's fine to discard
            # ids in workers.
            for idx in itertools.islice(
                self.sampler, worker_info.id, None, worker_info.num_workers
            ):
                yield self.dataset[idx]


class AspectRatioGroupedDataset(data.IterableDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(2)]
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def __iter__(self):
        for d in self.dataset:
            w, h = d["width"], d["height"]
            bucket_id = 0 if w > h else 1
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]


class AspectRatioGroupedDatasetTwoCrop(AspectRatioGroupedDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.
    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(2)]
        self._buckets_key = [[] for _ in range(2)]
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def __iter__(self):
        for d in self.dataset:
            # d is a tuple with len = 2
            # It's two images (same size) from the same image instance
            w, h = d[0]["width"], d[0]["height"]
            bucket_id = 0 if w > h else 1

            # bucket = bucket for normal images
            bucket = self._buckets[bucket_id]
            bucket.append(d[0])

            # buckets_key = bucket for augmented images
            buckets_key = self._buckets_key[bucket_id]
            buckets_key.append(d[1])
            if len(bucket) == self.batch_size:
                yield (bucket[:], buckets_key[:])
                del bucket[:]
                del buckets_key[:]


class AspectRatioGroupedSemiSupDatasetTwoCrop(AspectRatioGroupedDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.
    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: a tuple containing two iterable generators. （labeled and unlabeled data)
               Each element must be a dict with keys "width" and "height", which will be used
               to batch data.
            batch_size (int):
        """

        self.label_dataset, self.unlabel_dataset = dataset
        self.batch_size_label = batch_size[0]
        self.batch_size_unlabel = batch_size[1]

        self._label_buckets = [[] for _ in range(2)]
        self._label_buckets_key = [[] for _ in range(2)]
        self._label_buckets_style = [[] for _ in range(2)]
        self._unlabel_buckets = [[] for _ in range(2)]
        self._unlabel_buckets_key = [[] for _ in range(2)]
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def __iter__(self):
        label_bucket, unlabel_bucket = [], []
        for d_label, d_unlabel in zip(self.label_dataset, self.unlabel_dataset):

            # d is a tuple with len = 2
            # It's two images (same size) from the same image instance
            # d[0] is with strong augmentation, d[1] is with weak augmentation

            # because we are grouping images with their aspect ratio
            # label and unlabel buckets might not have the same number of data
            # i.e., one could reach batch_size, while the other is still not
            if len(label_bucket) != self.batch_size_label:
                w, h = d_label[0]["width"], d_label[0]["height"]
                label_bucket_id = 0 if w > h else 1
                label_bucket = self._label_buckets[label_bucket_id]
                label_bucket.append(d_label[0])
                label_buckets_key = self._label_buckets_key[label_bucket_id]
                label_buckets_key.append(d_label[1])
                label_buckets_style = self._label_buckets_style[label_bucket_id]
                label_buckets_style.append(d_label[2])

            if len(unlabel_bucket) != self.batch_size_unlabel:
                w, h = d_unlabel[0]["width"], d_unlabel[0]["height"]
                unlabel_bucket_id = 0 if w > h else 1
                unlabel_bucket = self._unlabel_buckets[unlabel_bucket_id]
                unlabel_bucket.append(d_unlabel[0])
                unlabel_buckets_key = self._unlabel_buckets_key[unlabel_bucket_id]
                unlabel_buckets_key.append(d_unlabel[1])

            # yield the batch of data until all buckets are full
            if (
                len(label_bucket) == self.batch_size_label
                and len(unlabel_bucket) == self.batch_size_unlabel
                and len(label_buckets_style) == self.batch_size_label
            ):
                # label_strong, label_weak, unlabed_strong, unlabled_weak

                assert len(label_buckets_style) == len(label_bucket), 'not equal length for buckets'
                yield (
                    label_bucket[:],
                    label_buckets_key[:],
                    label_buckets_style[:],
                    unlabel_bucket[:],
                    unlabel_buckets_key[:],
                )
                del label_bucket[:]
                del label_buckets_key[:]
                del label_buckets_style[:],
                del unlabel_bucket[:]
                del unlabel_buckets_key[:]