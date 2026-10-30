"""Bounded process-local decoded pairs; every cache hit still verifies sealed file bytes."""

from collections import OrderedDict

from PIL import Image

from damage_analytics.training.manifests import Release, Sample, read_pair


class PairCache:
    """LRU of native RGB/L pairs, never augmented tensors or stale unverified ground truth."""

    def __init__(self, megabytes: int = 0) -> None:
        if type(megabytes) is not int or megabytes < 0:
            raise ValueError("Cache budget must be a nonnegative integer MiB")
        self.budget = megabytes * 1024**2
        self.bytes = self.hits = self.misses = 0
        self.pairs: OrderedDict[str, tuple[Image.Image, Image.Image]] = OrderedDict()

    def read(self, release: Release, sample: Sample) -> tuple[Image.Image, Image.Image]:
        """Validate hashes on reuse; normal paired transforms never mutate cached PIL objects."""
        if sample.sample_id in self.pairs:
            release.file(sample.image)
            release.file(sample.mask)
            self.hits += 1
            self.pairs.move_to_end(sample.sample_id)
            return self.pairs[sample.sample_id]
        self.misses += 1
        pair = read_pair(release, sample)
        size = sample.width * sample.height * 4  # RGB uint8 + L uint8, excluding PIL overhead
        if 0 < size <= self.budget:
            while self.bytes + size > self.budget:
                _, (image, _) = self.pairs.popitem(last=False)
                self.bytes -= image.width * image.height * 4
            self.pairs[sample.sample_id] = pair
            self.bytes += size
        return pair
