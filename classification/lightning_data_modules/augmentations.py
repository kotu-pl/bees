from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandAugment, InterpolationMode, ToTensor, Normalize
from .transforms_utils import ResizePad224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class AugmentationFactory:
    _rand_table = {
        "low":    dict(num_ops=1, magnitude=5),
        "medium": dict(num_ops=2, magnitude=7),
        "high":   dict(num_ops=3, magnitude=9),
    }

    def __init__(self, level: str = "", resize_pad_224: bool = False, interpolation: InterpolationMode = InterpolationMode.BILINEAR, fill: int | tuple[int, int, int] | None = None):
        self.level = (level or "").lower()
        self.resize_pad_224 = resize_pad_224
        self.interpolation = interpolation
        self.fill = fill

    def build(self, train: bool = True):
        tf = []
        if self.resize_pad_224:
            tf.append(ResizeCrop224())

        if train:
            tf.extend(self._train_policy())

        tf.extend([ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)])

        return transforms.Compose(tf)

    def _train_policy(self):
        if self.level in ("", "none", "off"):
            return []

        if self.level == "auto":
            return [
                AutoAugment(
                    policy=AutoAugmentPolicy.IMAGENET,
                    interpolation=self.interpolation,
                    fill=self.fill
                )
            ]

        if self.level in self._rand_table:
            kw = self._rand_table[self.level]
            return [
                RandAugment(
                    interpolation=self.interpolation,
                    fill=self.fill,
                    **kw
                )
            ]

        raise ValueError(f"Unknown augmentation level: {self.level!r}")
