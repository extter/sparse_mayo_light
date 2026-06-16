from .blocks import DoubleConv, ResidualDoubleConv, DownBlock, UpBlock
from .unet import SimpleUNet, ResidualUNet

__all__ = [
    "DoubleConv",
    "ResidualDoubleConv",
    "DownBlock",
    "UpBlock",
    "SimpleUNet",
    "ResidualUNet",
    "SSIMCustomLoss",
    "MixedLoss",
    "MayoRecoCorruptedDataset",
    "get_dataloaders",
    "train",
    "get_device",
    "get_ct_operator",
    "plot_sample",
    "plot_results",
    "plot_loss_curves",
]


def __getattr__(name):
    if name in {"MayoRecoCorruptedDataset", "get_dataloaders"}:
        from .dataset import MayoRecoCorruptedDataset, get_dataloaders

        return {
            "MayoRecoCorruptedDataset": MayoRecoCorruptedDataset,
            "get_dataloaders": get_dataloaders,
        }[name]

    if name == "train":
        from .train import train

        return train

    if name in {"SSIMCustomLoss", "MixedLoss"}:
        from .losses import SSIMCustomLoss, MixedLoss

        return {
            "SSIMCustomLoss": SSIMCustomLoss,
            "MixedLoss": MixedLoss,
        }[name]

    if name in {"get_device", "get_ct_operator", "plot_sample", "plot_results", "plot_loss_curves"}:
        from .utilities import get_device, get_ct_operator, plot_sample, plot_results, plot_loss_curves

        return {
            "get_device": get_device,
            "get_ct_operator": get_ct_operator,
            "plot_sample": plot_sample,
            "plot_results": plot_results,
            "plot_loss_curves": plot_loss_curves,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
