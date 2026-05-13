
import torch 
import os 
import torch.nn as nn 
import torch.nn.init as init    
import time

def init_weights(net: nn.Module, init_type: str = "normal", gain: float = 0.02):
    """
    Initialize the weights of a network, based on the layer type.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


def get_config(model):
    r"""
    Return the configuration dictionary from the provided model, in the shape of
    a dictionary.
    """
    out_cfg = {
        "ch_in": model.ch_in,
        "ch_out": model.ch_out,
        "middle_ch": model.middle_ch,
        "n_layers_per_block": model.n_layers_per_block,
        "down_layers": model.down_layers,
        "up_layers": model.up_layers,
        "n_heads": model.n_heads,
        "final_activation": model.final_activation,
    }
    return out_cfg

def create_path_if_not_exists(path: str) -> None:
    r"""
    Check if the path exists. If this is not the case, it creates the required folders.

    :param str path: The path to be checked and created.
    """
    if not os.path.isdir(path):
        os.makedirs(path)

def formatted_time(start_time: float) -> str:
    r"""
    Given a starting time, computes the difference between the actual time and the starting time, and returns a nice string
    representation of time, in the format %H:%M:%S.

    :param float start_time: The starting time.
    """
    total_time = time.time() - start_time

    # Convert elapsed time to hours, minutes, and seconds
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Format using an f-string with %H:%M:%S style
    formatted_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    return formatted_time