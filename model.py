import torch
import torch.nn as nn
import torch.optim as optim

import config
from utils import save_checkpoint, load_checkpoint

class CNNBlock(nn.Module):
    """
    A helper class to create a convolutional layer followed by a batch normalization layer and a leaky ReLU activation function.
    """
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            bn_act (bool): whether to use batch normalization and activation. This is helpful when we want to use the convolutional layer in the last layer of the model.
            **kwargs: additional arguments to be passed to the nn.Conv2d layer.
        """
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn_act:
            x = self.bn(x)
            x = self.leaky(x)
        
        return x
        

class ResidualBlock(nn.Module):
    """
    A helper class to create a residual block.
    """
    def __init__(self, channels, use_residual=True, num_repeats=1):
        """
        Args:
            channels (int): number of input and output channels.
            use_residual (bool): whether to use the residual connection.
            num_repeats (int): number of times to repeat the block.
        """
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + self.use_residual * x

        return x
    

class ScalePrediction(nn.Module):
    """
    A helper class to create a scale prediction layer.
    """
    def __init__(self, in_channels, num_classes, anchors_per_scale=3):
        """
        Args:
            in_channels (int): number of input channels.
            num_classes (int): number of classes.
            anchors_per_scale (int): number of anchors per scale.
        """
        super(ScalePrediction, self).__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, kernel_size=3, padding=1),
            CNNBlock(2*in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1),
        )
        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale

    def forward(self, x):
        return (
            self.pred(x)
                .reshape(x.shape[0], self.anchors_per_scale, self.num_classes + 5, x.shape[2], x.shape[3])
                .permute(0, 1, 3, 4, 2)
        )
    

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80, cfg=config.ARCHITECTURE):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cfg = cfg
        self.layers = self._create_conv_layers()
        

    def forward(self, x):
        outputs = []
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs


    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in self.cfg:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(
                    ResidualBlock(
                        in_channels,
                        num_repeats=num_repeats,
                    )
                )

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(
                        nn.Upsample(scale_factor=2),
                    )
                    in_channels = in_channels * 3

        return layers
    

def test():
    num_classes = 7
    model = YOLOv3(num_classes=num_classes, cfg=config.ARCHITECTURE)
    img_size = 416
    x = torch.randn((2, 3, img_size, img_size))
    out = model(x)
    assert out[0].shape == (2, 3, img_size//32, img_size//32, 5 + num_classes)
    assert out[1].shape == (2, 3, img_size//16, img_size//16, 5 + num_classes)
    assert out[2].shape == (2, 3, img_size//8, img_size//8, 5 + num_classes)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print("Success!")

# test()