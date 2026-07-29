import torch
import torch.nn as nn
import torchvision.transforms as T

from util import swap_axis


class Generator1x(nn.Module):
    def __init__(self, conf):
        super(Generator1x, self).__init__()
        print("Generator 1x")
        struct = conf.G_structure
        # First layer - Converting RGB image to latent space
        self.first_layer = nn.Conv2d(
            in_channels=1, out_channels=conf.G_chan, kernel_size=struct[0], bias=False
        )

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            feature_block += [
                nn.Conv2d(
                    in_channels=conf.G_chan,
                    out_channels=conf.G_chan,
                    kernel_size=struct[layer],
                    bias=False,
                )
            ]
        self.feature_block = nn.Sequential(*feature_block)
        # Final layer - Down-sampling and converting back to image
        self.final_layer = nn.Conv2d(
            in_channels=conf.G_chan, out_channels=1, kernel_size=struct[-1], stride=1, bias=False
        )

        # Calculate number of pixels shaved in the forward pass
        img_out_size_1x = self.forward(torch.FloatTensor(torch.ones([1, 1, conf.input_crop_size, conf.input_crop_size]))).shape[-1]
        # k1x_size = conf.G_kernel_size
        # k2x_analytic_size = (3 * k1x_size - 2) - 2 * (k1x_size // 2)
        #
        # self.output_size = (conf.input_crop_size - 2 * (k2x_analytic_size // 2)) // 2
        # self.forward_shave = int(conf.input_crop_size) - self.output_size
        # self.kernel_size = conf.G_kernel_size

        self.output_size = self.forward(
            torch.FloatTensor(torch.ones([1, 1, conf.input_crop_size, conf.input_crop_size]))
        ).shape[-1]
        self.kernel_size = int(conf.input_crop_size) - self.output_size + 1
        self.forward_shave = int(conf.input_crop_size * conf.scale_factor) - self.output_size

    def forward(self, input_tensor):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        input_tensor = swap_axis(input_tensor)
        downscaled = self.first_layer(input_tensor)
        features = self.feature_block(downscaled)
        output = self.final_layer(features)
        return swap_axis(output)


class Discriminator1x(nn.Module):
    def __init__(self, conf, inp_size):
        super(Discriminator1x, self).__init__()

        # First layer - Convolution (with no ReLU)
        self.first_layer = nn.utils.spectral_norm(
            nn.Conv2d(
                in_channels=3, out_channels=conf.D_chan, kernel_size=conf.D_kernel_size, bias=True
            )
        )
        feature_block = []  # Stacking layers with 1x1 kernels
        for _ in range(1, conf.D_n_layers - 1):
            feature_block += [
                nn.utils.spectral_norm(
                    nn.Conv2d(
                        in_channels=conf.D_chan, out_channels=conf.D_chan, kernel_size=1, bias=True
                    )
                ),
                nn.BatchNorm2d(conf.D_chan),
                nn.ReLU(True),
            ]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels=conf.D_chan, out_channels=1, kernel_size=1, bias=True)
            ),
            nn.Sigmoid(),
        )

        # Calculate number of pixels shaved in the forward pass
        self.forward_shave = (
            inp_size
            - self.forward(torch.FloatTensor(torch.ones([1, 3, inp_size, inp_size]))).shape[-1]
        )

    def forward(self, input_tensor):
        receptive_extraction = self.first_layer(input_tensor)
        features = self.feature_block(receptive_extraction)
        final = self.final_layer(features)
        return final


class GeneratorS1(nn.Module):
    def __init__(self, conf):
        super(GeneratorS1, self).__init__()
        print("Generator S1")
        struct = conf.G_structure
        # First layer - Converting RGB image to latent space
        self.first_layer = nn.Conv2d(
            in_channels=1, out_channels=conf.G_chan, kernel_size=struct[0], bias=False
        )

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            feature_block += [
                nn.Conv2d(
                    in_channels=conf.G_chan,
                    out_channels=conf.G_chan,
                    kernel_size=struct[layer],
                    bias=False,
                )
            ]
        self.feature_block = nn.Sequential(*feature_block)
        # Final layer - Down-sampling and converting back to image
        self.final_layer = nn.Conv2d(
            in_channels=conf.G_chan, out_channels=1, kernel_size=struct[-1], stride=1, bias=False
        )

        # Calculate number of pixels shaved in the forward pass
        self.output_size = self.forward(
            torch.FloatTensor(torch.ones([1, 1, conf.input_crop_size, conf.input_crop_size]))
        ).shape[-1]
        self.forward_shave = int(conf.input_crop_size) - self.output_size

    def subForward(self, input_tensor):
        # GeneratorS1 takes input_tensor with shape B, 1, H, W
        # input_tensor = swap_axis(input_tensor)
        downscaled = self.first_layer(input_tensor)
        features = self.feature_block(downscaled)
        output = self.final_layer(features)
        # output = swap_axis(output)

        return output

    def forward(self, input_tensor):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        output = self.subForward(input_tensor)
        output = output[:, :, ::2, ::2]
        output = self.subForward(output)

        return output


class GeneratorS1_bicubicDown(nn.Module):
    def __init__(self, conf):
        super(GeneratorS1_bicubicDown, self).__init__()
        print("Generator S1 bicubicDown")
        struct = conf.G_structure
        # First layer - Converting RGB image to latent space
        self.first_layer = nn.Conv2d(
            in_channels=1, out_channels=conf.G_chan, kernel_size=struct[0], bias=False
        )

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            feature_block += [
                nn.Conv2d(
                    in_channels=conf.G_chan,
                    out_channels=conf.G_chan,
                    kernel_size=struct[layer],
                    bias=False,
                )
            ]
        self.feature_block = nn.Sequential(*feature_block)
        # Final layer - Down-sampling and converting back to image
        self.final_layer = nn.Conv2d(
            in_channels=conf.G_chan, out_channels=1, kernel_size=struct[-1], stride=1, bias=False
        )
        self.down = T.GaussianBlur((5, 5), sigma=(1.0, 1.0))

        # Calculate number of pixels shaved in the forward pass
        self.output_size = self.forward(
            torch.FloatTensor(torch.ones([1, 1, conf.input_crop_size, conf.input_crop_size]))
        ).shape[-1]
        self.forward_shave = int(conf.input_crop_size) - self.output_size

    def subForward(self, input_tensor):
        # GeneratorS1 takes input_tensor with shape B, 1, H, W
        # input_tensor = swap_axis(input_tensor)
        downscaled = self.first_layer(input_tensor)
        features = self.feature_block(downscaled)
        output = self.final_layer(features)
        # output = swap_axis(output)

        return output

    def forward(self, input_tensor):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        # output = self.subForward(input_tensor)
        output = self.down(input_tensor)
        output = output[:, :, ::2, ::2]
        output = self.subForward(output)

        return output


class GeneratorS1_1Down(nn.Module):
    def __init__(self, conf):
        super(GeneratorS1_1Down, self).__init__()
        print("Generator S1 1Down")
        struct = conf.G_structure
        # First layer - Converting RGB image to latent space
        self.first_layer = nn.Conv2d(
            in_channels=1, out_channels=conf.G_chan, kernel_size=struct[0], bias=False
        )

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            feature_block += [
                nn.Conv2d(
                    in_channels=conf.G_chan,
                    out_channels=conf.G_chan,
                    kernel_size=struct[layer],
                    bias=False,
                )
            ]
        self.feature_block = nn.Sequential(*feature_block)
        # Final layer - Down-sampling and converting back to image
        self.final_layer = nn.Conv2d(
            in_channels=conf.G_chan, out_channels=1, kernel_size=struct[-1], stride=1, bias=False
        )

        # Calculate number of pixels shaved in the forward pass
        self.output_size = self.forward(
            torch.FloatTensor(torch.ones([1, 1, conf.input_crop_size, conf.input_crop_size]))
        ).shape[-1]
        self.forward_shave = int(conf.input_crop_size) - self.output_size

    def subForward(self, input_tensor):
        # GeneratorS1 takes input_tensor with shape B, 1, H, W
        # input_tensor = swap_axis(input_tensor)
        downscaled = self.first_layer(input_tensor)
        features = self.feature_block(downscaled)
        output = self.final_layer(features)
        # output = swap_axis(output)

        return output

    def forward(self, input_tensor):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        # output = self.subForward(input_tensor)
        # output = self.down(input_tensor)
        # output = output[:, :, ::2, ::2]
        output = self.subForward(input_tensor)

        return output


class GeneratorS1_RGB(nn.Module):
    def __init__(self, conf):
        super(GeneratorS1_RGB, self).__init__()
        self.G_1 = GeneratorS1(conf)
        self.G_2 = GeneratorS1(conf)
        self.G_3 = GeneratorS1(conf)
        self.output_size = self.G_1.output_size

    def  subForward(self, input_tensor):
        assert input_tensor.shape[0] == 1 and input_tensor.shape[1] == 3
        input_tensor_1 = input_tensor[:, 0:1, ...]
        input_tensor_2 = input_tensor[:, 1:2, ...]
        input_tensor_3 = input_tensor[:, 2:3, ...]

        input_tensor_1 = self.G_1.subForward(input_tensor_1)
        input_tensor_2 = self.G_2.subForward(input_tensor_2)
        input_tensor_3 = self.G_3.subForward(input_tensor_3)

        output = torch.cat([input_tensor_1, input_tensor_2, input_tensor_3], dim=1)

        return output

    def forward(self, input_tensor):
        assert input_tensor.shape[1] == 3
        input_tensor_1 = input_tensor[:, 0:1, ...]
        input_tensor_2 = input_tensor[:, 1:2, ...]
        input_tensor_3 = input_tensor[:, 2:3, ...]

        input_tensor_1 = self.G_1.forward(input_tensor_1)
        input_tensor_2 = self.G_2.forward(input_tensor_2)
        input_tensor_3 = self.G_3.forward(input_tensor_3)

        output = torch.cat([input_tensor_1, input_tensor_2, input_tensor_3], dim=1)

        return output


class GeneratorS1_RGB_bicubicDown(nn.Module):
    def __init__(self, conf):
        super(GeneratorS1_RGB_bicubicDown, self).__init__()
        self.G_1 = GeneratorS1_bicubicDown(conf)
        self.G_2 = GeneratorS1_bicubicDown(conf)
        self.G_3 = GeneratorS1_bicubicDown(conf)
        self.output_size = self.G_1.output_size

    def subForward(self, input_tensor):
        assert input_tensor.shape[0] == 1 and input_tensor.shape[1] == 3
        input_tensor_1 = input_tensor[:, 0:1, ...]
        input_tensor_2 = input_tensor[:, 1:2, ...]
        input_tensor_3 = input_tensor[:, 2:3, ...]

        input_tensor_1 = self.G_1.subForward(input_tensor_1)
        input_tensor_2 = self.G_2.subForward(input_tensor_2)
        input_tensor_3 = self.G_3.subForward(input_tensor_3)

        output = torch.cat([input_tensor_1, input_tensor_2, input_tensor_3], dim=1)

        return output

    def forward(self, input_tensor):
        assert input_tensor.shape[1] == 3
        input_tensor_1 = input_tensor[:, 0:1, ...]
        input_tensor_2 = input_tensor[:, 1:2, ...]
        input_tensor_3 = input_tensor[:, 2:3, ...]

        input_tensor_1 = self.G_1.forward(input_tensor_1)
        input_tensor_2 = self.G_2.forward(input_tensor_2)
        input_tensor_3 = self.G_3.forward(input_tensor_3)

        output = torch.cat([input_tensor_1, input_tensor_2, input_tensor_3], dim=1)

        return output


class GeneratorS1_RGB_1Down(nn.Module):
    def __init__(self, conf):
        super(GeneratorS1_RGB_1Down, self).__init__()
        self.G_1 = GeneratorS1_1Down(conf)
        self.G_2 = GeneratorS1_1Down(conf)
        self.G_3 = GeneratorS1_1Down(conf)
        self.output_size = self.G_1.output_size

    def subForward(self, input_tensor):
        assert input_tensor.shape[0] == 1 and input_tensor.shape[1] == 3
        input_tensor_1 = input_tensor[:, 0:1, ...]
        input_tensor_2 = input_tensor[:, 1:2, ...]
        input_tensor_3 = input_tensor[:, 2:3, ...]

        input_tensor_1 = self.G_1.subForward(input_tensor_1)
        input_tensor_2 = self.G_2.subForward(input_tensor_2)
        input_tensor_3 = self.G_3.subForward(input_tensor_3)

        output = torch.cat([input_tensor_1, input_tensor_2, input_tensor_3], dim=1)

        return output

    def forward(self, input_tensor):
        assert input_tensor.shape[1] == 3
        input_tensor_1 = input_tensor[:, 0:1, ...]
        input_tensor_2 = input_tensor[:, 1:2, ...]
        input_tensor_3 = input_tensor[:, 2:3, ...]

        input_tensor_1 = self.G_1.forward(input_tensor_1)
        input_tensor_2 = self.G_2.forward(input_tensor_2)
        input_tensor_3 = self.G_3.forward(input_tensor_3)

        output = torch.cat([input_tensor_1, input_tensor_2, input_tensor_3], dim=1)

        return output


class GeneratorS1_Gplus(nn.Module):
    def __init__(self, conf):
        super(GeneratorS1_Gplus, self).__init__()
        self.G_1 = GeneratorS1(conf)
        self.G_2 = GeneratorS1(conf)
        self.G_3 = GeneratorS1(conf)
        self.output_size = self.G_1.output_size

    def subForward(self, input_tensor):
        assert input_tensor.shape[0] == 1 and input_tensor.shape[1] == 3
        input_tensor_1 = input_tensor[:, 0:1, ...]
        input_tensor_2 = input_tensor[:, 1:2, ...]
        input_tensor_3 = input_tensor[:, 2:3, ...]

        input_tensor_1 = self.G_1.subForward(self.G_2.subForward(input_tensor_1))
        input_tensor_2 = self.G_2.subForward(input_tensor_2)
        input_tensor_3 = self.G_3.subForward(self.G_2.subForward(input_tensor_3))

        output = torch.cat([input_tensor_1, input_tensor_2, input_tensor_3], dim=1)

        return output

    def forward(self, input_tensor):
        assert input_tensor.shape[0] == 1 and input_tensor.shape[1] == 3
        input_tensor_1 = input_tensor[:, 0:1, ...]
        input_tensor_2 = input_tensor[:, 1:2, ...]
        input_tensor_3 = input_tensor[:, 2:3, ...]

        input_tensor_1 = self.G_1.forward(self.G_2.forward(input_tensor_1))
        input_tensor_2 = self.G_2.forward(input_tensor_2)
        input_tensor_3 = self.G_3.forward(self.G_2.forward(input_tensor_3))

        output = torch.cat([input_tensor_1, input_tensor_2, input_tensor_3], dim=1)

        return output


class GeneratorS1C(nn.Module):
    def __init__(self, conf):
        super(GeneratorS1C, self).__init__()
        print("Generator S1C")
        struct = conf.G_structure
        # First layer - Converting RGB image to latent space
        self.first_layer = nn.Conv2d(
            in_channels=3, out_channels=conf.G_chan, kernel_size=struct[0], bias=False
        )

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            feature_block += [
                nn.Conv2d(
                    in_channels=conf.G_chan,
                    out_channels=conf.G_chan,
                    kernel_size=struct[layer],
                    bias=False,
                )
            ]
        self.feature_block = nn.Sequential(*feature_block)
        # Final layer - Down-sampling and converting back to image
        self.final_layer = nn.Conv2d(
            in_channels=conf.G_chan, out_channels=3, kernel_size=struct[-1], stride=1, bias=False
        )

        # Calculate number of pixels shaved in the forward pass
        self.output_size = self.forward(
            torch.FloatTensor(torch.ones([1, 3, conf.input_crop_size, conf.input_crop_size]))
        ).shape[-1]
        self.forward_shave = int(conf.input_crop_size) - self.output_size

    def subForward(self, input_tensor):
        # input_tensor = swap_axis(input_tensor)
        downscaled = self.first_layer(input_tensor)
        features = self.feature_block(downscaled)
        output = self.final_layer(features)
        # output = swap_axis(output)

        return output

    def forward(self, input_tensor):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        output = self.subForward(input_tensor)
        # output = output[:, :, ::2, ::2]
        img_tl = output[:, :, ::2, ::2]
        img_tr = output[:, :, ::2, 1::2]
        img_bl = output[:, :, 1::2, ::2]
        img_br = output[:, :, 1::2, 1::2]
        output = torch.stack([img_tl, img_tr, img_bl, img_br], dim=0).mean(dim=0)
        output = self.subForward(output)

        return output


class Generator(nn.Module):
    def __init__(self, conf):
        super(Generator, self).__init__()
        print("Generator Original")
        struct = conf.G_structure
        # First layer - Converting RGB image to latent space
        self.first_layer = nn.Conv2d(
            in_channels=1, out_channels=conf.G_chan, kernel_size=struct[0], bias=False
        )

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            feature_block += [
                nn.Conv2d(
                    in_channels=conf.G_chan,
                    out_channels=conf.G_chan,
                    kernel_size=struct[layer],
                    bias=False,
                )
            ]
        self.feature_block = nn.Sequential(*feature_block)
        # Final layer - Down-sampling and converting back to image
        self.final_layer = nn.Conv2d(
            in_channels=conf.G_chan,
            out_channels=1,
            kernel_size=struct[-1],
            stride=int(1 / conf.scale_factor),
            bias=False,
        )

        # Calculate number of pixels shaved in the forward pass
        self.output_size = self.forward(
            torch.FloatTensor(torch.ones([1, 1, conf.input_crop_size, conf.input_crop_size]))
        ).shape[-1]
        self.kernel_size = int(conf.input_crop_size) - self.output_size + 1
        self.forward_shave = int(conf.input_crop_size * conf.scale_factor) - self.output_size

    def forward(self, input_tensor):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        input_tensor = swap_axis(input_tensor)
        downscaled = self.first_layer(input_tensor)
        features = self.feature_block(downscaled)
        output = self.final_layer(features)
        return swap_axis(output)


class Generator_RGB(nn.Module):
    def __init__(self, conf):
        super(Generator_RGB, self).__init__()
        self.G_1 = Generator1x(conf)
        self.G_2 = Generator1x(conf)
        self.G_3 = Generator1x(conf)
        self.output_size = self.G_1.output_size // 2
        self.kernel_size = self.G_1.kernel_size

    def subForward(self, input_tensor):
        assert input_tensor.shape[1] == 3
        input_tensor_1 = input_tensor[:, 0:1, ...]
        input_tensor_2 = input_tensor[:, 1:2, ...]
        input_tensor_3 = input_tensor[:, 2:3, ...]

        input_tensor_1 = self.G_1.forward(input_tensor_1)
        input_tensor_2 = self.G_2.forward(input_tensor_2)
        input_tensor_3 = self.G_3.forward(input_tensor_3)

        output = torch.cat([input_tensor_1, input_tensor_2, input_tensor_3], dim=1)
        return output

    def forward(self, input_tensor):
        output = self.subForward(input_tensor)
        output = output[:, :, ::2, ::2]
        return output


class Discriminator(nn.Module):
    def __init__(self, conf, in_channels=3):
        super(Discriminator, self).__init__()

        # First layer - Convolution (with no ReLU)
        self.first_layer = nn.utils.spectral_norm(
            nn.Conv2d(
                in_channels=in_channels, out_channels=conf.D_chan, kernel_size=conf.D_kernel_size, bias=True
            )
        )
        feature_block = []  # Stacking layers with 1x1 kernels
        for _ in range(1, conf.D_n_layers - 1):
            feature_block += [
                nn.utils.spectral_norm(
                    nn.Conv2d(
                        in_channels=conf.D_chan, out_channels=conf.D_chan, kernel_size=1, bias=True
                    )
                ),
                nn.BatchNorm2d(conf.D_chan),
                nn.ReLU(True),
            ]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels=conf.D_chan, out_channels=1, kernel_size=1, bias=True)
            ),
            nn.Sigmoid(),
        )

        # Calculate number of pixels shaved in the forward pass
        self.forward_shave = (
            conf.input_crop_size
            - self.forward(
                torch.FloatTensor(torch.ones([1, in_channels, conf.input_crop_size, conf.input_crop_size]))
            ).shape[-1]
        )

    def forward(self, input_tensor):
        receptive_extraction = self.first_layer(input_tensor)
        features = self.feature_block(receptive_extraction)
        return self.final_layer(features)


def weights_init_D(m):
    """initialize weights of the discriminator"""
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0)
    elif class_name.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_G(m):
    """initialize weights of the generator"""
    if m.__class__.__name__.find("Conv") != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0)
