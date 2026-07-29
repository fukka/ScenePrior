import torch
import torch.nn as nn
from torch.autograd import Variable

from util import create_penalty_mask, map2tensor, resize_tensor_w_kernel, shave_a2b


# noinspection PyUnresolvedReferences
class GANLoss(nn.Module):
    """D outputs a [0,1] map of size of the input. This map is compared in a pixel-wise manner to 1/0 according to
    whether the input is real (i.e. from the input image) or fake (i.e. from the Generator)"""

    def __init__(self, d_last_layer_size):
        super(GANLoss, self).__init__()
        # The loss function is applied after the pixel-wise comparison to the true label (0/1)
        self.loss = nn.L1Loss(reduction="mean")
        # Make a shape
        d_last_layer_shape = [1, 1, d_last_layer_size, d_last_layer_size]
        # The two possible label maps are pre-prepared
        self.label_tensor_fake = Variable(torch.zeros(d_last_layer_shape).cuda(), requires_grad=False)
        self.label_tensor_real = Variable(torch.ones(d_last_layer_shape).cuda(), requires_grad=False)

    def forward(self, d_last_layer, is_d_input_real):
        # Determine label map according to whether current input to discriminator is real or fake
        label_tensor = self.label_tensor_real if is_d_input_real else self.label_tensor_fake

        # Compute the loss
        return self.loss(d_last_layer, label_tensor)


class DownScaleLoss(nn.Module):
    """Computes the difference between the Generator's downscaling and an ideal (bicubic) downscaling"""

    def __init__(self, scale_factor):
        super(DownScaleLoss, self).__init__()
        self.loss = nn.MSELoss()
        bicubic_k = [
            [
                0.0001373291015625,
                0.0004119873046875,
                -0.0013275146484375,
                -0.0050811767578125,
                -0.0050811767578125,
                -0.0013275146484375,
                0.0004119873046875,
                0.0001373291015625,
            ],
            [
                0.0004119873046875,
                0.0012359619140625,
                -0.0039825439453125,
                -0.0152435302734375,
                -0.0152435302734375,
                -0.0039825439453125,
                0.0012359619140625,
                0.0004119873046875,
            ],
            [
                -0.0013275146484375,
                -0.0039825439453130,
                0.0128326416015625,
                0.0491180419921875,
                0.0491180419921875,
                0.0128326416015625,
                -0.0039825439453125,
                -0.0013275146484375,
            ],
            [
                -0.0050811767578125,
                -0.0152435302734375,
                0.0491180419921875,
                0.1880035400390630,
                0.1880035400390630,
                0.0491180419921875,
                -0.0152435302734375,
                -0.0050811767578125,
            ],
            [
                -0.0050811767578125,
                -0.0152435302734375,
                0.0491180419921875,
                0.1880035400390630,
                0.1880035400390630,
                0.0491180419921875,
                -0.0152435302734375,
                -0.0050811767578125,
            ],
            [
                -0.0013275146484380,
                -0.0039825439453125,
                0.0128326416015625,
                0.0491180419921875,
                0.0491180419921875,
                0.0128326416015625,
                -0.0039825439453125,
                -0.0013275146484375,
            ],
            [
                0.0004119873046875,
                0.0012359619140625,
                -0.0039825439453125,
                -0.0152435302734375,
                -0.0152435302734375,
                -0.0039825439453125,
                0.0012359619140625,
                0.0004119873046875,
            ],
            [
                0.0001373291015625,
                0.0004119873046875,
                -0.0013275146484375,
                -0.0050811767578125,
                -0.0050811767578125,
                -0.0013275146484375,
                0.0004119873046875,
                0.0001373291015625,
            ],
        ]
        self.bicubic_kernel = Variable(torch.Tensor(bicubic_k).cuda(), requires_grad=False)
        self.scale_factor = scale_factor

    def forward(self, g_input, g_output):
        downscaled = resize_tensor_w_kernel(im_t=g_input, k=self.bicubic_kernel, sf=self.scale_factor)
        # Shave the downscaled to fit g_output
        # with torch.no_grad():
        #     print('DownScaleLoss', downscaled.shape, shave_a2b(downscaled, g_output).shape)
        return self.loss(g_output, shave_a2b(downscaled, g_output))


class SumOfWeightsLoss(nn.Module):
    """Encourages the kernel G is imitating to sum to 1"""

    def __init__(self):
        super(SumOfWeightsLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        # with torch.no_grad():
        #     print('SumOfWeightsLoss', torch.ones(1).to(kernel.device).shape, torch.sum(kernel).shape)
        return self.loss(torch.ones(1).to(kernel.device), torch.ones(1).to(kernel.device) * torch.sum(kernel))


class CentralizedLoss(nn.Module):
    """Penalizes distance of center of mass from K's center"""

    def __init__(self, k_size, scale_factor=0.5):
        super(CentralizedLoss, self).__init__()
        self.indices = Variable(torch.arange(0.0, float(k_size)).cuda(), requires_grad=False)
        wanted_center_of_mass = k_size // 2 + 0.5 * (int(1 / scale_factor) - k_size % 2)
        self.center = Variable(
            torch.FloatTensor([wanted_center_of_mass, wanted_center_of_mass]).cuda(), requires_grad=False
        )
        # print('CentralizedLoss k_size', k_size, scale_factor, wanted_center_of_mass, self.center.shape)
        self.loss = nn.MSELoss()

    def forward(self, kernel):
        """Return the loss over the distance of center of mass from kernel center"""
        r_sum, c_sum = torch.sum(kernel, dim=1).reshape(1, -1), torch.sum(kernel, dim=0).reshape(1, -1)
        # with torch.no_grad():
        #     print('CentralizedLoss', torch.stack((torch.matmul(r_sum, self.indices) / torch.sum(kernel),
        #                               torch.matmul(c_sum, self.indices) / torch.sum(kernel))).shape, self.center.shape)
        #     print(kernel.shape, r_sum.shape, c_sum.shape, self.indices.shape)
        return self.loss(
            torch.stack(
                (
                    torch.matmul(r_sum, self.indices) / torch.sum(kernel),
                    torch.matmul(c_sum, self.indices) / torch.sum(kernel),
                )
            ),
            self.center[:, None],
        )


class CenterOfMassAnisotropicLoss(nn.Module):
    """Encourages the kernel to follow a Gaussian-like shape, using the center of mass as the reference, while allowing for anisotropy."""

    def __init__(self, k_size, epsilon=1e-6):
        super(CenterOfMassAnisotropicLoss, self).__init__()
        self.k_size = k_size
        self.epsilon = epsilon  # Tolerance for small numerical differences
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        # Compute the center of mass of the kernel
        center_of_mass = self._compute_center_of_mass(kernel)

        # Compute distances from the center of mass
        distances = self._compute_distances(center_of_mass, kernel.shape[-2:])

        # Flatten the kernel and distances for comparison
        kernel_flat = kernel.view(-1)
        distances_flat = distances.view(-1)

        # Normalize the kernel values to be between 0 and 1
        kernel_flat = (kernel_flat - kernel_flat.min()) / (kernel_flat.max() - kernel_flat.min() + self.epsilon)

        # Sort the kernel values by their distances from the center of mass (smallest distances first)
        sorted_indices = torch.argsort(distances_flat)
        sorted_kernel = kernel_flat[sorted_indices]
        sorted_distances = distances_flat[sorted_indices]

        # Check if values closer to the center of mass are larger than those farther from the center
        diffs = sorted_kernel[1:] - sorted_kernel[:-1]
        # print(diffs)
        violations = (diffs > self.epsilon).float()  # Identify if there's any violation of the rule

        # Loss is proportional to how much the constraint is violated
        total_violations = torch.sum(violations)

        # Return a scaled loss based on the number of violations
        return total_violations / len(diffs)

    def _compute_center_of_mass(self, kernel):
        """Compute the center of mass using row and column sums."""
        # Get the size of the kernel
        k_size = kernel.shape[0]

        # Create the row and column indices
        y_indices = torch.arange(k_size).float().cuda().view(1, -1)  # y-coordinates
        x_indices = torch.arange(k_size).float().cuda().view(1, -1)  # x-coordinates

        # Compute the sum of the kernel along rows and columns
        r_sum = torch.sum(kernel, dim=1).reshape(1, -1)  # Row sums (y direction)
        c_sum = torch.sum(kernel, dim=0).reshape(1, -1)  # Column sums (x direction)

        # Compute the center of mass in the y direction (center_y)
        center_y = torch.sum(r_sum * y_indices) / torch.sum(r_sum)

        # Compute the center of mass in the x direction (center_x)
        center_x = torch.sum(c_sum * x_indices) / torch.sum(c_sum)

        return center_y, center_x

    def _compute_distances(self, center_of_mass, shape):
        """Compute Euclidean distances from the center of mass to each point in the kernel"""
        y, x = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]))
        y, x = y.float().cuda(), x.float().cuda()

        center_y, center_x = center_of_mass
        distances = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        return distances


class BoundariesLoss(nn.Module):
    """Encourages sparsity of the boundaries by penalizing non-zeros far from the center"""

    def __init__(self, k_size):
        super(BoundariesLoss, self).__init__()
        self.mask = map2tensor(create_penalty_mask(k_size, 30))
        self.zero_label = Variable(torch.zeros(1, 1, k_size, k_size).cuda(), requires_grad=False)
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        # with torch.no_grad():
        #     print('BoundariesLoss'),
        #     print(kernel.shape, self.mask.shape, self.zero_label.shape)
        #     print((kernel * self.mask).shape, self.zero_label.shape)
        return self.loss(kernel * self.mask, self.zero_label)


class SparsityLoss(nn.Module):
    """Penalizes small values to encourage sparsity"""

    def __init__(self):
        super(SparsityLoss, self).__init__()
        self.power = 0.2
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        # with torch.no_grad():
        #     print('SparsityLoss', (torch.abs(kernel) ** self.power).shape, torch.zeros_like(kernel).shape)
        return self.loss(torch.abs(kernel) ** self.power, torch.zeros_like(kernel))
