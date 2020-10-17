import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch.nn.parameter import Parameter
from torch import norm_except_dim
import torch
import torch.nn.functional as F
import torch.optim as optim


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h // self.bs, self.bs, w // self.bs, self.bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(n, c * (self.bs ** 2), h // self.bs, w // self.bs)
        return x


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, self.bs, self.bs, c // (self.bs ** 2), h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(n, c // (self.bs ** 2), h * self.bs, w * self.bs)
        return x


class IDAct(nn.Module):
    def forward(self, input):
        return input


"""
weight normalization compatible with tf implementation
"""


class MyWeightNorm2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, init=False):
        # def __init__(self, in_channel, out_channel, kernel_size,bias=True, init=False):
        super(MyWeightNorm2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.init = init

        self.v = Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.g = Parameter(torch.empty(out_channels))
        if bias:
            self.b = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.v.data, mean=0.0, std=0.05)
        torch.nn.init.constant_(self.g.data, val=1)
        if self.bias:
            torch.nn.init.constant_(self.b.data, val=0)

    def forward(self, input):
        v_norm = norm_except_dim(self.v, 2, dim=0)  # l2 norm, i.e. ||v||
        v_divide_v_norm = self.v / v_norm
        t = F.conv2d(input, v_divide_v_norm, None, stride=self.stride, padding=self.padding)
        if not self.init:
            std, mean = torch.std_mean(t, dim=[0, 2, 3])
            self.g.data = 1 / (std + 1e-10)
            self.b.data = -mean * self.g.data

            self.init = True

        out = self.g.view([1, self.out_channels, 1, 1]) * t + \
              self.b.view([1, self.out_channels, 1, 1])

        return out


class NormConv2d(nn.Module):
    """
    Convolutional layer with l2 weight normalization and learned scaling parameters
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.beta = nn.Parameter(
            torch.zeros([1, out_channels, 1, 1], dtype=torch.float32)
        )
        self.gamma = nn.Parameter(
            torch.ones([1, out_channels, 1, 1], dtype=torch.float32)
        )
        self.conv = weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            name="weight",
        )

    def forward(self, x):
        # weight normalization
        # self.conv.weight = normalize(self.conv.weight., dim=[0, 2, 3])
        out = self.conv(x)
        out = self.gamma * out + self.beta
        return out


class Downsample(nn.Module):
    def __init__(self, channels, out_channels=None, conv_layer=NormConv2d,):
        super().__init__()
        if out_channels == None:
            self.down = conv_layer(
                channels, channels, kernel_size=3, stride=2, padding=1,
            )
        else:
            self.down = conv_layer(
                channels, out_channels, kernel_size=3, stride=2, padding=1,
            )

    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, subpixel=True, conv_layer=NormConv2d, ):
        super().__init__()
        if subpixel:
            self.up = conv_layer(in_channels, 4 * out_channels, 3, padding=1, )
            self.op2 = DepthToSpace(block_size=2)
        else:
            # channels have to be bisected because of formely concatenated skips connections
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
            self.op2 = IDAct()  # identical act

    def forward(self, x):
        out = self.up(x)
        out = self.op2(out)
        return out


class VUnetResnetBlock(nn.Module):
    """
    Resnet Block as utilized in the vunet publication
    """

    def __init__(
            self,
            out_channels,
            use_skip=False,
            kernel_size=3,
            activate=True,
            conv_layer=NormConv2d,
            gated=False,
            final_act=False,
            dropout_prob=0.0,
    ):
        """

        :param n_channels: The number of output filters
        :param process_skip: the factor between output and input nr of filters
        :param kernel_size:
        :param activate:
        """
        super().__init__()
        self.dout = nn.Dropout(p=dropout_prob)
        self.use_skip = use_skip
        self.gated = gated
        if self.use_skip:
            self.conv2d = conv_layer(
                in_channels=2 * out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,

            )
            self.pre = conv_layer(
                in_channels=out_channels, out_channels=out_channels, kernel_size=1,
            )
        else:
            self.conv2d = conv_layer(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,

            )

        if self.gated:
            self.conv2d2 = conv_layer(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,

            )
            self.dout2 = nn.Dropout(p=dropout_prob)
            self.sigm = nn.Sigmoid()
        if activate:
            self.act_fn = nn.LeakyReLU() if final_act else nn.ELU()
        else:
            self.act_fn = IDAct()

    def forward(self, x, a=None):
        x_prc = x

        if self.use_skip:
            assert a is not None
            a = self.act_fn(a)
            a = self.pre(a)  # TODO 输入的a也是没激活的，self.pre 也不带激活函数, 现在加上了激活
            x_prc = torch.cat([x_prc, a], dim=1)

        x_prc = self.act_fn(x_prc)  # 上一层的激活 推迟
        x_prc = self.dout(x_prc)
        x_prc = self.conv2d(x_prc)

        if self.gated:
            x_prc = self.act_fn(x_prc)
            x_prc = self.dout(x_prc)
            x_prc = self.conv2d2(x_prc)
            a, b = torch.split(x_prc, 2, 1)
            x_prc = a * self.sigm(b)

        return x + x_prc  # 这一层没带激活的输出


if __name__ == '__main__':
    module = MyWeightNorm2d(in_channels=3, out_channels=10, kernel_size=3, padding=1)

    out1 = module(torch.rand(4, 3, 16, 16))  # 这里的结果是 v*x/||v||
    loss1 = torch.sum((torch.ones_like(out1) - out1) ** 2 + out1, dim=[0, 1, 2, 3])

    # module_dict = module.state_dict()
    # torch.save(module_dict, "./module.pt")
    #
    # module = MyWeightNorm2d(in_channels=3, out_channels=10, kernel_size=3, padding = 1, init = True)
    # module_dict_restored = torch.load("./module.pt")
    # module.load_state_dict(module_dict_restored)

    # optimizer = optim.Adam(module.parameters(), lr=1e-1, betas=(0.5, 0.9))
    # optimizer.zero_grad()
    # loss1.backward()
    # optimizer.step()
    #
    # repeat again
    out2 = module(torch.rand(4, 3, 16, 16))
    loss2 = torch.sum((torch.ones_like(out2) - out2) ** 2 + out2, dim=[0, 1, 2, 3])
    loss2.backward()
