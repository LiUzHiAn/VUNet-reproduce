import torch
from torch import nn
from torch.nn import ModuleDict, ModuleList, Conv2d
from edflow.util import retrieve
from models.modules import (
    VUnetResnetBlock,
    Upsample,
    Downsample,
    NormConv2d,
    SpaceToDepth,
    DepthToSpace,
)
import numpy as np
from torch.distributions import MultivariateNormal
from utils import DEVICE


class VUnetEncoder(nn.Module):
    def __init__(
            self,
            n_stages,
            nf_in=3,
            nf_start=64,
            nf_max=128,
            n_rnb=2,
            conv_layer=NormConv2d,
            dropout_prob=0.0,
    ):
        super().__init__()
        self.in_op = conv_layer(nf_in, nf_start, kernel_size=1)
        nf = nf_start
        self.blocks = ModuleDict()
        self.downs = ModuleDict()
        self.n_rnb = n_rnb
        self.n_stages = n_stages
        for i_s in range(self.n_stages):
            # prepare resnet blocks per stage
            if i_s > 0:
                self.downs.update(
                    {
                        f"s{i_s + 1}": Downsample(
                            nf, min(2 * nf, nf_max), conv_layer=conv_layer
                        )
                    }
                )
                nf = min(2 * nf, nf_max)

            for ir in range(self.n_rnb):
                stage = f"s{i_s + 1}_{ir + 1}"
                self.blocks.update(
                    {
                        stage: VUnetResnetBlock(
                            nf, conv_layer=conv_layer, dropout_prob=dropout_prob
                        )
                    }
                )

    def forward(self, x):
        out = {}
        h = self.in_op(x)
        for ir in range(self.n_rnb):
            h = self.blocks[f"s1_{ir + 1}"](h)
            out[f"s1_{ir + 1}"] = h

        for i_s in range(1, self.n_stages):

            h = self.downs[f"s{i_s + 1}"](h)  # TODO: ResBlock的输出没激活，而Downsample的开始也没有激活

            for ir in range(self.n_rnb):
                stage = f"s{i_s + 1}_{ir + 1}"
                h = self.blocks[stage](h)
                out[stage] = h

        return out


class ZConverter(nn.Module):
    def __init__(self, n_stages, nf, device, conv_layer=NormConv2d, dropout_prob=0.0):
        super().__init__()
        self.n_stages = n_stages
        self.device = device
        self.blocks = ModuleList()
        for i in range(3):  # 连续3个res block
            self.blocks.append(
                VUnetResnetBlock(
                    nf, use_skip=True, conv_layer=conv_layer, dropout_prob=dropout_prob
                )
            )
        self.conv1x1 = conv_layer(nf, nf, 1)
        self.up = Upsample(in_channels=nf, out_channels=nf, conv_layer=conv_layer)
        self.channel_norm = conv_layer(2 * nf, nf, 1)

        self.d2s = DepthToSpace(block_size=2)
        self.s2d = SpaceToDepth(block_size=2)

    def forward(self, x_f):
        params = {}
        zs = {}
        h = self.conv1x1(x_f[f"s{self.n_stages}_2"])
        # 倒数两层用来做 distribution inference
        for n, i_s in enumerate(range(self.n_stages, self.n_stages - 2, -1)):
            stage = f"s{i_s}"

            spatial_size = x_f[stage + "_2"].shape[-1]
            spatial_stage = "%dby%d" % (spatial_size, spatial_size)

            h = self.blocks[2 * n](h, x_f[stage + "_2"])

            params[spatial_stage] = h  # 后验的参数
            z = self._latent_sample(params[spatial_stage])  # 传入的是均值，返回采样后的值
            zs[spatial_stage] = z
            # post
            if n == 0:
                gz = torch.cat([x_f[stage + "_1"], z], dim=1)
                gz = self.channel_norm(gz)
                h = self.blocks[2 * n + 1](h, gz)
                h = self.up(h)

        return params, zs

    def _latent_sample(self, mean):
        # 标准多元高斯分布采样
        normal_sample = torch.randn(mean.size()).to(self.device)
        return mean + normal_sample

    # def _latent_sample(self, mean):
    #     sample_mean = torch.squeeze(torch.squeeze(mean, dim=-1), dim=-1)
    #
    #     sampled = MultivariateNormal(
    #         loc=torch.zeros_like(sample_mean, device=self.device),
    #         covariance_matrix=torch.eye(sample_mean.shape[-1], device=self.device),
    #     ).sample()  # 标准多元高斯分布采样
    #     # sigma * epsilon + mu
    #     return (sampled + sample_mean).unsqueeze(dim=-1).unsqueeze(dim=-1)


class VUnetDecoder(nn.Module):
    def __init__(
            self,
            n_stages,
            nf=128,
            nf_out=3,
            n_rnb=2,
            conv_layer=NormConv2d,
            spatial_size=256,
            final_act=True,
            dropout_prob=0.0,
    ):
        super().__init__()

        self.final_act = final_act
        self.blocks = ModuleDict()
        self.ups = ModuleDict()
        self.n_stages = n_stages
        self.n_rnb = n_rnb
        for i_s in range(self.n_stages - 2, 0, -1):
            # for final stage, bisect number of filters
            if i_s == 1:
                # upsampling operations
                self.ups.update(
                    {
                        f"s{i_s + 1}": Upsample(
                            in_channels=nf, out_channels=nf // 2, conv_layer=conv_layer,
                        )
                    }
                )
                nf = nf // 2
            else:
                # upsampling operations
                self.ups.update(
                    {
                        f"s{i_s + 1}": Upsample(
                            in_channels=nf, out_channels=nf, conv_layer=conv_layer,
                        )
                    }
                )

            # resnet blocks
            for ir in range(self.n_rnb, 0, -1):
                stage = f"s{i_s}_{ir}"
                self.blocks.update(
                    {
                        stage: VUnetResnetBlock(
                            nf,
                            use_skip=True,
                            conv_layer=conv_layer,
                            dropout_prob=dropout_prob,
                        )
                    }
                )

        # final 1x1 convolution
        self.final_layer = conv_layer(nf, nf_out, kernel_size=1)

        # conditionally: set final activation
        # if self.final_act:
        self.final_act = nn.Tanh()

    def forward(self, x, skips):
        """

        Parameters
        ----------
        x : torch.Tensor
            Latent representation to decode.
        skips : dict
            The skip connections of the VUnet

        Returns
        -------
        out : torch.Tensor
            An image as described by :attr:`x` and :attr:`skips`
        """
        out = x
        for i_s in range(self.n_stages - 2, 0, -1):
            out = self.ups[f"s{i_s + 1}"](out)

            for ir in range(self.n_rnb, 0, -1):
                stage = f"s{i_s}_{ir}"
                out = self.blocks[stage](out, skips[stage])

        out = self.final_layer(out)
        if self.final_act:  # final activation
            out = self.final_act(out)
        return out


# IMPORTANT: upsampling always uses the same number of filters in and out, The number changes before in the second resnet block!!
class VUnetBottleneck(nn.Module):
    def __init__(
            self,
            n_stages,
            nf,
            device,
            n_rnb=2,
            n_auto_groups=4,
            conv_layer=NormConv2d,
            dropout_prob=0.0,
    ):
        super().__init__()
        self.device = device  # gpu or cpu
        self.blocks = ModuleDict()
        self.channel_norm = ModuleDict()
        self.conv1x1 = conv_layer(nf, nf, 1)
        self.up = Upsample(in_channels=nf, out_channels=nf, conv_layer=conv_layer)
        self.depth_to_space = DepthToSpace(block_size=2)
        self.space_to_depth = SpaceToDepth(block_size=2)
        self.n_stages = n_stages
        self.n_rnb = n_rnb  # 2
        # number of autoregressively modeled groups
        self.n_auto_groups = n_auto_groups
        for i_s in range(self.n_stages, self.n_stages - 2, -1):  # only 2 lyrs
            self.channel_norm.update({f"s{i_s}": conv_layer(2 * nf, nf, 1)})
            for ir in range(self.n_rnb):
                self.blocks.update(
                    {
                        f"s{i_s}_{ir + 1}": VUnetResnetBlock(
                            nf,
                            use_skip=True,
                            conv_layer=conv_layer,
                            dropout_prob=dropout_prob,
                        )
                    }
                )

        self.auto_blocks = ModuleList()
        # model the autoregressively groups rnb
        for i_a in range(4):
            if i_a < 1:  # 第一层不加skip conn
                self.auto_blocks.append(
                    VUnetResnetBlock(
                        nf, conv_layer=conv_layer, dropout_prob=dropout_prob
                    )
                )
                self.param_converter = conv_layer(4 * nf, nf, kernel_size=1)
            else:
                self.auto_blocks.append(
                    VUnetResnetBlock(
                        nf,
                        use_skip=True,
                        conv_layer=conv_layer,
                        dropout_prob=dropout_prob,
                    )
                )

    def forward(self, x_e, z_post, mode="train"):
        """

        Parameters
        ----------
        x_e : torch.Tensor
            The output from the encoder E_theta 先验encoder的特征
        z_post : torch.Tensor
            The output from the encoder F_phi 后验的采样
        mode : str
            Determines the mode of the bottleneck, must be in
            ["train","appearance_transfer","sample_appearance"]

        Returns
        -------
        h : torch.Tensor
            the output of the last layer of the bottleneck which is
            subsequently used by the decoder.
        posterior_params : torch.Tensor
            The flattened means of the posterior distributions p(z|ŷ,x) of the
            two bottleneck stages.
        prior_params : dict(str: torch.Tensor)
            The flattened means of the prior distributions p(z|ŷ) of the two
            bottleneck stages.
        z_prior : torch.Tensor
            The current samples of the two stages of the prior distributions of
            both two bottleneck stages, flattened.
        """
        p_params = {}
        z_prior = {}
        # 是否使用后验的z
        use_z = mode == "train" or mode == "appearance_transfer"

        h = self.conv1x1(x_e[f"s{self.n_stages}_2"])
        for i_s in range(self.n_stages, self.n_stages - 2, -1):
            stage = f"s{i_s}"
            spatial_size = x_e[stage + "_2"].shape[-1]
            spatial_stage = "%dby%d" % (spatial_size, spatial_size)

            h = self.blocks[stage + "_2"](h, x_e[stage + "_2"])

            if spatial_size == 1:
                p_params[spatial_stage] = h
                # posterior_params[stage] = z_post[stage + "_2"]
                prior_samples = self._latent_sample(p_params[spatial_stage])

                z_prior[spatial_stage] = prior_samples
                # posterior_samples = self._latent_sample(posterior_params[stage])
            else:
                # 是否使用后验的z
                if use_z:
                    z_flat = (
                        self.space_to_depth(z_post[spatial_stage])
                        if z_post[spatial_stage].shape[2] > 1
                        else z_post[spatial_stage]
                    )
                    sec_size = z_flat.shape[1] // 4
                    z_groups = torch.split(
                        z_flat, [sec_size, sec_size, sec_size, sec_size], dim=1
                    )  # 按通道分成4组
                # 先验的参数估计
                param_groups = []
                sample_groups = []

                param_features = self.auto_blocks[0](h)
                param_features = self.space_to_depth(param_features)
                # convert to fit depth  为了通道数匹配
                param_features = self.param_converter(param_features)
                # auto-gressive params inference
                for i_a in range(len(self.auto_blocks)):
                    param_groups.append(param_features)

                    prior_samples = self._latent_sample(param_groups[-1])

                    sample_groups.append(prior_samples)

                    if i_a + 1 < len(self.auto_blocks):  # 0,1,2
                        if use_z:
                            feedback = z_groups[i_a]  # 后验采样的split
                        else:
                            feedback = prior_samples
                        # todo 这里应该是 self.auto_blocks[i_a+1]
                        param_features = self.auto_blocks[i_a + 1](param_features, feedback)

                p_params_stage = self.__merge_groups(param_groups)
                prior_samples = self.__merge_groups(sample_groups)  # 先验的auto gressive采样，d2s
                p_params[spatial_stage] = p_params_stage  # prior params
                z_prior[spatial_stage] = prior_samples

            if use_z:
                z = (
                    self.depth_to_space(z_post[spatial_stage])
                    if z_post[spatial_stage].shape[-1] != h.shape[-1]
                    else z_post[spatial_stage]  # stage s8 will not use depth2space
                )
            else:
                z = prior_samples

            # h = torch.cat([h, z], dim=1)  # todo: bug??? should be  torch.cat([stage_1, z], dim=1)
            # h = self.channel_norm[stage](
            #     h)  # 这里应该命名为 hz=self.channel_norm[stage](h),然后下一行为 h = self.blocks[stage + "_1"](hz, x_e[stage + "_1"])
            # h = self.blocks[stage + "_1"](h, x_e[stage + "_1"])  # todo: bug??? should be  torch.cat([h, gz], dim=1)

            gz = torch.cat([x_e[stage + "_1"], z], dim=1)
            gz = self.channel_norm[stage](gz)
            h = self.blocks[stage + "_1"](h, gz)

            if i_s == self.n_stages:
                h = self.up(h)

        return h, p_params, z_prior  # h还没上采样

    def __split_groups(self, x):
        # split along channel axis
        sec_size = x.shape[1] // 4
        return torch.split(
            self.space_to_depth(x), [sec_size, sec_size, sec_size, sec_size], dim=1,
        )

    def __merge_groups(self, x):
        # merge groups along channel axis
        return self.depth_to_space(torch.cat(x, dim=1))

    def _latent_sample(self, mean):
        # 标准多元高斯分布采样
        normal_sample = torch.randn(mean.size()).to(self.device)
        return mean + normal_sample


class VUnet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        final_act = retrieve(config, "model_pars/final_act", default=False)
        nf_max = retrieve(config, "model_pars/nf_max", default=128)
        nf_start = retrieve(config, "model_pars/nf_start", default=64)
        spatial_size = retrieve(config, "model_pars/spatial_size", default=128)
        dropout_prob = retrieve(config, "model_pars/dropout_prob", default=0.1)
        f_in_channels = retrieve(config, "model_pars/img_channels", default=3)
        e_in_channels = retrieve(config, "model_pars/pose_channels", default=3)

        in_plane_factor = retrieve(config, "model_pars/in_plane_factor", default=2)
        num_crops = retrieve(config, "model_pars/num_crops", default=8)

        self.output_channels = 3

        device = DEVICE

        # define required parameters
        n_stages = 1 + int(np.round(np.log2(spatial_size)))-2
        # if final activation shall be utilized, choose common pytorch convolution as conv layer, else custom Module that follows the original implementation
        conv_layer_type = Conv2d if final_act else NormConv2d

        # image processing encoder to produce the prosterior p( z | x,ŷ )
        # 实际上，输入的只有x
        self.f_phi = VUnetEncoder(
            n_stages=n_stages - in_plane_factor,
            # n_stages=n_stages,
            nf_in=f_in_channels*num_crops,
            # nf_in=f_in_channels,
            nf_start=nf_start,
            nf_max=nf_max,
            conv_layer=conv_layer_type,
            dropout_prob=dropout_prob,
        )

        # stickman processing encoder to produce the prior p(z|ŷ)
        self.e_theta = VUnetEncoder(
            n_stages=n_stages,
            nf_in=e_in_channels,
            nf_start=nf_start,
            nf_max=nf_max,
            conv_layer=conv_layer_type,
            dropout_prob=dropout_prob,
        )

        # zconverter
        self.zc = ZConverter(
            n_stages=n_stages - in_plane_factor,
            # n_stages=n_stages,
            nf=nf_max,
            device=device,
            conv_layer=conv_layer_type,
            dropout_prob=dropout_prob,
        )

        # bottleneck
        self.bottleneck = VUnetBottleneck(
            n_stages=n_stages,
            nf=nf_max,
            device=device,
            conv_layer=conv_layer_type,
            dropout_prob=dropout_prob,
        )

        # decoder
        self.decoder = VUnetDecoder(
            n_stages=n_stages,
            nf=nf_max,
            nf_out=self.output_channels,
            conv_layer=conv_layer_type,
            spatial_size=spatial_size,
            final_act=final_act,
            dropout_prob=dropout_prob,
        )
        self.saved_tensors = None

    def forward(self, inputs, mode="train"):
        '''
        有以下几种用法：
        1. 训练阶段。 pose和appearance是对应的，decoder从后验采样，进行重建
        2. appearance transfer阶段。 此时输入的 pose和appearance不是对应的，
            即我们希望，可以把appearance 迁移到另外的一个pose上去。
            此时，我们需要用到后验的均值作为采样，然后decoder 用采样和pose的特征进行重建。
        3. sample appearance阶段。 此时的输入只有一个pose，我们从先验中采样（每次采样都不一样），
            然后decode用采样和pose的特征进行重建，所以sample多次的输出也不一样。
        '''
        # 后验
        x_f = self.f_phi(inputs['appearance'])
        # sample z  后验的 参数 和 采样
        q_means, zs = self.zc(x_f)

        # 先验
        x_e = self.e_theta(inputs['pose'])

        # with prior and posterior distribution; don't use prior samples within this training
        # x_e-先验 zs-后验的采样
        if mode == "train":
            out_b, p_means, ps = self.bottleneck(x_e, zs, mode)  # h, p_params, z_prior
        elif mode == "appearance_transfer":
            out_b, p_means, ps = self.bottleneck(x_e, q_means, mode)
        elif mode == "sample_appearance":
            out_b, p_means, ps = self.bottleneck(x_e, {}, mode)
        else:
            raise ValueError(
                'The mode of vunet has to be one of ["train",'
                + '"appearance_transfer","sample_appearance"], but is '
                + mode
            )

        # decode, feed in the output of bottleneck and the encoder features
        out_img = self.decoder(out_b, x_e)

        self.saved_tensors = dict(q_means=q_means, p_means=p_means)
        return out_img


if __name__ == '__main__':
    import yaml

    with open("../hyper-parameters.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    vunet = VUnet(config).to(DEVICE)
    x = torch.randn(4, 3, 256, 256).cuda()
    c = torch.randn(4, 3, 256, 256).cuda()
    xn = torch.randn(4, 3 * 8, 64, 64).cuda()
    cn = torch.randn(4, 3 * 8, 64, 64)
    in_dict = dict(x=x, c=c, xn=xn, cn=cn)
    out = vunet(in_dict)
    print(out.size())
    torch.save(vunet.state_dict(), "./vunet.pt")

    vunet_restore = VUnet(config).to(DEVICE)
    """一定记得要把WeightNorm中的init设置为True"""
    # for module in vunet_restore.modules():
    #     if isinstance(module, MyWeightNorm2d):
    #         module.init = True
    model_state_dict_restored = torch.load("./vunet.pt")
    vunet_restore.load_state_dict(model_state_dict_restored)
    vunet_restore.eval()

    # import torch.onnx
    #
    # onnx_path = "onnx_model_name.onnx"
    # torch.onnx.export(model=vunet, args=(stickman, img), f=onnx_path)

    # from tensorboardX import SummaryWriter
    #
    # with SummaryWriter(comment='VUNet') as w:
    #     w.add_graph(vunet, [stickman, img])
