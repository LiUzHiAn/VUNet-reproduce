import torch
from collections import namedtuple
from torchvision.models import vgg19
from torchvision import transforms
from typing import Tuple
import torch.nn as nn


def update_loss_weights_inplace(loss_config, step):
    # 缓慢增加kl loss的权重
    for weight_dict in loss_config.values():
        if "end_ramp_it" in weight_dict:
            if step < weight_dict["end_ramp_it"] // 2:
                weight_dict["weight"] = weight_dict["start_ramp_val"]
            elif step > weight_dict["end_ramp_it"] // 4 * 3:
                weight_dict["weight"] = weight_dict["end_ramp_val"]
            else:
                ramp_progress = (step - weight_dict["end_ramp_it"] // 2) / (
                        weight_dict["end_ramp_it"] // 4 * 3 - weight_dict["end_ramp_it"] // 2
                )
                ramp_diff = weight_dict["end_ramp_val"] - weight_dict["start_ramp_val"]
                weight_dict["weight"] = (
                        ramp_progress * ramp_diff + weight_dict["start_ramp_val"]
                )


def update_lr_dynamically(step,
                          start, end,
                          start_value, end_value, ):
    """linear from (a, alpha) to (b, beta), i.e.
        (beta - alpha)/(b - a) * (x - a) + alpha"""
    if step <= start:
        return start_value
    elif step >= end:
        return end_value
    linear = (
            (end_value - start_value) / (end - start) *
            (step - start) +
            start_value
    )
    return linear


def latent_kl(prior_mean, posterior_mean):
    """
    :param prior_mean:
    :param posterior_mean:
    :return:
    """
    kl = 0.5 * torch.pow(prior_mean - posterior_mean, 2)
    kl = torch.sum(kl, dim=[1, 2, 3])

    return kl


def aggregate_kl_loss(prior_means, posterior_means):
    kl_stages = []
    for p, q in zip(list(prior_means.values()), list(posterior_means.values())):
        kl_stages.append(latent_kl(p, q).unsqueeze(dim=-1))

    kl_stages = torch.cat(kl_stages, dim=-1)
    kl_loss = torch.sum(kl_stages, dim=-1)
    return kl_loss


class L1LossInstances(torch.nn.L1Loss):
    """L1Loss, which reduces to instances of the batch
    """

    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        super_result = super().forward(image, target)
        reduced = super_result.mean(axis=(1, 2, 3))
        return reduced


VGGTargetLayers = {
    "3": "relu1_2",
    "8": "relu2_2",
    "13": "relu3_2",
    "22": "relu4_2",
    "31": "relu5_2",
}


class GramLoss(nn.Module):
    def __init__(self):
        super(GramLoss, self).__init__()
        self.grammer = _Gram()

    def forward(self, pred, target):
        pred_gram = self.grammer(pred)
        target_gram = self.grammer(target)
        diff = torch.mean(
            torch.abs(pred_gram - target_gram),
            dim=[1, 2]
        )
        return diff


class _Gram(nn.Module):
    def __init__(self):
        super(_Gram, self).__init__()

    def forward(self, input):
        bs, c, h, w = input.size()
        feature = input.view(bs, c, h * w)  # [bs,c,h*w]
        feature_t = feature.permute(0, 2, 1).contiguous()  # [bs,h*w,c]
        gram = torch.bmm(feature, feature_t)  # [bs,c,c]
        gram /= (4.0 * h * w)
        return gram


class VGGPerceptualLossInstances(torch.nn.Module):
    def __init__(self, config, resize=False, use_gram=False):
        super(VGGPerceptualLossInstances, self).__init__()
        self.config = config
        self.vgg_feat_weights = config.setdefault(
            "vgg_feat_weights", (len(VGGTargetLayers) + 1) * [1.0]
        )
        assert len(self.vgg_feat_weights) == len(VGGTargetLayers) + 1

        if use_gram:
            self.grammer = GramLoss()
            self.gram_feat_weigths = config.setdefault(
                "gram_feat_weights", (len(VGGTargetLayers) + 1) * [0.1]
            )
        else:
            self.grammer = None
            self.gram_feats_weigths = None

        device = "cuda:0"
        self.l1_losser = nn.L1Loss(reduction="none").to(device)
        blocks = []
        blocks.append(vgg19(pretrained=True).to(device).features[:4].eval())
        blocks.append(vgg19(pretrained=True).to(device).features[4:9].eval())
        blocks.append(vgg19(pretrained=True).to(device).features[9:14].eval())
        blocks.append(vgg19(pretrained=True).to(device).features[14:23].eval())
        blocks.append(vgg19(pretrained=True).to(device).features[23:32].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, target):

        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # [-1,1] to [0,1]
        input = torch.clamp((input + 1) / 2, 0, 1)
        target = torch.clamp((target + 1) / 2, 0, 1)

        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        x = input
        y = target
        instance_loss = torch.mean(torch.abs(x - y), dim=[1, 2, 3]) * self.vgg_feat_weights[0]  # l1 loss
        if self.grammer is not None:
            instance_loss += self.grammer(x, y) * self.gram_feat_weigths[0]

        for i in range(len(self.blocks)):
            block = self.blocks[i]
            x = block(x)
            y = block(y)
            instance_loss += self.vgg_feat_weights[i + 1] * torch.mean(torch.abs(x - y), dim=[1, 2, 3])
            if self.grammer is not None:
                instance_loss += self.gram_feat_weigths[i + 1] * self.grammer(x, y)
        return instance_loss


if __name__ == '__main__':
    import yaml

    config = yaml.safe_load(open('../hyper-parameters.yaml'))
    vgg19_losser = VGGPerceptualLossInstances(config, use_gram=True).cuda()
    pred = torch.randn(4, 3, 224, 224).cuda()
    target = torch.randn(4, 3, 224, 224).cuda()
    vgg19_loss = vgg19_losser(pred, target)
    print(vgg19_loss)

    # grammer = GramLoss()
    # res = grammer(torch.ones(2, 3, 128, 128), torch.ones(2, 3, 128, 128))
    # print(res.size(), res)
