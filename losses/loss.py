import torch.nn as nn
from torchvision.models import vgg19
import torchvision.transforms as transforms
import torch
from collections import namedtuple
from losses.utils import aggregate_kl_loss

VGG19_OUTPUT = namedtuple(
    "VGG19_OUTPUT", ["input", "relu1_2", "relu2_2", "relu3_2", "relu4_2", "relu5_2"],
)
VGG19_TARGET_LAYERS = {
    "3": "relu1_2",
    "8": "relu2_2",
    "13": "relu3_2",
    "22": "relu4_2",
    "31": "relu5_2",
}


def scale_img(x):
    """
    Scale a input image with range [-1,1] into range [0,1]
    :param x:
    :return:
    """
    # ma = torch.max(x)
    # mi = torch.min(x)
    out = (x + 1.0) / 2.0
    out = torch.clamp(out, 0.0, 1.0)
    return out


class _PerceptualVGG(nn.Module):
    def __init__(self, vgg_model):
        super(_PerceptualVGG, self).__init__()
        self.vgg_model = vgg_model
        self.vgg_lyrs = self.vgg_model.features

        self.input_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )

    def forward(self, x):
        out = {"input": x}
        x = scale_img(x)
        # normalize appropriate for vgg
        x = torch.stack([self.input_transform(elem) for elem in torch.unbind(x)])

        with torch.no_grad():
            for module_num, submodule in self.vgg_lyrs._modules.items():
                if module_num in VGG19_TARGET_LAYERS:
                    x = submodule(x)
                    out[VGG19_TARGET_LAYERS[module_num]] = x
                else:
                    x = submodule(x)

        return VGG19_OUTPUT(**out)


class PerceptualLoss(nn.Module):
    def __init__(self, vgg19_feats_weights, use_gram=True, gram_feats_weigths=None):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg19(pretrained=True).eval()

        self.custome_vgg = _PerceptualVGG(self.vgg)
        self.vgg19_feats_weights = vgg19_feats_weights

        assert len(vgg19_feats_weights) == len(VGG19_TARGET_LAYERS) + 1
        if use_gram:
            self.grammer = GramLoss()
            self.gram_feats_weigths = gram_feats_weigths
        else:
            self.grammer = None
            self.self.gram_feats_weigths = None

    def forward(self, pred, target):
        pred_feats = self.custome_vgg(pred)
        target_feats = self.custome_vgg(target)

        vgg_loss = []
        for i, (pred_f, target_f) in enumerate(zip(pred_feats, target_feats)):
            vgg_loss.append(self.vgg19_feats_weights[i] * \
                            torch.mean(torch.abs(pred_f - target_f), dim=[1, 2, 3]).unsqueeze(dim=-1))
        vgg_loss = torch.cat(vgg_loss, dim=-1)

        if self.grammer is not None:
            gram_loss = []
            for i, (pred_f, target_f) in enumerate(zip(pred_feats, target_feats)):
                gram_loss.append(self.gram_feats_weigths[i] * self.grammer(pred_f, target_f).unsqueeze(dim=-1))
            gram_loss = torch.cat(gram_loss, dim=-1)

        if self.grammer is not None:
            loss = vgg_loss + gram_loss
        else:
            loss = vgg_loss

        loss = torch.sum(loss, dim=-1)
        return loss


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, prior_means, posterior_means):
        return aggregate_kl_loss(prior_means, posterior_means)


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


if __name__ == '__main__':
    vgg19_losser = PerceptualLoss(vgg19_feats_weights=[1, 1, 1, 1, 1, 1],
                                  use_gram=True,
                                  gram_feats_weigths=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    pred = torch.randn(4, 3, 224, 224)
    target = torch.randn(4, 3, 224, 224)
    vgg19_loss = vgg19_losser(pred, target)
    print(vgg19_loss)

    # grammer = GramLoss()
    # res = grammer(torch.ones(2, 3, 128, 128), torch.ones(2, 3, 128, 128))
    # print(res.size(), res)
