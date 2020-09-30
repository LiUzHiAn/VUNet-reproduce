import torch


def update_loss_weights_inplace(loss_config, step):
    # 缓慢增加kl loss的权重
    for weight_dict in loss_config.values():
        if "start_ramp_it" in weight_dict:
            if step < weight_dict["start_ramp_it"]:
                weight_dict["weight"] = weight_dict["start_ramp_val"]
            elif step > weight_dict["end_ramp_it"]:
                weight_dict["weight"] = weight_dict["end_ramp_val"]
            else:
                ramp_progress = (step - weight_dict["start_ramp_it"]) / (
                        weight_dict["end_ramp_it"] - weight_dict["start_ramp_it"]
                )
                ramp_diff = weight_dict["end_ramp_val"] - weight_dict["start_ramp_val"]
                weight_dict["weight"] = (
                        ramp_progress * ramp_diff + weight_dict["start_ramp_val"]
                )


def latent_kl(prior_mean, posterior_mean):
    """
    :param prior_mean:
    :param posterior_mean:
    :return:
    """
    kl = 0.5 * torch.pow(prior_mean - posterior_mean, 2)
    kl = torch.sum(kl, dim=[1, 2, 3])
    # kl = torch.mean(kl)

    return kl


def aggregate_kl_loss(prior_means, posterior_means):
    # kl_loss = torch.sum(
    #     torch.cat(
    #         [
    #             latent_kl(p, q).unsqueeze(dim=-1)
    #             for p, q in zip(
    #             list(prior_means.values()), list(posterior_means.values())
    #         )
    #         ],
    #         dim=-1,
    #     ),
    #     dim=-1,
    # )
    kl_stages = []
    for p, q in zip(list(prior_means.values()), list(posterior_means.values())):
        kl_stages.append(latent_kl(p, q).unsqueeze(dim=-1))

    kl_stages = torch.cat(kl_stages, dim=-1)
    kl_loss = torch.sum(kl_stages, dim=-1)
    return kl_loss
