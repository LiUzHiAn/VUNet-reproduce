import torch
import numpy as np
from common import AverageMeter, DEVICE
import torch.nn as nn
from losses.utils import *
from losses.loss import PerceptualLoss, KLLoss, GramLoss
from tensorboardX import SummaryWriter
import time
from datasets.utils import postprocess


class Trainer(object):
    def __init__(self, config, model, optimizer, dataloader, dataloader_val,
                 log_dir, ckpt_path,
                 lr_scheduler=None):
        self.config = config
        self.device = DEVICE
        self.model = model
        self.optimizer = optimizer
        self.data_loader = dataloader
        self.data_loader_val = dataloader_val
        self.ckpt_path = ckpt_path

        self.log_dir = log_dir  # tensorboard 保存路径

        # epoch-based training
        self.len_epoch = len(self.data_loader)

        self.lr_scheduler = lr_scheduler
        self.save_every = 2  # 每2个epoch save一下
        self.log_step = self.len_epoch // 50

        # self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        # 定义各部分损失函数
        self.l1_losser = nn.L1Loss(reduction="none").to(self.device)
        self.vgg_losser = PerceptualLoss(
            vgg19_feats_weights=self.config["losses"]["vgg"]["vgg_feat_weights"],
            use_gram=True,
            gram_feats_weigths=self.config["losses"]["gram"]["gram_weights"]
        ).to(self.device)
        self.kl_losser = KLLoss().to(self.device)

        self.writer = SummaryWriter(logdir=self.log_dir)

        self.log_timer = AverageMeter()

    def train(self):
        for epoch in range(self.config["num_epochs"]):
            self._train_epoch(epoch)
            # self._val_epoch(epoch)

    def _train_epoch(self, epoch):
        """
           Training logic for an epoch
           :param epoch: Integer, current training epoch.
           :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        _tic = time.time()
        for batch_idx, data in enumerate(self.data_loader):
            keyponits_coordinates = data["keyponits_coordinates"]
            im = data["im"]
            stickman = data["stickman"]
            im_norm = data["im_norm"]
            stickman_norm = data["stickman_norm"]

            input_to_model = {"x": im.to(self.device),
                              "c": stickman.to(self.device),
                              "xn": im_norm.to(self.device),
                              "cn": stickman_norm.to(self.device)}

            pred = self.model(input_to_model)
            model_output = {"pred": pred}
            model_output.update(self.model.saved_tensors)

            batch_losses = {}
            # l1 loss
            if "l1" in self.config["losses"].keys():
                batch_losses["l1"] = torch.mean(
                    self.l1_losser(
                        model_output["pred"], input_to_model["x"]
                    ), dim=[1, 2, 3])
            # KL 散度
            if "KL" in self.config["losses"].keys() and "q_means" in model_output.keys():
                batch_losses["KL"] = self.kl_losser(
                    model_output["p_means"], model_output["q_means"],
                )
            # vgg loss
            if "vgg" in self.config["losses"]:
                batch_losses["vgg"] = self.vgg_losser(
                    model_output["pred"], input_to_model["x"]
                )

            # update kl weights
            update_loss_weights_inplace(self.config["losses"], 1 + batch_idx + self.len_epoch * epoch)
            # 损失函数总和
            batch_overall_loss = 0
            for key in batch_losses.keys():
                batch_overall_loss += self.config["losses"][key]["weight"] * batch_losses[key]
            batch_overall_loss = torch.mean(batch_overall_loss)
            self.optimizer.zero_grad()
            batch_overall_loss.backward()
            self.optimizer.step()

            if batch_idx % self.log_step == 0:
                self.log_timer.update(time.time() - _tic)
                _tic = time.time()
                print("Train Epoch: %d, loss:%.4f, time used: %.4f (per %d batches)" % (
                    epoch,
                    batch_overall_loss,
                    self.log_timer.avg,
                    self.log_step))
                self.writer.add_scalar("train_loss/kl", batch_losses["KL"][0],
                                       global_step=batch_idx + self.len_epoch * epoch)
                self.writer.add_scalar("train_loss/vgg", batch_losses["vgg"][0],
                                       global_step=batch_idx + self.len_epoch * epoch)
                self.writer.add_scalar("train_loss/l1", batch_losses["l1"][0],
                                       global_step=batch_idx + self.len_epoch * epoch)
                self.writer.add_image("train_image/pose", postprocess(stickman[0].cpu().numpy()),
                                      global_step=batch_idx + self.len_epoch * epoch)
                self.writer.add_image("train_image/target", postprocess(im[0].cpu().numpy()),
                                      global_step=batch_idx + self.len_epoch * epoch)
                self.writer.add_image("train_image/pred", postprocess(model_output["pred"][0].cpu().detach().numpy()),
                                      global_step=batch_idx + self.len_epoch * epoch)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        print("============Epoch %d done! Time used: %.4f================" % (
            epoch, self.log_timer.avg * self.len_epoch / self.log_step))

        if epoch % self.save_every:
            self.save(self.ckpt_path + "%d" % epoch)
            print("============Ckpt saved at %s!================" % (self.ckpt_path + "%d" % epoch))

    def _val_epoch(self, epoch):
        self.model.eval()

        _tic = time.time()
        l1_loss = AverageMeter()
        kl_loss = AverageMeter()
        vgg_loss = AverageMeter()
        overall_loss = AverageMeter()

        # rnd_verbose_batch = np.random.randint(0, len(self.data_loader_val))
        rnd_verbose_batch = 0
        rnd_output_verbose = {}
        for batch_idx, data in enumerate(self.data_loader_val):
            keyponits_coordinates = data["keyponits_coordinates"]
            im = data["im"]
            stickman = data["stickman"]
            im_norm = data["im_norm"]
            stickman_norm = data["stickman_norm"]

            input_to_model = {"x": im.to(self.device),
                              "c": stickman.to(self.device),
                              "xn": im_norm.to(self.device),
                              "cn": stickman_norm.to(self.device)}

            pred = self.model(input_to_model, mode='train')
            model_output = {"pred": pred}
            model_output.update(self.model.saved_tensors)

            if batch_idx == rnd_verbose_batch:
                rnd_output_verbose = {"pose": stickman, "target": im, "pred": pred}
            batch_losses = {}
            # l1 loss
            if "l1" in self.config["losses"].keys():
                batch_losses["l1"] = torch.mean(
                    self.l1_losser(
                        model_output["pred"], input_to_model["x"]
                    ), dim=[1, 2, 3]
                )
                l1_loss.update(batch_losses["l1"].mean())
            # KL 散度
            self.config["losses"]["KL"]["weight"] = 1
            if "KL" in self.config["losses"].keys() and "q_means" in model_output.keys():
                batch_losses["KL"] = self.kl_losser(
                    model_output["p_means"], model_output["q_means"],
                )
                kl_loss.update(batch_losses["KL"].mean())
            # vgg loss
            if "vgg" in self.config["losses"]:
                batch_losses["vgg"] = self.vgg_losser(
                    model_output["pred"], input_to_model["x"]
                )
                vgg_loss.update(batch_losses["vgg"].mean())
            # 损失函数总和
            batch_overall_loss = 0
            for key in batch_losses.keys():
                batch_overall_loss += self.config["losses"][key]["weight"] * batch_losses[key]
                overall_loss.update(batch_overall_loss.mean())

        self.writer.add_scalar("eval_loss/kl", kl_loss.avg, global_step=epoch)
        self.writer.add_scalar("eval_loss/vgg", vgg_loss.avg, global_step=epoch)
        self.writer.add_scalar("eval_loss/l1", l1_loss.avg, global_step=epoch)
        self.writer.add_image("eval_image/pose", postprocess(rnd_output_verbose["pose"][0].cpu().numpy()),
                              global_step=epoch)
        self.writer.add_image("eval_image/target", postprocess(rnd_output_verbose["target"][0].cpu().numpy()),
                              global_step=epoch)
        self.writer.add_image("eval_image/pred", postprocess(rnd_output_verbose["pred"][0].cpu().detach().numpy()),
                              global_step=epoch)

        _toc = time.time()
        print("Eval Epoch: %d, loss:%.4f, time used: %.4f" % (
            epoch,
            overall_loss.avg,
            _toc - _tic))

    def save(self, checkpoint_path):
        '''Defines how to save the model. Called everytime a log output is
        produced and when ``ctrl-c``, i.e. ``KeybordInterrupt`` is invoked.
        '''
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint_path):
        '''Defines how to load a model. Called when running edflow with the
        ``-p`` or ``-c`` parameter.'''
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
