from datasets.MyDataset import MyDeepFashion
from models.vunet import VUnet
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import yaml
from tensorboardX import SummaryWriter
from edflow.data.util import adjust_support

from losses.loss import (
    update_loss_weights_inplace,
    update_lr_dynamically,
    L1LossInstances,
    aggregate_kl_loss,
    VGGPerceptualLossInstances
)

from utils import np2pt, pt2np
import numpy as np
import cv2
import os
from shutil import copyfile

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def check_then_mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


class MyTrain(object):
    def __init__(self):
        self.config = yaml.safe_load(open('./hyper-parameters.yaml'))
        self.log_dir = self.config["log_dir"]
        self.ckpt_dir = self.config["ckpt_dir"]
        self.exp_name = self.config["exp_name"]

        # prepare logging and ckpt dir
        check_then_mkdir(self.config["log_dir"])
        check_then_mkdir(self.config["ckpt_dir"])

        # copy current experiment hyper-parameters settings
        check_then_mkdir(os.path.join(self.log_dir, self.exp_name))
        copyfile("./hyper-parameters.yaml",
                 os.path.join(self.log_dir, self.exp_name, "hyper-parameters.yaml"))

        self.model = VUnet(self.config)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])

        self.L1LossInstances = L1LossInstances()
        self.vggPerceptualLossInstances = VGGPerceptualLossInstances(
            self.config["losses"]["perceptual"], use_gram=False
        )

        if torch.cuda.is_available():
            self.model.cuda()
            self.L1LossInstances.cuda()
            self.vggPerceptualLossInstances.cuda()

        self.TrainDataset = MyDeepFashion('/home/liuzhian/hdd/datasets/deepfashion/index.p')
        self.TestDataset = MyDeepFashion('/home/liuzhian/hdd/datasets/deepfashion/index.p', is_train=False)

        self.TrainLoader = DataLoader(self.TrainDataset, num_workers=0, batch_size=self.config['batch_size'],
                                      shuffle=True)
        self.TestLoader = DataLoader(self.TestDataset, num_workers=0, batch_size=self.config['batch_size'],
                                     shuffle=False)

        self.summary = SummaryWriter(logdir=os.path.join(self.log_dir, self.exp_name))
        self.last_epoch = 0

    def criterion(self, inputs, predictions):
        '''
        Combines all losses with weights as defined in the config. Add new losses here.
        '''
        # update kl weights
        update_loss_weights_inplace(self.config["losses"], self.global_step)

        # calculate losses
        instance_losses = {}

        if "color_L1" in self.config["losses"].keys():
            instance_losses["color_L1"] = self.L1LossInstances(
                predictions["image"], inputs["target"]
            )

        if "color_gradient" in self.config["losses"].keys():
            instance_losses["color_gradient"] = self.L1LossInstances(
                torch.abs(predictions["image"][..., 1:] - predictions["image"][..., :-1]),
                torch.abs(inputs["target"][..., 1:] - inputs["target"][..., :-1]), ) + self.L1LossInstances(
                torch.abs(
                    predictions["image"][..., 1:, :] - predictions["image"][..., :-1, :]
                ),
                torch.abs(inputs["target"][..., 1:, :] - inputs["target"][..., :-1, :]),
            )

        if "KL" in self.config["losses"].keys() and "q_means" in predictions:
            instance_losses["KL"] = aggregate_kl_loss(
                predictions["q_means"], predictions["p_means"]
            )

        if "perceptual" in self.config["losses"]:
            instance_losses["perceptual"] = self.vggPerceptualLossInstances(
                predictions["image"], inputs["target"]
            )

        instance_losses["total"] = sum(
            [
                self.config["losses"][key]["weight"] * instance_losses[key]
                for key in instance_losses.keys()
            ]
        )

        # reduce to batch granularity
        batch_losses = {k: v.mean() for k, v in instance_losses.items()}

        losses = dict(instances=instance_losses, batch=batch_losses)

        return losses

    def save(self, checkpoint_path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.last_epoch = int((checkpoint_path.split('-')[-1]).split('.')[0]) + 1

    def train(self):
        self.global_step = 0
        if not hasattr(self, '_model_is_cuda'):
            if torch.cuda.is_available():
                self.model.cuda()
                self._model_is_cuda = True
            else:
                self._model_is_cuda = False

        for epoch in range(self.last_epoch, self.last_epoch + self.config["num_epochs"]):
            running_loss = 0
            for i, data in enumerate(self.TrainLoader, 0):
                self.global_step = epoch * len(self.TrainLoader) + i
                stickman = np2pt(data['stickman'])
                appearance = np2pt(data['appearance'])
                target = np2pt(data['target'])

                inputs = {}
                inputs['pose'] = stickman
                inputs['appearance'] = appearance
                inputs['target'] = target

                predictions = self.model(inputs)
                predictions = {'image': predictions}
                predictions.update(self.model.saved_tensors)

                losses = self.criterion(inputs, predictions)
                loss = losses['batch']['total']
                running_loss += loss.item()

                # update lr in linear
                cur_lr = update_lr_dynamically(self.global_step,
                                               self.config["losses"]["KL"]["start_ramp_it"],
                                               self.config["losses"]["KL"]["end_ramp_it"],
                                               self.config["lr"], 1e-6)
                for g in self.optimizer.param_groups:
                    g['lr'] = cur_lr

                if i % 20 == 19:
                    print('[epoch: %d, sample: %5d] loss %5f' % (epoch, i, running_loss / 19))
                    running_loss = 0
                    self.summary.add_scalar("train_loss/total_loss", loss, global_step=self.global_step + 1)
                    self.summary.add_scalar("train_loss/color_L1_loss", losses['batch']['color_L1'],
                                            global_step=self.global_step + 1)
                    self.summary.add_scalar("train_loss/color_gradient_loss", losses['batch']['color_gradient'],
                                            global_step=self.global_step + 1)
                    self.summary.add_scalar("train_loss/KL_loss", losses['batch']['KL'],
                                            global_step=self.global_step + 1)
                    self.summary.add_scalar("train_loss/perceptual_loss", losses['batch']['perceptual'],
                                            global_step=self.global_step + 1)

                    target_img = adjust_support(data['target'][0].numpy(), '0->1', '-1->1')
                    stickman_img = adjust_support(data['stickman'][0].numpy(), '0->1', '-1->1')
                    appearance_img = adjust_support(data['appearance'][0].numpy(), '0->1', '-1->1')
                    prediction_img = pt2np(predictions['image'])
                    prediction_img = adjust_support(prediction_img[0], '0->1', '-1->1')
                    self.summary.add_image("train_img/stickman", np.transpose(stickman_img, axes=(2, 0, 1)),
                                           global_step=self.global_step + 1)
                    self.summary.add_image("train_img/target", np.transpose(target_img, axes=(2, 0, 1)),
                                           global_step=self.global_step + 1)
                    self.summary.add_image("train_img/prediction",
                                           np.transpose(prediction_img, axes=(2, 0, 1)),
                                           global_step=self.global_step + 1)
                    # self.summary.add_image("train_img/appearance",
                    #                        np.transpose(appearance_img, axes=(2, 0, 1)),
                    #                        global_step=self.global_step)

                    self.summary.add_scalar("train_loss/lr", cur_lr, global_step=self.global_step + 1)
                    self.summary.add_scalar("train_loss/kl_weight", self.config["losses"]["KL"]["weight"],
                                            global_step=self.global_step + 1)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if epoch % 5 == 4:
                check_then_mkdir(os.path.join(self.ckpt_dir, self.exp_name))
                self.save(os.path.join(self.ckpt_dir, self.exp_name, "ckpt-epoch-%d.pth" % epoch))

    def test(self):
        for i, data in enumerate(self.TestLoader, 0):
            inputs = {}
            inputs['pose'] = np2pt(data['stickman'])
            inputs['appearance'] = np2pt(data['appearance'])
            inputs['target'] = np2pt(data['target'])

            with torch.no_grad():
                # predictions = self.model(inputs, mode="appearance_transfer")
                predictions = self.model.eval()(inputs, mode="sample_appearance")

                stickman_img = adjust_support(data['stickman'][0].numpy(), '0->255', '-1->1')
                appearance_img = adjust_support(data['appearance'][0].numpy(), '0->255', '-1->1')

                prediction_img = pt2np(predictions)
                prediction_img = adjust_support(prediction_img[0], '0->255', '-1->1')

                cv2.imshow('stickman', stickman_img[:, :, ::-1])
                cv2.imshow('appearance', appearance_img[:, :, ::-1])
                cv2.imshow('prediction', prediction_img[:, :, ::-1])
                cv2.waitKey(0)


if __name__ == '__main__':
    myt = MyTrain()
    # myt.restore(os.path.join(myt.ckpt_dir, myt.exp_name, 'myckpt-epoch-4.pth'))

    myt.train()
    # myt.test()
