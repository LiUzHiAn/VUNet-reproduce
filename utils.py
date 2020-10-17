from itertools import repeat
import pandas as pd
import torch
import numpy as np

from tensorboard.backend.event_processing import event_accumulator
import pathlib
import cv2
from itertools import islice

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def np2pt(array):
    '''Converts a numpy array to torch Tensor. If possible pushes to the
    GPU.'''
    tensor = torch.tensor(array, dtype=torch.float32)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    tensor = tensor.permute(0, 3, 1, 2)
    tensor = tensor.contiguous()
    return tensor


def pt2np(tensor):
    '''Converts a torch Tensor to a numpy array.'''
    array = tensor.detach().cpu().numpy()
    array = np.transpose(array, (0, 2, 3, 1))
    return array


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def extract_tensorboard_imgs(
        path="/home/liuzhian/hdd/code/VUNet-reproduce/log/yw_grad_WN_finalTanh_tfTrainTrick_rightVGG",
        out_dir="/home/liuzhian/hdd/code/VUNet-reproduce/log/tb_extracted"):
    event_acc = event_accumulator.EventAccumulator(
        path=path, size_guidance={'images': 0})
    event_acc.Reload()

    outdir = pathlib.Path(out_dir)
    outdir.mkdir(exist_ok=True, parents=True)

    for tag in event_acc.Tags()['images']:
        events = event_acc.Images(tag)

        tag_name = tag.replace('/', '_')
        if "appearance" in tag_name:
            continue
        dirpath = outdir / tag_name
        dirpath.mkdir(exist_ok=True, parents=True)

        for index, event in islice(enumerate(events), 0, len(events), 5):
            s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
            image = cv2.imdecode(s, cv2.IMREAD_COLOR)
            outpath = dirpath / '{:05}.jpg'.format(index)
            cv2.imwrite(outpath.as_posix(), image)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    extract_tensorboard_imgs()
