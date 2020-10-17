from torch.utils.data import Dataset, DataLoader
import os
import pickle
import cv2
import numpy
from datasets.utils import keypoints2stickman, normalize
from datasets.keypoints_mode import OPEN_POSE18
from edflow.data.util import adjust_support
from PIL import Image


class MyDeepFashion(Dataset):
    def __init__(self, dpf_index_filepath, is_train=True, spatial_size=256):
        super(MyDeepFashion, self).__init__()
        self.index_filepath = dpf_index_filepath
        self.root_path = os.path.dirname(dpf_index_filepath)
        self.is_train = is_train
        self.spatial_size = spatial_size

        with open(self.index_filepath, 'rb') as f:
            self.index = pickle.load(f)

        self.sample_idx = self.prepare_sample_idx(is_train)
        # self.sample_idx2 = self.prepare_sample_idx(False)
        pass

    @property
    def MyRNG(self):
        current_pid = os.getpid()
        if getattr(self, '_initpid', None) != current_pid:
            self._initpid = current_pid
            self._prng = numpy.random.RandomState()
        return self._prng

    def joints_valid(self, joints):
        return False if numpy.sum(joints < 0) > 0 else True

    def prepare_sample_idx(self, is_train):
        total_samp_num = len(self.index['train'])

        # required_points = self.index['joint_order']  # a more strict samples filter

        required_points = ["lshoulder", "rshoulder", "lhip", "rhip"]
        required_points_indices = [self.index['joint_order'].index(point) for point in required_points]

        sample_idx = [i for i in range(total_samp_num) if
                      self.index['train'][i] == is_train and
                      self.joints_valid(self.index['joints'][i][required_points_indices])]
        return sample_idx

    def __len__(self):
        return len(self.sample_idx)

    def __getitem__(self, item):
        idx = self.sample_idx[item]
        img = numpy.asarray(Image.open(self.root_path + '/' + self.index['imgs'][idx]))
        img = adjust_support(img, '-1->1', '0->255')

        keypoints = self.index['joints'][idx] * numpy.array(self.spatial_size)[None]
        stickman = keypoints2stickman(keypoints, spatial_size=self.spatial_size,
                                      keypoints_mode=OPEN_POSE18)
        stickman = adjust_support(stickman, '-1->1', '0->255')

        # idx2 = self.sample_idx[self.MyRNG.choice(len(self.sample_idx))]
        # img2 = numpy.array(cv2.imread(self.root_path + '/' + self.index['imgs'][idx2]))
        # img2 = adjust_support(img2, '-1->1', '0->255')
        img2 = numpy.copy(img)

        im_norm, stickman_norm = normalize(img, keypoints, stickman, self.index["joint_order"], 2)

        return {'appearance': im_norm, 'stickman': stickman, 'target': img2}
        # return {'appearance': img, 'stickman': stickman, 'target': img2}

if __name__ == '__main__':
    ds = MyDeepFashion('/home/liuzhian/hdd/datasets/deepfashion/index.p')
    print(len(ds))
    dloader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)
    dataiter = iter(dloader)

    for i, samp in enumerate(dloader):
        cv2.imshow('target', adjust_support(samp['target'][0].numpy(), '0-255', '-1->1')[:, :, ::-1])
        stickman_img = adjust_support(samp['stickman'][0].numpy(), '0->255', '-1->1')
        cv2.imshow('stickman', stickman_img[:, :, ::-1])
        cv2.waitKey(0)
