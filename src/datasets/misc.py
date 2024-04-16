import numpy as np
import random
from PIL import Image

from .augmentations import apply_augment


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img


# 自定义添加椒盐噪声的 transform
class PepperNoise(object):
    """增加椒盐噪声
    Args:
        snr (float): Signal Noise Rate，衡量噪声的比例，图像中正常像素占全部像素的占比。
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr=0.9, p=0.5):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p

    # transform 会调用该方法
    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        # 如果随机概率小于 seld.p，则执行 transform
        if random.uniform(0, 1) < self.p:
            # 把 image 转为 array
            img_ = np.array(img).copy()
            # 获得 shape
            h, w, c = img_.shape
            # 信噪比
            signal_pct = self.snr
            # 椒盐噪声的比例 = 1 -信噪比
            noise_pct = (1 - self.snr)
            # 选择的值为 (0, 1, 2)，每个取值的概率分别为 [signal_pct, noise_pct/2., noise_pct/2.]
            # 椒噪声和盐噪声分别占 noise_pct 的一半
            # 1 为盐噪声，2 为 椒噪声
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct / 2., noise_pct / 2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255  # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            # 再转换为 image
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        # 如果随机概率大于 seld.p，则直接返回原图
        else:
            return img


class GaussianNoise(object):
    """增加高斯噪声"""

    def __init__(self, mean=0, sigma=0.01, p=0.5):

        self.mean = mean
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            # 将图片标准化
            img_ = np.array(img).copy()
            img_ = img_ / 255.0

            # 产生高斯 noise
            noise = np.random.normal(self.mean, self.sigma, img_.shape)
            # 将噪声和图片叠加
            gaussian_out = img_ + noise
            # 将超过 1 的置 1，低于 0 的置 0
            gaussian_out = np.clip(gaussian_out, 0, 1)
            # 将图片灰度范围的恢复为 0-255
            gaussian_out = np.uint8(gaussian_out * 255)
            return Image.fromarray(gaussian_out)
        else:
            return img
