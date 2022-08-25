import os
import numpy as np
import cv2
import random
import math
import tqdm

from data.degradations import circular_lowpass_kernel, random_mixed_kernels
from data.degradations import random_add_gaussian_noise, random_add_poisson_noise

class LQDataGenerator(object):
    def __init__(self, opt):
        super(LQDataGenerator, self).__init__()
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.paths = [os.path.join(self.gt_folder, v) for v in os.listdir(self.gt_folder)]

        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']
        self.betap_range = opt['betap_range']
        self.sinc_prob = opt['sinc_prob']

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.pulse_tensor = np.zeros(shape=[21, 21], dtype='float32')  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def generate_lq_data(self, save_folder):
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)

        for gt_path in tqdm.tqdm(self.paths):
            img_gt = cv2.imread(gt_path, -1)
            img_gt = img_gt.astype(np.float32)/255.
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.opt['sinc_prob']:
                # this sinc filter setting is for kernels ranging from [7, 21]
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel = random_mixed_kernels(
                    self.kernel_list,
                    self.kernel_prob,
                    kernel_size,
                    self.blur_sigma,
                    self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range,
                    self.betap_range,
                    noise_range=None)
            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
            kernel = kernel.astype(np.float32)

            # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.opt['sinc_prob2']:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel2 = random_mixed_kernels(
                    self.kernel_list2,
                    self.kernel_prob2,
                    kernel_size,
                    self.blur_sigma2,
                    self.blur_sigma2, [-math.pi, math.pi],
                    self.betag_range2,
                    self.betap_range2,
                    noise_range=None)

            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
            kernel2 = kernel2.astype(np.float32)

            # ------------------------------------- sinc kernel ------------------------------------- #
            if np.random.uniform() < self.opt['final_sinc_prob']:
                kernel_size = random.choice(self.kernel_range)
                omega_c = np.random.uniform(np.pi / 3, np.pi)
                sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
                # paddle.disable_static()
                # sinc_kernel = paddle.Tensor(sinc_kernel)
            else:
                sinc_kernel = self.pulse_tensor

            sinc_kernel = sinc_kernel.astype(np.float32)
            lq = self.feed_data(img_gt, kernel, kernel2, sinc_kernel)
            save_path = os.path.join(save_folder, os.path.basename(gt_path))
            cv2.imwrite(save_path, lq.astype(np.uint8))


    def feed_data(self,img_gt, kernel1, kernel2, sinc_kernel):
        # numpy
        # training data synthesis
        gt = img_gt

        [ori_w, ori_h, _] = gt.shape

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = cv2.filter2D(gt, -1, kernel1)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        if mode=='area':
            out = cv2.resize(out, (int(ori_h*scale), int(ori_w*scale)), interpolation=cv2.INTER_AREA)
        elif mode=='bilinear':
            out = cv2.resize(out, (int(ori_h * scale), int(ori_w * scale)), interpolation=cv2.INTER_LINEAR)
        else:
            out = cv2.resize(out, (int(ori_h * scale), int(ori_w * scale)), interpolation=cv2.INTER_CUBIC)

        # noise
        gray_noise_prob = self.opt['gray_noise_prob']
        if np.random.uniform() < self.opt['gaussian_noise_prob']:
            out = random_add_gaussian_noise(
                out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise(
                out,
                scale_range=self.opt['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        # JPEG compression
        jpeg_p = np.random.uniform(low=self.opt['jpeg_range'][0], high=self.opt['jpeg_range'][1])
        jpeg_p = int(jpeg_p)
        out = np.clip(out, 0, 1)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_p]
        _, encimg = cv2.imencode('.jpg', out * 255., encode_param)
        out = np.float32(cv2.imdecode(encimg, 1))/255


        # out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.opt['second_blur_prob']:
            out = cv2.filter2D(out, -1, kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range2'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        if mode == 'area':
            out = cv2.resize(out, (int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), interpolation=cv2.INTER_AREA)
        elif mode == 'bilinear':
            out = cv2.resize(out, (int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), interpolation=cv2.INTER_LINEAR)
        else:
            out = cv2.resize(out, (int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), interpolation=cv2.INTER_CUBIC)
        # noise
        gray_noise_prob = self.opt['gray_noise_prob2']
        if np.random.uniform() < self.opt['gaussian_noise_prob2']:
            out = random_add_gaussian_noise(
                out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise(
                out,
                scale_range=self.opt['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            if mode == 'area':
                out = cv2.resize(out, (ori_h // self.opt['scale'], ori_w // self.opt['scale']), interpolation=cv2.INTER_AREA)
            elif mode == 'bilinear':
                out = cv2.resize(out, (ori_h // self.opt['scale'], ori_w // self.opt['scale']), interpolation=cv2.INTER_LINEAR)
            else:
                out = cv2.resize(out, (ori_h // self.opt['scale'], ori_w // self.opt['scale']), interpolation=cv2.INTER_CUBIC)

            # out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
            out = cv2.filter2D(out, -1, sinc_kernel)
            # JPEG compression
            jpeg_p = np.random.uniform(low=self.opt['jpeg_range'][0], high=self.opt['jpeg_range'][1])
            jpeg_p = jpeg_p
            out = np.clip(out, 0, 1)

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_p]
            _, encimg = cv2.imencode('.jpg', out * 255., encode_param)
            out = np.float32(cv2.imdecode(encimg, 1)) / 255

        else:
            # JPEG compression
            jpeg_p = np.random.uniform(low=self.opt['jpeg_range'][0], high=self.opt['jpeg_range'][1])
            jpeg_p = jpeg_p
            out = np.clip(out, 0, 1)

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_p]
            _, encimg = cv2.imencode('.jpg', out * 255., encode_param)
            out = np.float32(cv2.imdecode(encimg, 1)) / 255
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            if mode == 'area':
                out = cv2.resize(out, (ori_h // self.opt['scale'], ori_w // self.opt['scale']),
                                 interpolation=cv2.INTER_AREA)
            elif mode == 'bilinear':
                out = cv2.resize(out, (ori_h // self.opt['scale'], ori_w // self.opt['scale']),
                                 interpolation=cv2.INTER_LINEAR)
            else:
                out = cv2.resize(out, (ori_h // self.opt['scale'], ori_w // self.opt['scale']),
                                 interpolation=cv2.INTER_CUBIC)

        # clamp and round
        lq = np.clip((out * 255.0), 0, 255)

        return lq


