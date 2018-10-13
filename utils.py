import numpy as np
import time

class FPS_Tracker(object):
    def __init__(self):
        self._t0 = time.time()
        self._frames = 0
    
    def new_frame(self):
        t = time.time()
        delta = t - self._t0
        self._frames += 1
        if delta >= 1:
            print('FPS: {:.4f}'.format(self._frames/delta))
            self._t0 = t
            self._frames = 0

def img_white_balance(img, white_ratio):
    for channel in range(img.shape[2]):
        channel_max = np.percentile(img[:, :, channel], 100-white_ratio)
        channel_min = np.percentile(img[:, :, channel], white_ratio)
        img[:, :, channel] = (channel_max-channel_min) * (img[:, :, channel] / 255.0)
    return img

