import os
import numpy as np
import cv2
# from skimage.registration import optical_flow_tvl1, optical_flow_ilk

import matplotlib.pyplot as plt

# from openpiv import tools, scaling, pyprocess, validation, filters
# from UnFlowNet.models import Network, device, estimate
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from raft import raft_estimator
from opticalflow import opticalflow
from correlation import crosscorrelation


""" Basic functions
"""

class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


# interpolation with opencv
def remap(img, x, y):
    x, y = np.float32(x), np.float32(y)
    out = cv2.remap(img, y, x, cv2.INTER_CUBIC)  # INTER_LANCZOS4 INTER_CUBIC INTER_LINEAR
    return out


def warping(img1, img2, u, v, method='CDI'):
    # FDI: out1(x,y)=img1(x+u, y+v)
    # FDI, CDI, FDDI, CDDI

    assert img1.shape == img2.shape == u.shape == v.shape

    x, y = np.meshgrid(np.arange(u.shape[0]), np.arange(u.shape[1]), indexing='ij')
    # u, v = -u, -v
    if method == 'FDI':
        # out1= img1.copy()
        out1= remap(img1, x, y)
        out2= remap(img2, x+u, y+v)
    elif method == 'FDI2':
        out1= remap(img1, x-u, y-v)
        out2= remap(img2, x, y)
        # out2= img2.copy()
    elif method == 'CDI':
        out1= remap(img1, x-0.5*u, y-0.5*v)
        out2= remap(img2, x+0.5*u, y+0.5*v)
    else:
        raise NotImplementedError
    return out1, out2


# Our wrapper for iterative deformation PIV
class DeformPIV():
    def __init__(self, config):
        self._c = config
        # assert self._c.pivmethod in ['opticalflow','opticalflow2','opticalflow3', 'openpiv', 'deeppiv','raft_estimator']
        # assert self._c.deform in ['FDI', 'FDI2', 'CDI', 'FDDI', 'FDDI2', 'CDDI']
        assert self._c.pivmethod in ['opticalflow','crosscorrelation','raft_estimator']
        assert self._c.deform in ['FDI', 'FDI2', 'CDI']

        self.onepass = eval(self._c.pivmethod) # opticalflow1
        self.warping = self._c.deform

    def compute(self, image1, image2, u=None, v=None):
        assert image1.shape == image2.shape
        # obtain the initial vector field
        if u is not None:
            assert image1.shape == image2.shape == u.shape == v.shape
        else:
            u, v = np.zeros_like(image1), np.zeros_like(image1)

        # iterative operation
        for iter in range(self._c.runs):
            # image warping
            img1, img2 = warping(image1, image2, u, v, self.warping)

            # update the estimation
            if self._c.pivmethod == 'opticalflow':
                x, y, du, dv = self.onepass(img1, img2)
            if self._c.pivmethod == 'crosscorrelation':
                win_szs=[[32,32],[16,16],[16,16]]+[[8,8]]*20
                x, y, du, dv = self.onepass(img1, img2,  win_sz=win_szs[iter])

            u, v = u+du, v+dv
            
            # using a blur trick to make the iteration stable
            u = cv2.medianBlur(u.astype(np.float32),5)
            v = cv2.medianBlur(v.astype(np.float32),5)
        
            smooth_k = 3
            for i in range(3):
                u = cv2.blur(u, (smooth_k,smooth_k)) 
                v = cv2.blur(v, (smooth_k,smooth_k))
        
        return x, y, u, v


def unit_test():
    config = AttrDict()
    config.pivmethod = 'crosscorrelation' # ['crosscorrelation', 'opticalflow', 'raft_estimator']
    config.deform = 'FDI' # ['FDI', 'CDI', 'CDDI', 'FDDI']
    config.runs = 3
    
    img1 = cv2.imread("./TestImages/Case3/vp1a.tif", 0)
    img2 = cv2.imread("./TestImages/Case3/vp1b.tif", 0)
    piv = DeformPIV(config)
    x1, y1, u1, v1 = piv.compute(img1, img2)
    
    plt.figure()
    # plt.quiver(x1,y1,u1,v1) # Without any modification
    plt.quiver(x1[::8,::8],y1[::8,::8],u1[::8,::8],v1[::8,::8])
    plt.title("The one pass CC")
    plt.show()


if __name__=="__main__":
    unit_test()

