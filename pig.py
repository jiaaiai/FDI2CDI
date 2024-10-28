import numpy as np
import torch   
import cv2
import matplotlib.pyplot as plt

from flow import uniform, solid_rot, lamb_oseen, sin_flow

EPS = 1e-10


def erf(x):
    """
    It's hard to believe that we have to wrapper the erf function from pytorch
    """
    x = torch.tensor(x)
    y = torch.erf(x).cpu().numpy()
    return y


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def config():
    cfg = AttrDict
    cfg.img_sz = (306,306) # For images
    cfg.ppp = 0.15
    cfg.dp = 2.5
    cfg.d_std = 0.1
    cfg.i_std =0.1
    cfg.miss_ratio = 0.1
    
    cfg.flow=uniform # For flows
    return cfg


# PIG for one images
def gen_pair_particles(config):
    # generate particles' parameters
    p1, p2= AttrDict(), AttrDict()

    # settings 
    img_sz = config.img_sz
    ppp = config.ppp
    dp, d_std = config.dp, config.d_std
    i_std = config.i_std
    miss_ratio = config.miss_ratio

    flow = config.flow

    # generate particles' parameters
    p1, p2= AttrDict(), AttrDict()
    p1.num = p2.num = np.round(ppp*np.prod(img_sz)).astype(np.int32)
    p1.nd = p2.nd = dp
    p1.x = p2.x = np.random.uniform(0, img_sz[1], p1.num)
    p1.y = p2.y = np.random.uniform(0, img_sz[0], p1.num)
    p1.d = p2.d = np.abs(np.random.randn(p1.num)*d_std+ dp)
    p1.i = p2.i = np.random.randn(p1.num)*i_std+ 0.85

    V, V_fdi, V_cdi = flow(p1.x, p2.y)
    p2.x, p2.y = p2.x+V_fdi[0], p2.y+V_fdi[1]
    return p1, p2


def particles2image(config, particle):
    """
    Using the erf function to synthesis the particle images
    """
    img_sz = config.img_sz
    image = np.zeros(img_sz)
    u, v = np.meshgrid(np.arange(img_sz[1]),np.arange(img_sz[0]))
    
    x_s = np.reshape(particle.x, (-1))
    y_s = np.reshape(particle.y, (-1))
    dp_s = np.reshape(particle.d, (-1))
    intensity_s = np.reshape(particle.i, (-1))
    dp_nominal=particle.nd

    for x, y, dp, intensity in zip(x_s, y_s, dp_s, intensity_s):
        # print(x, y, dp, intensity)
        ind_x1 = np.int32(np.clip(x-3*dp-2, 0, img_sz[1]-6*dp-3))
        ind_y1 = np.int32(np.clip(y-3*dp-2, 0, img_sz[0]-6*dp-3))
        
        # ind_x1 = np.int(min(max(0, x-3*dp-2), img_sz[1]-6*dp-3))
        # ind_y1 = np.int(min(max(0, y-3*dp-2), img_sz[0]-6*dp-3))
        ind_x2 = ind_x1 + np.int32(6*dp+3)
        ind_y2 = ind_y1 + np.int32(6*dp+3)

        # print(ind_x1, ind_x2, ind_y1, ind_y2)
        # print(ind_x1.dtype, ind_x2.dtype, ind_y1.dtype, ind_y2.dtype)
        
        lx = u[ind_y1:ind_y2, ind_x1:ind_x2] -x
        ly = v[ind_y1:ind_y2, ind_x1:ind_x2] -y
        b = dp/np.sqrt(8) # from the Gaussian intensity profile assumption

        img =(erf((lx+0.5)/b)-erf((lx-0.5)/b))*(erf((ly+0.5)/b)-erf((ly-0.5)/b))
        img = img*intensity  
        
        image[ind_y1:ind_y2, ind_x1:ind_x2] =  image[ind_y1:ind_y2, ind_x1:ind_x2]+ img
    
    b_n = dp_nominal/np.sqrt(8)
    partition = 1.5*(erf(0.5/b_n)-erf(-0.5/b_n))**2
    image = np.clip(image/partition,0,1.0) 
    image = image*255.0
    image  = np.round(image)
    return image


def gen_piv_pair(config):
    p1, p2 = gen_pair_particles(config)
    img1 = particles2image(config, p1)
    img2 = particles2image(config, p2)

    img_sz = config.img_sz
    gx, gy = np.meshgrid(np.arange(img_sz[1]),np.arange(img_sz[0]))
    V, V_fdi, V_cdi = config.flow(gx, gy)
    return (img1,img2), (V, V_fdi, V_cdi)


def test1():
    # Test for the default config
    cfg = config()
    imgs, Vs = gen_piv_pair(cfg)
    
    # show the results
    plt.figure(figsize=(8,4))
    for k, img in enumerate(imgs):
        plt.subplot(1,2,k+1)
        plt.imshow(img);
    
    plt.figure(figsize=(14,4))
    for k, V in enumerate(Vs):
        plt.subplot(1,3,k+1)
        plt.quiver(V[0][::8,::8],V[1][::8,::8]);
    
    plt.show()

def test2():
    def flow(x,y):
        return lamb_oseen(x,y, Gamma=1e-4, rc=40)
    
    cfg = config()
    cfg.flow = flow
    imgs, Vs = gen_piv_pair(cfg)
    
    # show the results
    plt.figure(figsize=(8,4))
    for k, img in enumerate(imgs):
        plt.subplot(1,2,k+1)
        plt.imshow(img);
    
    plt.figure(figsize=(14,4))
    for k, V in enumerate(Vs):
        plt.subplot(1,3,k+1)
        plt.quiver(V[0][::8,::8],V[1][::8,::8]);
    
    plt.show()


if __name__=='__main__':
    test1()
    test2()
