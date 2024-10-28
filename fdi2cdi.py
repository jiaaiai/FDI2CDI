import numpy as np
from scipy import ndimage
import cv2

# correction function
def remap(img,x,y, method="Bilinear"): # an improved implementation for cv2.remap
    assert method in ["BSPL3", "BSPL4", "BSPL5", "Bilinear", "Bicubic", "SINC8"]
    if method == "BSPL3":
        return ndimage.map_coordinates(img,(y,x), order=3, mode="nearest", prefilter=True)
    elif method == "BSPL4":
        return ndimage.map_coordinates(img,(y,x), order=4, mode="nearest", prefilter=True)
    elif method == "BSPL5":
        return ndimage.map_coordinates(img,(y,x), order=5, mode="nearest", prefilter=True)
    
    x, y = np.float32(x), np.float32(y)
    if method == "Bilinear":
        return cv2.remap(img, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) 
    if method == "Bicubic":
        return cv2.remap(img, x, y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) 
    if method == "SINC8":
        return cv2.remap(img, x, y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE) 


def fdi2cdi(u_t, v_t, t=None, reverse=False, delta=1, method="Bilinear", iter=10):
    """
    v_{cdi}(x)=v_{t}(x+(t-0.5)v_{cdi}(x))
    This implicit translation is implemented as an iterative updating procedure
    """
    assert u_t.shape == v_t.shape
    # print(method, iter)
    x, y = np.meshgrid(np.arange(u_t.shape[1]), np.arange(u_t.shape[0]))
    u_f, v_f = u_t/delta, v_t/delta

    w = -(t-0.5) if reverse else (t-0.5)
    
    u_c, v_c = np.zeros_like(u_f), np.zeros_like(u_f)
    for _ in range(iter):
        u_c = remap(u_f, x+w*u_c, y+w*v_c,method=method)
        v_c = remap(v_f, x+w*u_c, y+w*v_c,method=method)
    u_c, v_c = delta*u_c, delta*v_c
    return u_c, v_c


def test():
    import matplotlib.pyplot as plt
    from flow import uniform, solid_rot, lamb_oseen, sin_flow
    
    names = ["uniform", "solid_rot", "lamb_oseen", "sin_flow"]
    
    x = np.arange(0,256,1)
    y = np.arange(0,256,1)
    x, y = np.meshgrid(x,y)
    
    V1 = uniform(x,y)
    V2 = solid_rot(x,y)
    V3 = lamb_oseen(x,y)
    V4 = sin_flow(x,y)
    
    
    for k, V in enumerate([V1, V2, V3, V4]):
        u_tru, v_tru = V[0]
        u_fdi, v_fdi = V[1]
        u_cdi, v_cdi = V[2]
        u_fdi2cdi, v_fdi2cdi = fdi2cdi(u_fdi, v_fdi, t=0, delta=1)
        
        error = np.sqrt(np.square(u_cdi-u_fdi2cdi)+np.square(v_cdi-v_fdi2cdi))
        print(f"{np.max(u_cdi):5.3f} {np.mean(error):.5f}")
    
        plt.figure(figsize=(14,2.5))
        plt.subplot(1,5,1); plt.quiver(u_tru[::8,::8], v_tru[::8,::8]); plt.title(f"{names[k]}_Truth")
        plt.subplot(1,5,2); plt.quiver(u_fdi[::8,::8], v_fdi[::8,::8]); plt.title(f"{names[k]}_FDI")
        plt.subplot(1,5,3); plt.quiver(u_cdi[::8,::8], v_cdi[::8,::8]); plt.title(f"{names[k]}_CDI")
        plt.subplot(1,5,4); plt.quiver(u_fdi2cdi[::8,::8], v_fdi2cdi[::8,::8]); plt.title(f"{names[k]}_FDI2CDI")
        plt.subplot(1,5,5); plt.imshow(error[10:-10,10:-10]); plt.colorbar();plt.title(f"FDI2CDI_ERROR")
        # print(np.std(error))
    plt.show()

if __name__ == "__main__":
    test()