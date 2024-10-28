import cv2
import numpy as np
import matplotlib.pyplot as plt


def grid_window(image, win_sz=[32,32]):
    """
    image: W*H
    wins:  W1*H1*w*h (interrogation windows)
    """ 
    assert len(image.shape)==2, "Only monocolor image is considered"
    W, H = image.shape
    w, h = win_sz[0], win_sz[1]

    # A trick to pad mean value to the boundary
    image = np.pad(image[1:-1,1:-1], (1,), constant_values=np.mean(image))

    # local index
    local_x, local_y = np.meshgrid(np.arange(w)-w//2, np.arange(h)-h//2, indexing="ij")
    local_x, local_y = local_x.reshape([1,1,w,h]), local_y.reshape([1,1,w,h])
    
    # start position of windows
    global_x, global_y = np.arange(0,W), np.arange(0,H)
    global_x, global_y = np.meshgrid(global_x, global_y, indexing="ij")
    global_x, global_y = global_x.reshape([W,H,1,1]), global_y.reshape([W,H,1,1])

    # outputs
    x, y = global_x+local_x, global_y+local_y
    x, y = np.clip(x,0,W-1), np.clip(y,0,H-1)
    wins = image[x, y]
    return wins


def preprocessing(win):
    win=win- np.mean(win, axis=(-2,-1), keepdims=True)
    win=win/(np.linalg.norm(win, axis=(-2,-1), keepdims=True)+1e-9)
    return win
        

def scc(win1, win2):
    """ standard fft based cross correlation
    """
    F1 = np.fft.rfft2(win1, axes=(-2,-1))
    F2 = np.fft.rfft2(win2, axes=(-2,-1))
    R = np.conj(F1)*F2
    r = np.fft.irfft2(R, axes=(-2,-1))
    shift_r = np.fft.fftshift(r, axes=(-2,-1))
    return shift_r


def spof(img1, img2):
    """ symmetric phase-only filter"""
    F1 = np.fft.rfft2(img1, axes=(-2,-1))
    F2 = np.fft.rfft2(img2, axes=(-2,-1))
    R = np.conj(F1)*F2
    R = R/(np.sqrt(np.abs(R))+1e-5)
    r = np.fft.irfft2(R, axes=(-2,-1))
    shift_r = np.fft.fftshift(r, axes=(-2,-1))
    return shift_r


def argmax2d(R):
    """
    R: W*H*w*h
    mx, my: W*H
    mx(x,y),my(x,y) = argmax_{u,v} R(x,y,u,v)
    """
    W,H,w,h = R.shape
    r_flatten = R.reshape(W, H, -1)
    mx, my = np.unravel_index(np.argmax(r_flatten, axis=-1), [w,h])
    mx, my = np.clip(mx,2,w-3), np.clip(my,2,h-3)
    return mx,my

def subpixel(Rm, s=3):
    """ Fit the subpixel displacement with Gaussian function
    Rm:W*H*5
    ds:W*H*1
    """
    W,H,B = Rm.shape
    w1 = [0,-2,4,-2,0]  if s==3 else [-5,2,6,2,-5] # [-20,10,20,10,-20]  more options
    w2 = [0,-1,0,1,0] if s==3 else [-3,-3,0,3,3]   # [-14,-7,0,7,14] 
    w1, w2 = np.array(w1).reshape(1,1,-1), np.array(w2).reshape(1,1,-1)

    logRm = np.log(np.clip(Rm,1e-3,None))
    ds = np.sum(logRm*w2,axis=-1,keepdims=True)/np.sum(logRm*w1+1e-8,axis=-1,keepdims=True)
    return ds


def displace(R):
    """
    R: W*H*w*h
    mx_,my_: W*H 
    R(x_,y_,mx_,my_) has the maximum value in subpixel
    """
    W,H,w,h = R.shape

    # Find the integer displacement 
    mx, my = argmax2d(R) 
    # mx, my = np.clip(mx,2,w-3), np.clip(my,2,h-3) 

    # Read out the R values around the maximal point (x_,y_,mx,my) 
    x_, y_ = np.meshgrid(np.arange(W), np.arange(H), indexing="ij")
    mx, my, x_, y_ = mx[...,np.newaxis], my[...,np.newaxis], x_[...,np.newaxis], y_[...,np.newaxis]
    loc_ = np.array([-2,-1,0,1,2]).reshape([1,1,5])
    
    Rx = R[x_,y_,mx+loc_,my]  # R(x,y_,mx-2:mx+2,my)
    Ry = R[x_,y_,mx,my+loc_]  # R(x,y_,mx,my-2:my+2)


    #Turn the local values to subpixel displacement
    subx = subpixel(Rx)
    suby = subpixel(Ry)
    return mx+subx-w/2, my+suby-h/2


def crosscorrelation(img1,img2, win_sz=[32,32]):
    # The interrogation windows
    win1 = grid_window(img1,win_sz=win_sz)
    win2 = grid_window(img2,win_sz=win_sz)

    # Image pre-processing for cross correlation
    win1, win2 = preprocessing(win1), preprocessing(win2)

    # Standard cross correlation
    R = scc(win1, win2)
    # R = spof(win1, win2)

    # Find the maximum position
    mx, my = displace(R)
    u, v = mx[...,0], my[...,0]
    x, y = np.meshgrid(np.arange(img1.shape[0]), np.arange(img1.shape[1]), indexing="ij")
    u,v = np.float32(u), np.float32(v)
    return x, y, u, v
    

def unit_test():
    # img1 = cv2.imread("./TestImages/Case3/vp1a.tif", 0)[:256,:256]
    # img2 = cv2.imread("./TestImages/Case3/vp1b.tif", 0)[:256,:256]
    img1 = cv2.imread("./TestImages/Case3/vp1a.tif", 0)
    img2 = cv2.imread("./TestImages/Case3/vp1b.tif", 0)
    
    x1,y1,u1,v1 = crosscorrelation(img1, img2)
    plt.figure()
    # plt.quiver(x1,y1,u1,v1) # Without any modification
    plt.quiver(x1[::8,::8],y1[::8,::8],u1[::8,::8],v1[::8,::8])
    plt.title("The one pass CC")
    plt.show()

if __name__=="__main__":
    unit_test()