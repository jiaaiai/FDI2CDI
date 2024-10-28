import numpy as np
import cv2
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
import matplotlib.pyplot as plt

# piv kernel with optical flow (opencv, Farneback)
def opticalflow(img1, img2, level=4):
    flow1 = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, level, 35, 10, 15, 1.3, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    flow2 = cv2.calcOpticalFlowFarneback(img2, img1, None, 0.5, level, 35, 10, 15, 1.3, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    flow = (flow1-flow2)/2
    u, v = flow[...,1], flow[...,0]
    x, y = np.meshgrid(np.arange(img1.shape[0]), np.arange(img1.shape[1]), indexing="ij")
    return x, y, u, v


# piv kernel with optical flow (skimage)
def opticalflow2(img1, img2):
    u, v = optical_flow_ilk(img1, img2, radius=16,num_warp=1, gaussian=True, prefilter=True)
    x, y = np.meshgrid(np.arange(img1.shape[0]), np.arange(img1.shape[1]), indexing="ij")
    return x, y, u, v

# piv kernel with optical flow (opencv, DualTVL1)
def opticalflow3(img1, img2):
    optical_flow= cv2.optflow.DualTVL1OpticalFlow_create(lambda_=0.1,nscales=10,epsilon=0.05,warps=3)
    flow = optical_flow.calc(img1.astype(np.float32), img2.astype(np.float32), None)
    u, v = flow[...,1], flow[...,0]
    x, y = np.meshgrid(np.arange(img1.shape[0]), np.arange(img1.shape[1]), indexing="ij")
    return x, y, u, v


def unit_test():
    # img1 = cv2.imread("./TestImages/Case3/vp1a.tif", 0)[:256,:256]
    # img2 = cv2.imread("./TestImages/Case3/vp1b.tif", 0)[:256,:256]
    img1 = cv2.imread("./TestImages/Case3/vp1a.tif", 0)
    img2 = cv2.imread("./TestImages/Case3/vp1b.tif", 0)
    
    x1,y1,u1,v1 = opticalflow(img1, img2)
    plt.figure()
    # plt.quiver(x1,y1,u1,v1) # Without any modification
    plt.quiver(x1[::8,::8],y1[::8,::8],u1[::8,::8],v1[::8,::8])
    plt.title("The one pass OF")
    plt.show()


if __name__=="__main__":
    unit_test()
