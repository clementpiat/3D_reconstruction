from utils import is_inside

from scipy.optimize import minimize
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random as rd

def get_X(P1, P2, x1, x2):
    A = np.array([
        x1[0]*P1[2,:] - P1[0,:],
        x1[1]*P1[2,:] - P1[1,:],
        x2[0]*P2[2,:] - P2[0,:],
        x2[1]*P2[2,:] - P2[1,:]
    ])
    
    def fun(x):
        x[-1] = 1
        return np.sum((A @ x)**2)
    
    x0 = [0,0,0,1]
    res = minimize(fun, x0)
    return res.x

def get_SIFT_keypoints(img, silhouette, n_keypoints_min=20, show=False):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    
    kp = sift.detect(gray, None)
    x_list = list(map(lambda x: list(map(int, x.pt))+[1], kp))
    
    x_list, kp = zip(*[(x, k) for x, k in zip(x_list, kp) if is_inside(x, silhouette)])

    if show:
        img_kp = cv.drawKeypoints(gray, kp, img.copy())
        plt.imshow(img_kp)

    h,w,_ = img.shape
    x_list = list(x_list)
    n = len(x_list)
    while(n<n_keypoints_min):
        x = [rd.randint(0,w-1), rd.randint(0,h-1), 1]
        if is_inside(x, silhouette):
            x_list.append(x)
            n+=1

    return x_list