from utils import is_inside

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random as rd

# X = (x,y,z,w) is a 3D-point
def get_X(P1, P2, x1, x2):
    A = np.array([
        x1[0]*P1[2,:] - P1[0,:],
        x1[1]*P1[2,:] - P1[1,:],
        x2[0]*P2[2,:] - P2[0,:],
        x2[1]*P2[2,:] - P2[1,:]
    ])
    
    return - np.linalg.pinv(A[:,:3]) @ A[:,-1]

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