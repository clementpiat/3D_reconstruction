from utils import x_to_np_coordinates, epipolar_y, x_to_img_coordinates, scatter

import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

class Stereo:
    def __init__(self, w, h, distance="MS"):
        self.w = w
        self.h = h
        self.method = distance

    def distance(self, a, b):
        if self.method == "MS":
            return np.sum(((a-a.mean()) - (b-b.mean()))**2)
        else:
            return spatial.distance.cosine(a-a.mean(), b-b.mean())

    def find_x2(self, x1, F, img1, img2, kernel_size=[5,5], show=False):
        padding_i, padding_j = kernel_size[0]//2, kernel_size[1]//2
        
        i, j = x_to_np_coordinates(x1)
        if i <= padding_i or i >= self.h - padding_i or j <= padding_j or j >= self.w - padding_j:
            return None

        patch1 = img1[(i-padding_i):(i+padding_i+1), (j-padding_j):(j+padding_j+1)].flatten()
        
        shortest_distance = np.inf
        best_x2 = None
        l = F @ x1
        distances = []
        for x in range(padding_j, self.w-padding_j):
            y = epipolar_y(x, l)
            i, j = x_to_np_coordinates([x, y, 1])
            if i > padding_i and i < self.h - padding_i:
                patch2 = img2[(i-padding_i):(i+padding_i+1), (j-padding_j):(j+padding_j+1)].flatten()
                d = self.distance(patch1, patch2)
                distances.append(d)
                if d < shortest_distance:
                    shortest_distance = d
                    best_x2 = [x, y, 1]
        if show:
            plt.plot(distances)
            plt.show()
            
        return best_x2

    def plot_epipolar_line(self, img, l, show=True):
        plt.imshow(img)
        i1,j1 = x_to_img_coordinates([0,  epipolar_y(0, l), 1])
        i2,j2 = x_to_img_coordinates([self.w-1,  epipolar_y(self.w-1, l), 1])
        plt.plot([i1,i2], [j1, j2], 'r')
        plt.xlim([0,self.w-1])
        plt.ylim([self.h-1,0])
        if show:
            plt.show()

    def show_best_x2(self, img1, img2, x1, x2, l):
        f = plt.figure()
        f.add_subplot(1,2,1)
        plt.imshow(img1)
        scatter(x1)
        f.add_subplot(1,2,2)
        self.plot_epipolar_line(img2, l, show=False)
        scatter(x2)
        plt.show()