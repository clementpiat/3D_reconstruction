import matplotlib.pyplot as plt
import os
import numpy as np

obj="pig"

def get_silhouette(number):
    return plt.imread(os.path.join(f'../data/{obj}_data/silhouettes', f'{str(number).zfill(4)}.pgm')) == 0

def imread(number):
    return plt.imread(os.path.join(f'../data/{obj}_data/images', f'{str(number).zfill(4)}.ppm'))

def get_P(number):
    return np.loadtxt(os.path.join(f'../data/{obj}_data/calib', f'{str(number).zfill(4)}.txt'), skiprows=1)

def read(number):
    return imread(number), get_P(number), get_silhouette(number)

# x = (X, Y, w)
def x_to_img_coordinates(x):
    return int(x[0]/x[2]), int(x[1]/x[2])

# x = (X, Y, w)
def x_to_np_coordinates(x):
    i, j = x_to_img_coordinates(x)
    return j, i

# x = (X, Y, w)
def scatter(x):
    i, j = x_to_img_coordinates(x)
    plt.scatter([i], [j], c='r')

def epipolar_y(x, l):
    return (-l[2] - l[0]*x)/l[1]


def lie_matrix(v):
    M = np.zeros((3,3))
    M[0,1] = -v[2]
    M[0,2] =  v[1]
    M[1,0] =  v[2]
    M[1,2] = -v[0]
    M[2,0] = -v[1]
    M[2,1] =  v[0]
    return M

def compute_F(P1, P2):
    P1x_ = np.linalg.inv(P1[:,:3])
    return (lie_matrix(P2[:,-1]) - lie_matrix(P2[:,:3] @ P1x_ @ P1[:,-1])) @ P2[:,:3] @ P1x_

def is_inside(x, silhouette):
    return silhouette[x_to_np_coordinates(x)]
