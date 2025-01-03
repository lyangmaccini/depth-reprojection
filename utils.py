import numpy as np
import re
import math
import imageio

# PFM file reading adapted from materials from the Computer Vision Group, University of Freiburg,
# https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html and 
# https://lmb.informatik.uni-freiburg.de/resources/datasets/IO.py 
def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(b'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: 
        endian = '<'
        scale = -scale
    else:
        endian = '>'

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

# Allows for rotation of projection matrix as well using Rodrigues' rotation matrix
def get_projection_matrix(rx, ry, rz, theta, tx, ty, tz): 
    projection = np.zeros((3, 4))
    c = math.cos(theta)
    s = math.sin(theta)
    projection[0] = np.array([c + rx * rx * (1-c), rx * ry * (1-c) - rz * s, rx * rz * (1-c) + ry * s, tx])
    projection[1] = np.array([rx * ry * (1-c), c + ry * ry * (1-c), ry * rz * (1-c) - rx * s, ty])
    projection[2] = np.array([rx * rz * (1-c) - ry * s, ry * rz * (1-c) + rx * s, c + rz * rz * (1-c), tz])
    return projection

def make_gif(data_filepath):
    images = []
    filenames = ["data/" + data_filepath + "/im0.png", "outputs/interpolation/" + data_filepath + "/" + data_filepath + "_quarter_view.png",  "outputs/interpolation/" + data_filepath + "/" + data_filepath + "_halfway_view.png",  "outputs/interpolation/" + data_filepath + "/" + data_filepath + "_three_quarter_view.png", "data/" + data_filepath + "/im1.png"]
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('outputs/interpolation/' + data_filepath + "/" + data_filepath + '.gif', images)
    print("GIF saved.")