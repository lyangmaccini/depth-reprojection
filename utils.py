import numpy as np
import re
import math

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
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def get_projection_matrix(rx, ry, rz, theta, tx, ty, tz): # theta should be in radians
    projection = np.zeros((3, 4))
    c = math.cos(theta)
    s = math.sin(theta)
    projection[0] = np.array([c + rx * rx * (1-c), rx * ry * (1-c) - rz * s, rx * rz * (1-c) + ry * s, tx])
    projection[1] = np.array([rx * ry * (1-c), c + ry * ry * (1-c), ry * rz * (1-c) - rx * s, ty])
    projection[2] = np.array([rx * rz * (1-c) - ry * s, ry * rz * (1-c) + rx * s, c + rz * rz * (1-c), tz])
    return projection


