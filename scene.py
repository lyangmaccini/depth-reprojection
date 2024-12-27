import cv2
import numpy as np
import utils

class Scene:
    def __init__(self, filepath):
        self.im0 = cv2.imread(filepath + "/im0.png") # shape: height, width, ch
        self.im1 = cv2.imread(filepath + "/im1.png")
        self.read_calib(filepath)
        self.f = self.cam0[0][0]
        self.disp0 = utils.readPFM(filepath + "/disp0.pfm")[0]
        self.disp1 = utils.readPFM(filepath + "/disp1.pfm")[0]   
        self.disparity_to_depth()
        self.setup_extrinsics()

    def read_calib(self, filepath):
        c = open(filepath + "/calib.txt", "r")

        self.cam0 = c.readline()
        self.cam0 = self.cam0[6:-2]
        self.cam0 = self.cam0.split("; ")
        self.cam0 = [float(num) for line in self.cam0 for num in line.split(" ")]
        self.cam0 = np.array(self.cam0).reshape(3, 3)

        self.cam1 = c.readline()
        self.cam1 = self.cam1[6:-2]
        self.cam1 = self.cam1.split("; ")
        self.cam1 = [float(num) for line in self.cam1 for num in line.split(" ")]
        self.cam1 = np.array(self.cam1).reshape(3, 3)

        self.doffs = c.readline() # distance between cameras in x direction 
        self.doffs = float(self.doffs[6:]) 

        self.baseline = c.readline()
        self.baseline = float(self.baseline[9:])

        self.width = c.readline()
        self.width = int(self.width[6:]) 

        self.height = c.readline()
        self.height = int(self.height[7:]) 

        c.close()

    def disparity_to_depth(self):
        self.holes_disp0 = np.isinf(self.disp0)
        self.holes_disp1 = np.isinf(self.disp1)
        self.disp0 =  self.baseline * self.f / (self.disp0 + self.doffs)
        self.disp1 = self.baseline * self.f / (self.disp1 + self.doffs)
    
    def setup_extrinsics(self):
        self.extrinsic_cam0 = utils.get_projection_matrix(0, 0, 0, 0, 0, 0, 0)
        self.extrinsic_cam1 = utils.get_projection_matrix(0, 0, 0, 0, -self.baseline, 0, 0)

    def in_bounds(self, coords):
        return coords[0] > -1 and coords[0] < self.width and coords[0] > -1 and coords[1] < self.height