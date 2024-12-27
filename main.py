import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import math
import argparse
import scene

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        required=True,
                        help="Desired image dataset from Middlebury Stereo dataset to reproject- ex: backpack, mask, umbrella")
    parser.add_argument("-i", "--interpolate",
                        required=True,
                        help="Whether to produce an interpolated view halfway between both cameras or an extrapolated view beyond both cameras")
    args = parser.parse_args()
    data_filepath = args.dataset
    interpolate = args.interpolate

    scene = scene.Scene(data_filepath)

    if interpolate:
        interp_im0 = np.zeros_like(scene.im0)
        interp_im1 = np.zeros_like(scene.im1)
        interpolated_image = np.zeros_like(scene.im0)

        i = 0.5

        extrinsic_midpoint = utils.get_projection_matrix(0, 0, 0, 0, -i * scene.baseline, 0, 0)
        cam_midpoint = np.copy(scene.cam0)
        cam_midpoint[0][2] += i * scene.doffs 

        holes_im0 = list(np.ndindex(scene.height, scene.width))
        holes_im1 = list(np.ndindex(scene.height, scene.width))

        for r in range(scene.height):
            for c in range(scene.width):
                if not scene.holes_disp0[r, c]:
                    # Find 3D points in each pixel 
                    coords_3d_cam0 = scene.disp0[r, c] * np.linalg.inv(scene.cam0) @ np.array([c, r, 1])

                    # Find the new indices of the 3D point in the interpolated image
                    coords_2d_cam0 = cam_midpoint @ extrinsic_midpoint @ np.array([coords_3d_cam0[0], coords_3d_cam0[1], coords_3d_cam0[2], 1]) # the coordinates of the original pixel in the interpolated image
                    coords_2d_cam0 = [int(coord/coords_2d_cam0[2]) for coord in coords_2d_cam0[:2]] # convert from homogenous coordinates to indices

                    if scene.in_bounds(coords_2d_cam0):
                        # Save the original pixel to the new image
                        interp_im0[coords_2d_cam0[1], coords_2d_cam0[0]] = scene.im0[r, c] 

                        # Keep track of which pixels have been mapped from in the original image for hole detection
                        if (coords_2d_cam0[1], coords_2d_cam0[0]) in holes_im0:
                            holes_im0.remove((coords_2d_cam0[1], coords_2d_cam0[0]))
                
                if not scene.holes_disp1[r, c]:
                    coords_3d_cam1 = scene.disp1[r, c] * np.linalg.inv(scene.cam1) @ np.array([c, r, 1])
                    # Account for the x-direction translation of the second camera
                    coords_3d_cam1[0] += scene.baseline 

                    coords_2d_cam1 = cam_midpoint @ extrinsic_midpoint @ np.array([coords_3d_cam1[0], coords_3d_cam1[1], coords_3d_cam1[2], 1])

                    coords_2d_cam1 = [int(coord/coords_2d_cam1[2]) for coord in coords_2d_cam1[:2]]

                    if scene.in_bounds(coords_2d_cam1):
                        interp_im1[coords_2d_cam1[1], coords_2d_cam1[0]] = scene.im1[r, c]

                        if (coords_2d_cam1[1], coords_2d_cam1[0]) in holes_im1:
                            holes_im1.remove((coords_2d_cam1[1], coords_2d_cam1[0]))
        
        interpolated_image = i * interp_im0 + (1 - i) * interp_im1
        for r, c in holes_im0:
            interpolated_image[r, c] = interp_im1[r, c]
        for r, c in holes_im1:
            interpolated_image[r, c] = interp_im0[r, c]

        cv2.imwrite("interp_im0.png", interp_im0)
        cv2.imwrite("interp_im1.png", interp_im1)
        cv2.imwrite("interpolated_output.png", interpolated_image)
        print("Interpolated image saved.")
        cv2.imwrite("filtered_interp_im.png", cv2.medianBlur(interpolated_image.astype('int8'), 3))

        


