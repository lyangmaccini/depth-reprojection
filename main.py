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

    scene = scene.Scene("data/" + data_filepath)

    if interpolate:
        new_im0 = np.zeros_like(scene.im0)
        new_im1 = np.zeros_like(scene.im1)
        interpolated_image = np.zeros_like(scene.im0)

        i = 0.5

        extrinsic_midpoint = utils.get_projection_matrix(0, 0, 0, 0, -i * scene.baseline, 0, 0)
        cam_midpoint = np.copy(scene.cam0)
        cam_midpoint[0][2] += i * scene.doffs 

        depth_im0 = np.ones((new_im0.shape[0], new_im0.shape[1])) * np.inf
        depth_im1 = np.ones((new_im0.shape[0], new_im0.shape[1])) * np.inf

        for r in range(scene.height):
            for c in range(scene.width):
                if not scene.holes_disp0[r, c]:
                    depth = scene.disp0[r, c]
                    # Find 3D point in each pixel 
                    coords_3d_cam0 = depth * np.linalg.inv(scene.cam0) @ np.array([c, r, 1])

                    # Find the coordinates of the original pixel in the interpolated image
                    coords_2d_cam0 = cam_midpoint @ extrinsic_midpoint @ np.array([coords_3d_cam0[0], coords_3d_cam0[1], coords_3d_cam0[2], 1]) 
                    
                    # Convert to row/column indices
                    r_new, c_new = int(coords_2d_cam0[1]/coords_2d_cam0[2]), int(coords_2d_cam0[0]/coords_2d_cam0[2])

                    if scene.in_bounds(coords_2d_cam0):
                        # Save the original pixel to the new image, making sure that the closer object is visible if 2 object of different depths map to the same pixel
                        if depth < depth_im0[r_new, c_new]: 
                            new_im0[r_new, c_new] = scene.im0[r, c] 

                        # Keep track of which pixels have been mapped to and what depth what is in that pixel is at
                        depth_im0[r_new, c_new] = depth

                if not scene.holes_disp1[r, c]:
                    depth = scene.disp1[r, c]
                    coords_3d_cam1 = depth * np.linalg.inv(scene.cam1) @ np.array([c, r, 1])
                    
                    # Account for the x-direction translation of the second camera
                    coords_3d_cam1[0] += scene.baseline 

                    coords_2d_cam1 = cam_midpoint @ extrinsic_midpoint @ np.array([coords_3d_cam1[0], coords_3d_cam1[1], coords_3d_cam1[2], 1])
                    r_new, c_new = int(coords_2d_cam1[1]/coords_2d_cam1[2]), int(coords_2d_cam1[0]/coords_2d_cam1[2])

                    if scene.in_bounds(coords_2d_cam1):
                        if depth < depth_im1[r_new, c_new]:
                            new_im1[r_new, c_new] = scene.im1[r, c]
                        depth_im1[r_new, c_new] = depth
        print("Individual images reprojected.")
        
        # Interpolate between the two reprojected images
        interpolated_image = 0.5 * new_im0 + 0.5 * new_im1

        # Find what pixels are holes in each image by finding what wasn't mapped to in the new images
        holes_im0 = np.transpose((depth_im0 == np.inf).nonzero())
        holes_im1 = np.transpose((depth_im1 == np.inf).nonzero())

        # Fill in holes with pixels from the other image to keep full opacity of pixels that aren't holes in at least one image
        for hole in holes_im0:
            interpolated_image[hole[0], hole[1]] = new_im1[hole[0], hole[1]]
        for hole in holes_im1:
            interpolated_image[hole[0], hole[1]] = new_im0[hole[0], hole[1]]
                
        holes_im0 = set([tuple(row) for row in holes_im0])
        holes_im1 = set([tuple(row) for row in holes_im1])
        overlapping_holes = holes_im0 & holes_im1

        # Average over nearby pixels for holes present in both images to fill in gaps
        for hole in overlapping_holes:
            closest_pixel = None
            second_closest_pixel = None
            r, c = hole[0], hole[1]
            searching_bound = 1
            while second_closest_pixel is None:
                for i in range(r - searching_bound, r + searching_bound + 1):
                    for j in range(c - searching_bound, c + searching_bound + 1):
                        if scene.in_bounds([j, i]) and (i, j) not in overlapping_holes:
                            if closest_pixel is None:
                                closest_pixel = interpolated_image[i, j]
                            else:
                                second_closest_pixel = interpolated_image[i, j]
                searching_bound += 1
            interpolated_image[r, c] = np.round((closest_pixel + second_closest_pixel) / 2)

        cv2.imwrite("interpolated_output.png", interpolated_image)
        # Use a median blur to remove any uncaught holes 
        cv2.imwrite("interpolated_" + data_filepath + ".png", cv2.medianBlur(interpolated_image.astype('float32'), 3))
        print("Interpolated image saved.")