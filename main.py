import cv2
import numpy as np
import utils
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
    parser.add_argument("-x", "--xcoord",
                        required=False,
                        help="Where to extrapolate the camera to in the x direction, where x = 1 moves the camera from camera 0's position to camera 1's position. Must be greater than 1 for extrapolation")
    parser.add_argument("-y", "--ycoord",
                        required=False,
                        help="Where to extrapolate the camera to in the y direction, where y = 1 moves the camera up by the length of the baseline")
    args = parser.parse_args()
    data_filepath = args.dataset
    interpolate = (args.interpolate == "True")
    ix = args.xcoord
    if ix is not None:
        ix = float(ix)

    scene = scene.Scene("data/" + data_filepath)

    if interpolate:
        if ix is None:
            ix = 0.5
        iy = 0
    else:
        if ix is None:
            ix = 0
        iy = args.ycoord
        if iy is None:
            iy = 0
        else:
            iy = float(iy)

    new_im0 = np.zeros_like(scene.im0)
    new_im1 = np.zeros_like(scene.im1)
    combined_image = np.zeros_like(scene.im0)

    extrinsic_midpoint = utils.get_projection_matrix(0, 0, 0, 0, -ix * scene.baseline, iy * scene.baseline, 0)
    cam_midpoint = np.copy(scene.cam0)
    cam_midpoint[0][2] += ix * scene.doffs 
    cam_midpoint[1][2] += iy * scene.doffs

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

                if scene.in_bounds(r_new, c_new):
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

                if scene.in_bounds(r_new, c_new):
                    if depth < depth_im1[r_new, c_new]:
                        new_im1[r_new, c_new] = scene.im1[r, c]
                        depth_im1[r_new, c_new] = depth
    print("Individual images reprojected.")
        
    # Interpolate between the two reprojected images
    combined_image = 0.5 * new_im0 + 0.5 * new_im1

    # Find what pixels are empty in each image by finding what wasn't mapped to in the new images
    empty_pixels_im0 = np.transpose((depth_im0 == np.inf).nonzero())
    empty_pixels_im1 = np.transpose((depth_im1 == np.inf).nonzero())

    # Fill in empty pixels with pixels from the other image to keep full opacity of pixels that aren't empty in at least one image
    for pixel in empty_pixels_im0:
        combined_image[pixel[0], pixel[1]] = new_im1[pixel[0], pixel[1]]
    for pixel in empty_pixels_im1:
        combined_image[pixel[0], pixel[1]] = new_im0[pixel[0], pixel[1]]

    # Find empty pixels in the same place in both images, which can't be filled with the above method  
    empty_pixels_im0 = set([tuple(row) for row in empty_pixels_im0])
    empty_pixels_im1 = set([tuple(row) for row in empty_pixels_im1])
    overlapping_empty_pixels = empty_pixels_im0 & empty_pixels_im1

    if interpolate or (not interpolate and ix != 0 and iy != 0):
        # Find the closest pixels to average to fill in remaining gaps for interpolation/diagonal extrapolation gaps.
        for pixel in overlapping_empty_pixels:
            closest_pixel = None
            r, c = pixel[0], pixel[1]
            searching_bound = 1
            while closest_pixel is None:
                for i in range(r - searching_bound, r + searching_bound + 1):
                    j = c + searching_bound
                    if scene.in_bounds(i, j) and (i, j) not in overlapping_empty_pixels:
                        closest_pixel = combined_image[i, j]
                    j = c - searching_bound
                    if scene.in_bounds(i, j) and (i, j) not in overlapping_empty_pixels:
                        closest_pixel = combined_image[i, j]
                for j in range(c - searching_bound, c + searching_bound + 1):
                    i = r + searching_bound
                    if scene.in_bounds(i, j) and (i, j) not in overlapping_empty_pixels:
                        closest_pixel = combined_image[i, j]
                    i = r - searching_bound
                    if scene.in_bounds(i, j) and (i, j) not in overlapping_empty_pixels:
                        closest_pixel = combined_image[i, j]
                searching_bound += 1
            combined_image[r, c] = closest_pixel

        image_path = "outputs/interpolation/" + data_filepath + "/interpolated_" + data_filepath + ".png"
    else:
        # To fill larger gaps in the image for extrapolation, find the closest pixels that are likely to be behind the 
        # object causing the gaps. To ensure that the gap is filled with a background object, check the depth of the pixel
        # used to fill the gap. Finally, count how many times surrounding pixels have been checked, in case no nearby pixels 
        # have sufficient depth. In that case, ignore depth requirements and find the closest pixel.
        for pixel in overlapping_empty_pixels:
            closest_pixel = None
            r, c = pixel[0], pixel[1]
            searching_bound = 1
            depth = scene.disp1[r, c]
            counter = 0 
            skip_depth = False 
            while closest_pixel is None:
                counter += 1
                # If there is no x-direction movement, new gaps can likely be filled with the background above or below an object.
                if ix == 0: 
                    if counter > 2 * scene.height and not skip_depth: # ensure that the whole column hasn't already been checked
                        searching_bound = 1
                        skip_depth = True
                    j = c
                    i = r
                    if iy < 0:
                        if i + searching_bound < scene.height - 1:
                            i += searching_bound
                        else:
                            i -= searching_bound - r + 1
                    if iy > 0:
                        if i - searching_bound > 0:
                            i -= searching_bound
                        else:
                            i += searching_bound - r + 1
                # Similarly, if there is no y-direction movement, fill new gaps with background to either side of an object.
                if iy == 0:
                    if counter >= 2 * scene.width and not skip_depth: # ensure that the whole row hasn't been already checked
                        searching_bound = 1
                        skip_depth = True
                    j = c
                    i = r
                    if ix < 0:
                        if j - searching_bound > 0:
                            j -= searching_bound - c + 1
                        else:
                            j += searching_bound - c + 1
                    if ix > 0:
                        if j + searching_bound < scene.width - 1:
                            j += searching_bound
                        else:
                            j -= searching_bound
                if scene.in_bounds(i, j) and (i, j) not in overlapping_empty_pixels:
                    if not skip_depth and iy == 0 and ix == 0:
                        if depth_im1[i, j] > 1.5 * depth:
                            closest_pixel = combined_image[i, j]
                    else:
                        closest_pixel = combined_image[i, j]
                searching_bound += 1
            combined_image[r, c] = closest_pixel

        image_path = "outputs/extrapolation/" + data_filepath + "/extrapolated_" + data_filepath + "_" + str(ix) + "_" + str(iy) + ".png"

    # Use a median blur to reduce the effect of any remaining irregulaties 
    cv2.imwrite(image_path, cv2.medianBlur(combined_image.astype('float32'), 3))
    print("Image saved to " + image_path + ".")