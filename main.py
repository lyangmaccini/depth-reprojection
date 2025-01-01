import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import imageio
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
    
    # images = []
    # filenames = ["data/" + data_filepath + "/im0.png", "outputs/interpolation/" + data_filepath + "/" + data_filepath + "_quarter_view.png",  "outputs/interpolation/" + data_filepath + "/" + data_filepath + "_halfway_view.png",  "outputs/interpolation/" + data_filepath + "/" + data_filepath + "_three_quarter_view.png", "data/" + data_filepath + "/im1.png"]
    # for filename in filenames:
    #     images.append(imageio.imread(filename))
    # imageio.mimsave('outputs/interpolation/' + data_filepath + "/" + data_filepath + '.gif', images)
    # print("GIF saved.")

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

    if interpolate:
        for pixel in overlapping_empty_pixels:
            closest_pixel = None
            second_closest_pixel = None
            r, c = pixel[0], pixel[1]
            searching_bound = 1
            while second_closest_pixel is None:
                for i in range(r - searching_bound, r + searching_bound + 1):
                    for j in range(c - searching_bound, c + searching_bound + 1):
                        if scene.in_bounds(i, j) and (i, j) not in overlapping_empty_pixels:
                            if closest_pixel is None:
                                closest_pixel = combined_image[i, j]
                            else:
                                second_closest_pixel = combined_image[i, j]
                searching_bound += 1
            combined_image[r, c] = np.round((closest_pixel + second_closest_pixel) / 2)

        # Use a median blur to reduce the effect of any irregulaties 
        image_path = "outputs/interpolation/" + data_filepath + "/interpolated_" + data_filepath + ".png"
        cv2.imwrite(image_path, cv2.medianBlur(combined_image.astype('float32'), 3))
        print("Interpolated image saved to " + image_path + ".")
    else:
        cv2.imwrite("jgg.png", combined_image)
        for pixel in overlapping_empty_pixels:
            closest_pixel = None
            r, c = pixel[0], pixel[1]
            searching_bound = 1
            while closest_pixel is None:
                if ix == 0:
                    j = c
                    i = r
                    if iy < 0:
                        i -= searching_bound
                    if iy > 0:
                        i += searching_bound
                if iy == 0:
                    j = c
                    i = r
                    if ix < 0:
                        if j - searching_bound > 0:
                            j -= searching_bound
                        else:
                            j += searching_bound
                    if ix > 0:
                        if j + searching_bound < scene.width - 1:
                            j += searching_bound
                        else:
                            j -= searching_bound
                if ix == 0 or iy == 0:
                    if scene.in_bounds(i, j) and (i, j) not in overlapping_empty_pixels:
                        closest_pixel = combined_image[i, j]
                else:
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

        cv2.imwrite("extrapolated_output.png", combined_image)
        image_path = "outputs/extrapolation/" + data_filepath + "/extrapolated_" + data_filepath + ".png"
        cv2.imwrite(image_path, cv2.medianBlur(combined_image.astype('float32'), 3))
        print("Extrapolated image saved to " + image_path + ".")