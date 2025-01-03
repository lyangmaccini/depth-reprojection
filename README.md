# Depth Map Image Re-projection
In this project, I use two views of the same scene and their corresponding disparity maps to 
first generate images interpolating between the two views, and then extrapolate beyond either 
view. When running the project, you can choose what dataset from the Middlebury stereo datasets
to run. Currently, the available datasets are those in the ```data``` folder: ```backpack```, ```jadeplant```, ```piano```, 
and ```motorcycle```. To choose a dataset, pass the dataset's name in the ```data``` folder 
as the ```-d``` or ```--dataset``` argument. Further, you must specify whether you want to interpolate or extrapolate 
from the original image with the ```-i``` or ```--interpolate``` argument, which should be either ```True``` or
```False```. Finally, you can specify how much to move the camera around by with the arguments ```-x``` or ```-xcoord``` and 
```-y``` or ```-ycoord```.

For example, to generate a view a quarter of the way between both cameras on the backpack dataset,
run ```python main.py -d backpack -i True -x 0.25```.

# Results- Interpolation
Backpack dataset: 
View halfway between the two input views-
![backpack_halfway_view](https://github.com/user-attachments/assets/9c80fb2e-afee-40d8-b4f9-ee108a0ca0f1)

Motorcycle dataset:
![motorcycle_halfway_view](https://github.com/user-attachments/assets/df238cbf-520b-45f0-9712-fb8e68445921)

To see more examples of interpolation, GIFs of camera translation, and other datasets, see ```outputs/interpolation```.

# Results- Extrapolation 
Backpack dataset, view beyond either camera in the x-direction (to the right of the rightmost camera, moved 0.25 time the baseline to the right): 
![extrapolated_backpack_1 25_0](https://github.com/user-attachments/assets/53ebd884-d2ab-4bc3-bd92-ae4ac8597575)

Motorcycle dataset, view to the left of leftmost camera (camera moved by 0.5 times the baseline to the left):
![extrapolated_motorcycle_-0 5_0](https://github.com/user-attachments/assets/b6006a0f-d0c6-4887-9a67-5dff60f5c2a9)

To see more examples of extrapolation and other datasets, see ```outputs/extrapolation```.
