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
Backpack dataset, view beyond either camera in the y-direction (above both cameraa, moved 0.1 times the baseline up): 
![extrapolated_backpack_0_0 1](https://github.com/user-attachments/assets/347e8df1-a7db-4cd4-8517-943b6786a537)

To see more examples of extrapolation and other datasets, see ```outputs/extrapolation```. Images are all named beginning with "extrapolation",
then the dataset, then the x-direction movement, and then the y-direction movement. 
