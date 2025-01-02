# Depth Map Image Re-projection
In this project, I use two views of the same scene and their corresponding disparity maps to 
first generate images interpolating between the two views, and then extrapolate beyond either 
view. When running the project, you can choose what dataset from the Middlebury stereo datasets
to run. Currently, the available datasets are those in the ```data``` folder: ```backpack```, ```jadeplant```,
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

GIF showing 3 views generated between both cameras-
![backpack](https://github.com/user-attachments/assets/2bf52952-63cc-4516-a58c-dce310cf7753)

Jade plant dataset:
![jadeplant_halfway_view](https://github.com/user-attachments/assets/09d8e3e5-95e3-43cf-837a-f72ec65675c9)
![jadeplant](https://github.com/user-attachments/assets/a0653c0a-a411-48b0-89bb-94711012eb2b)

Motorcycle dataset:
![motorcycle_halfway_view](https://github.com/user-attachments/assets/df238cbf-520b-45f0-9712-fb8e68445921)
![motorcycle](https://github.com/user-attachments/assets/af85899d-fc1d-46cb-8185-152b97341519)

To see individual examples of interpolation used to make the GIFs, see ```outputs/interpolation```.

# Results- Extrapolation 
Backpack dataset, view beyond either camera in the x-direction (to the right of the rightmost camera): 

Backpack dataset, view to the left of leftmost camera:
