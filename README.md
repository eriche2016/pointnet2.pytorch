This project largely borrowed code from [pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch) by fxia22.
### all the steps to run this project can be checked in [run.sh](https://github.com/eriche2016/pointnet2.pytorch/blob/master/run.sh), except for visualization steps

### visualization step
Here i download opencv binary in [Download opencv-3.2.0-vc14.exe (123.5 MB)](https://sourceforge.net/projects/opencvlibrary/files/opencv-win/) 
and then install it, Goto ```opencv/build/python/2.7``` folder.
Copy ```cv2.pyd``` to ```C:/Python27/lib/site-packages```. 
then build the ```render_balls_so.cpp```. 
And run ```python display_results.py```. see [here](https://github.com/eriche2016/pointnet2.pytorch/tree/master/tools/visualizations)
 
### demo images 
part segmentation result:
![seg](https://github.com/eriche2016/pointnet2.pytorch/blob/master/tools/pics/display.png)