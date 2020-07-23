# Traffic Sign Detection

This project is part of the bachelor thesis "Traffic Sign Detection for Model Cars using the Histogram of Oriented Gradients and Support Vector Machines".
The trained model is able to detect six German traffic signs:

- speed limit 30 km/h
- speed limit 50 km/h
- speed limit 60 km/h
- stop
- priority
- yield

# Prerequisites

- OpenCV 3.2
- Python 2.7
- Numpy 1.16.5
- scikit-learn 0.20.4
- Matplotlib 2.2.4
- scikit-image 0.14.5
- CVBridge
- ROS melodic
- autominy project


# System structure

## detect images
The folder "detect_images" contains the code that has been used to validate the implemented system using the images of a created dataset.
To run the code a dataset of the form
```	.
	├── ...
	├── dataset
	│   ├── 30
	│   ├── 50
	│   ├── 60
	│   ├── priority
	│   ├── stop
	│   ├── yield
	│   └──  nosign
	└── ...
```
must exist. The path to that dataset needs to be passed as argument.
   ```python shape_number_detect.py /.../dataset ```

## train
The folder "train" contains the files to train both support vector machines, the file for processing the images and a folder containing the data that has been used for training.
No extra parameters are needed to run the code.

## ROS package detect_traffic_sign
The folder "detect_traffic_sign" is the ROS package that has been created.

# Instructions on how to use the ROS package
1. add the ROS package to an exisiting ROS Workspace
2. go to your Workspace
   ```cd ../catkin```
3. build the package with catkin build
4. start the program
   ```rosrun detect_traffic_sign ros_detect_traffic_sign.py```
3. To see the results start either rviz or rqt in a new terminal. The images with the found traffic signs will be published to the topic "traffic_sign".

## Images

<img src="https://user-images.githubusercontent.com/46397845/88280803-cbaedb00-cce6-11ea-8a73-31783e524b0d.jpg">
<img src="https://user-images.githubusercontent.com/46397845/88280795-c94c8100-cce6-11ea-9acf-bff941424a9f.jpg">
<img src="https://user-images.githubusercontent.com/46397845/88280797-ca7dae00-cce6-11ea-9547-81f260bf6f4a.jpg">
