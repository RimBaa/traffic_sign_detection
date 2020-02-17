## Traffic Sign Detection

This project is part of the bachelor thesis "Traffic Sign Detection for Model Cars using the Histogram of Oriented Gradients and Support Vector Machines".

## Prerequisites

- OpenCV 3.2
- Python 2.7
- Numpy 1.16.5
- scikit-learn 0.20.4
- Matplotlib 2.2.4
- scikit-image 0.14.5
- CVBridge


## System structure
- detect images
The folder "detect_images" contains the Code that has been used to validate the implemented system using the images of a created dataset.

The second folder "detect_traffic_sign" is the ROS package that has been created.
Both folders contain the trained support vector machines and all files needed to start the program.

##Instructions on how to use the ROS package
1. Add the ROS package to an exisiting ROS Workspace and build the package.
2. Start the program.
  - ```rosrun detect_traffic_sign ros_detect_traffic_sign.py```
3. To see the results start either rviz or rqt in a new terminal. The found traffic signs will be published to the Topic "traffic_sign"
