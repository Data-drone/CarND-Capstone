# Udacity Self Driving Car Nano-Degree Capstone

based off of project repo: https://github.com/udacity/CarND-Capstone

This is the repo for my final Udacity Self Driving Car Nano-degree Capstone Project

## Notes to Review

This is an individual submission by:

Brian Law - bpl.law@gmail.com


### Introduction


As part of this project, we are required to build out a full car control system.

### Main Nodes

#### Waypoint Updater

Finds waypoints ahead of the car:

This service subscribes to:
- `/base_waypoints`
- `/current_pose`
- `'/traffic_waypoint` - to implement

and publishes waypoints to:
- `/final_waypointss`

##### Tasks

Find a waypoints ahead of vehicle by looking at the current pose finding it in the base_waypoints.


#### DBW Node

Receive Final waypoints and and publish commands to twist controller:
TODO - May need to edit cpp code to make it drive a bit smoother

This service subscribes to:
- `/final_waypoints`
- `/vehicle/dbw_enabled` - to find out if it is self driving or not

and publishes commands to:
- `/vehicle/throttle_cmd`
- `/vehicle/brake_cmd`
- `/vehicle/stearing_cmd`

##### Tasks:

Write functions to convert the waypoints into a target throttle, break and stearing settings.
Helper functions are procided in the form of `pid` and `lowpass ` for deciding acceleration and `yaw_controller` for steering.

#### Traffic Light Detection

This service subscribes to:
- `/base_waypoints`
- `/current_pose`
- `/image_color`
- `/vehicle/traffic_lights`

and publishes index of waypoint to:
- `/traffic_waypoint`

##### Tasks:

Find the nearest traffic light from the base_waypoints. Based on the image_color determine whether it is necessary to stop.


###### Approach:

We used the Bosch Traffic Light Dataset to build a classifier leveraging the tensorflow object detection API and also using a pretrained faster rcnn inception v2 model.

We experimented with a ssd model but it didn't work as well with small sized traffic lights.

#### Object Detection Lab

Build an object detector for car

##### Tasks:

Code up an object detector module:
