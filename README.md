# Udacity Self Driving Car Nano-Degree Capstone

based off of project repo: https://github.com/udacity/CarND-Capstone

This is the repo for my final Udacity Self Driving Car Nano-degree Capstone Project


### Introduction


As part of this project, we are required to build out a full car control system.

### Main Nodes

#### Waypoint Updater

Finds waypoints ahead of the car:

This service subscribes to:
- `/base_waypoints`
- `/current_pose`

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
#### Waypoint Updater
