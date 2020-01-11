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

Finds waypoints ahead of the car to signal where to drive:

This service subscribes to:
- `/base_waypoints`
- `/current_pose`
- `'/traffic_waypoint`

Based on the base waypoints of the car and current pose of the car, the waypoint updater publishes a list of waypoints for the car to follow. The `traffic_waypoint` message comes from the traffic light detector and signals to the car that a red light has been detected.

When a red light has been detected, rather than just using the base waypoints and continuing to follow the circuit, the `self.decelerate_waypoints` method is triggered and will plot a waypoint path to bring the car to a standstill.  

The final waypoints are published to:
- `/final_waypointss`


#### DBW Node

Receive Final waypoints and and publish commands to twist controller:
Optional TODO - May need to edit cpp code to make it drive a bit smoother

This service subscribes to:
- `/final_waypoints`
- `/vehicle/dbw_enabled`

The `/vehicle/dbw_enabled` signals when it is in drive by wire mode. We don't want to continue tracking waypoints and error if it is being driven in manual as that will result in bad cumulative error for the PID controllers.

The final waypoints published are used by the dbw_node to work out the brake, steering and throttle settings in order to stay on the road. The current velocity is smoothed through a low pass filter before it is fed into the controllers as one of the input variables.

For brake, steering and throttle different controllers are used.

For throttle, a PID controller with:
- P = 0.3
- I = 0.1
- D = 0.0

is used to work out the throttle setting based on the difference between the target and current car velocities.

For the steering, a Yaw controller was already provided in the project code and this was utilised in the final submission.  

and publishes commands to:
- `/vehicle/throttle_cmd`
- `/vehicle/brake_cmd`
- `/vehicle/stearing_cmd`


#### Traffic Light Detection

This service subscribes to:
- `/base_waypoints`
- `/current_pose`
- `/image_color`
- `/vehicle/traffic_lights`

The traffic light detector node does most of it's work through the image callback. As the stop line positions are all provided to the car ahead of time, the light detector first ascertains the cars position relative to the know stop lines. 

If it is near a traffic light, the car will try to detect the traffic light and see what state it is in. If it detects red up to `STATE_COUNT_THRESHOLD` times, it will send the stop line coordinates to: 

- `/traffic_waypoint`

*See [Light Detection Readme](ros/src/tl_detector/train_detector/README.md) for details on light detection.


#### Object Detection Lab

Optional TODO:
Build an object detector for car

##### Tasks:

Code up an object detector module:


### Other TODO

- check using rostopic for image harvesting?
- check out model training and special models for special people


### Running Docker container

Most of the development work was done locally rather than using the workspace on Udacity.
The capstone image was adapted for running locally. The required including Pillow 6.2.1 to address an issue with capturing images:

To run the simulator use:

```
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone:local
```

TODO:
In order to use rviz, it is required to use an nvidia image base and also to pass through display capability to the container: 

this needs to be fixed we need an nvidia base image for capstone - need to reverse engineer the base ros image to use nvidia one rather than basic ubuntu image


docker run --env="DISPLAY" -p 4567:4567 -v "$HOME/.Xauthority:/root/.Xauthority:rw" -v $PWD:/capstone -v /tmp/log:/root/.ros/ --device=/dev/dri:/dev/dri  --rm -it capstone:local
```

