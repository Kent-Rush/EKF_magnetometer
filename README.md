# EKF_magnetometer

This was written during my stay at the Kitakyushi Institute of Technology in Fukuoka, Japan.
The BIRDS-4 Detumbling maneuver failed and I created this Extended Kalman Filter forumation as an attempt to estimate
anguler velocity from only mangetometer data so that we could compare with a potentially broken angular velocity sensor.

Sources:
"CubeSat Attitude Determination via Kalman Filtering of Magnetometer and Solar Cell Data" by Erik Babcock and Timothy Bretl
"Attitude Determination for Small Satellites using Magnetometer and Solar Panel Data" by Todd Humphreys

To run:

python clean_ekf.py
