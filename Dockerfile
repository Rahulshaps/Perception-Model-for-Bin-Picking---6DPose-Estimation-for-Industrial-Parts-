# Author: Rahul Sha Pathepur Shankar, Vaishnav Raja & Reza Farrokhi Saray
# CS 5330 Pattern Recognition and Computer Vision - Spring 2025 - Final Project

# This code is part of the 6D pose estimation for the Industrial Plentopic Dataset (IPD).
# This code is used to create a Docker image for the 6D pose estimation project

FROM Rahulsha/6DPoseEstimation:latest
RUN apt-get update
RUN apt-get install -y libegl1-mesa-dev libgles2-mesa-dev libx11-dev libxext-dev libxrender-dev python3-pip
RUN python3 -m pip install --upgrade setuptools --break-system-packages
COPY Installation_requirements.txt .
RUN pip install -r requirements.txt --break-system-packages
