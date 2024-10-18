**Fetch Robot Handover System using Reinforcement Learning**

This repository contains the code and models developed for training and integrating a reinforcement learning (RL)-based handover system for the Fetch robot. The system aims to enhance Human-Robot Interaction (HRI) by enabling the Fetch robot to autonomously execute handover tasks based on real-time observations and predictions.

**Project Overview**

The objective of this project is to develop a system where the Fetch robot can perform adaptive object handovers in collaborative environments using RL models. The system is divided into three main modules:

1. Arm Status Action: Predicts the movement of the robot’s arm (reaching, tucking, stationary).
2. Base Status Action: Predicts the movement of the robot’s base (stationary, rotating, moving toward operator or participant).
3. Handover Status Action: Predicts the handover location (left, middle, right).
4. System Integration: Integrates the trained RL models into the Fetch robot's control system for real-time execution.

**File Descriptions**

1. Arm.ipynb: Jupyter notebook for training the model to predict the arm status action.
2. Base.ipynb: Jupyter notebook for training the model to predict the base status action.
3. Handover.ipynb: Jupyter notebook for training the model to predict the handover status action.
4. RLinfer.py: Python script for performing system integration and deploying the trained models onto the Fetch robot.
