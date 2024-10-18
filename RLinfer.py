#!/usr/bin/env python3
# import the required modules
from __future__ import print_function
from codecs import xmlcharrefreplace_errors
from click import exceptions
import pandas as pd
import numpy as np
import csv
import datetime
import statistics
import itertools
import time

import rospy
import rospkg
from std_msgs.msg import String, Bool
from ds4_driver.msg import Status
from user_study.msg import Emotion, Pose, PoseArray
from geometry_msgs.msg import Twist, Point, PointStamped
from control_msgs.msg import JointJog

import tf

import d3rlpy
import torch

BODYPARTS = [
    'nose',
    'left_eye_inner',
    'lef_eye',
    'left_eye_outer',
    'right_eye_inner',
    'right_eye',
    'right_eye_outer',
    'left_ear',
    'right_ear',
    'mouth_left',
    'mouth_right',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_pinky',
    'right_pinky',
    'left_index',
    'right_index',
    'left_thumb',
    'right_thumb',
    'left_hip',
    'right_hip'
]

EXPRESSIONS = [
    'neutral',
    'happy',
    'sad',
    'surprise',
    'fear',
    'disgust',
    'anger',
    'contempt'
]

class RLinfer:
    def __init__(self):
        rospy.init_node('user_study', anonymous = True)        

        self.listener = tf.TransformListener()

        self.base_lin = torch.tensor([0.0])
        self.base_ang = torch.tensor([0.0])
        self.t_base = torch.tensor([0, 0])
        self.R_base = torch.tensor([0])

        self.ee_lin = torch.tensor([0])
        
        self.t_gripper = torch.tensor([0, 0, 0])
        self.R_gripper = torch.tensor([0, 0, 0, 1])

        self.t_handover_goal = torch.tensor([0, 0, 0])
        self.R_handover_goal = torch.tensor([0, 0, 0, 1])

        self.emotion_global = torch.tensor([0.0] * (len(EXPRESSIONS) + 2))
        self.emotion_fetch = torch.tensor([0.0] * (len(EXPRESSIONS) + 2))

        self.body_pose = torch.tensor([0.0] * len(BODYPARTS) * 4)

        self.reward = 0

        self.rl_base_pub = rospy.Publisher('/rl_base_action', String, queue_size = 1) 
        self.rl_arm_pub = rospy.Publisher('/rl_arm_action', String, queue_size = 1)
        self.rl_handover_pub = rospy.Publisher('rl_handover_action', String, queue_size = 1)

        rospy.Subscriber("/status", Status, self.status_callback, queue_size = 1)
        self.status_pub = rospy.Publisher("/status", Status, queue_size = 1)

        self.model_arm = d3rlpy.load_learnable('/home/caris/GohWK/ros_workspace/model/arm.d3')
        self.model_base = d3rlpy.load_learnable('/home/caris/GohWK/ros_workspace/model/base.d3')
        self.model_handover = d3rlpy.load_learnable('/home/caris/GohWK/ros_workspace/model/handover.d3')
        
        self.arm_wrapper = self.model_arm.as_stateful_wrapper(target_return = 1000)
        self.base_wrapper = self.model_base.as_stateful_wrapper(target_return = 1000)
        self.handover_wrapper = self.model_handover.as_stateful_wrapper(target_return = 2000)

    def base_callback(self, msg):
        self.base_lin = torch.tensor([msg.linear.x])
        self.base_ang = torch.tensor([msg.angular.z])

    def arm_callback(self, msg):
        if np.sum(msg.velocities) != 0:
            print(np.sum(msg.velocities))


    def status_callback(self, msg):
        if msg.header.frame_id == 'ds4':
            if msg.button_l1:
                self.pause = True
            elif msg.button_r1:
                self.pause = False
            elif msg.button_cricle:
                rospy.loginfo_throttle(1.0, 'Registered GOOD handover')
                self.reward = 10
            elif msg.button_cross:
                rospy.loginfo_throttle(1.0, 'Registered BAD handover')
                self.reward = -10
            elif msg.button_l2:
                rospy.loginfo_throttle(1.0, 'Registered ROBOT TO HUMAN handover')
            elif msg.button_r2:
                rospy.loginfo_throttle(1.0, 'Registered HUAMN TO ROBOT handover')
            else:
                self.reward = 0

    def tf_callback(self, msg):
        try:
            (trans, rot) = self.listener.lookupTransform('map', 'base_link', rospy.Time(0))
            self.t_base = torch.tensor(trans[0:2])
            self.R_base = torch.tensor([tf.transformations.euler_from_quaternion(rot, axes = 'sxyz')[-1]])
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

        try:
            (trans, rot) = self.listener.lookupTransform('base_link', 'gripper_link', rospy.Time(0))
            self.t_gripper = torch.tensor(trans)
            self.R_gripper = torch.tensor(rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

        try:
            (trans, rot) = self.listener.lookupTransform('base_link', 'handover_goal', rospy.Time(0))
            self.t_handover_goal = torch.tensor(trans)
            self.R_handover_goal = torch.tensor(rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

    def emotion_global_callback(self, msg):
        self.emotion_global = torch.tensor(list(msg.expressions) + [msg.valence, msg.arousal])

    def emotion_fetch_callback(self, msg):
        self.emotion_fetch = torch.tensor(list(msg.expressions) + [msg.valence, msg.arousal])

    def body_pose_callback(self, msg):
        self.body_pose = torch.zeros(len(BODYPARTS) * 4)

        # Update the tensor directly
        for i in range(len(BODYPARTS)):
            self.body_pose[4*i:4*i+4] = torch.tensor([
                msg.array[i].x,
                msg.array[i].y,
                msg.array[i].z,
                msg.array[i].confidence
            ], dtype=torch.float32)

    def run_rl_infer(self):
        while not rospy.is_shutdown():
            # Concatenate all tensors into a single observation
            self.base_observation = torch.cat([
                self.base_lin,
                self.base_ang,
                self.t_base,
                self.R_base,
                self.emotion_global,
                self.emotion_fetch,
                self.body_pose
            ], dim = 0)
            
            status = Status()
            status.header.frame_id = 'replay'
            self.base_observation = np.array(self.base_observation)
            self.base_action = self.base_wrapper.predict(self.base_observation, self.reward)
            print(self.base_observation)
            if self.base_action == 2:
                status.axis_left_x = 1.0
                status.button_l1 = 1
            elif self.base_action == 1 or self.base_action == 3:
                status.axis_right_y = 1.0
                status.button_l1 = 1
            self.status_pub.publish(status)

            rospy.Subscriber('/base_controller/command', Twist, self.base_callback, queue_size = 1)
            rospy.Subscriber('/tf', String, self.tf_callback, queue_size = 1)
            rospy.Subscriber("/arm_controller/joint_velocity/command", JointJog, self.arm_callback, queue_size = 1)
            rospy.Subscriber('/emotion/global', Emotion, self.emotion_global_callback, queue_size = 1)
            rospy.Subscriber('/emotion/fetch', Emotion, self.emotion_fetch_callback, queue_size = 1)
            
            rospy.Subscriber('/body_pose', PoseArray, self.body_pose_callback, queue_size = 1)

            self.arm_observation = torch.cat([
                self.ee_lin,
                self.t_gripper,
                self.R_gripper,
                self.emotion_global,
                self.emotion_fetch,
                self.body_pose
            ])

            status = Status()
            status.header.frame_id = 'replay'
            self.arm_observation = np.array(self.arm_observation)
            self.arm_action = self.arm_wrapper.predict(self.arm_observation, self.reward)

            if self.arm_action == 1:
                status.axis_right_y = 1.0
                status.button_l1 = 1
                status.button_r1 = 1
            elif self.arm_action == 2:
                status.axis_right_y = -1.0
                status.button_l1 = 1
                status.button_r1 = 1

            self.handover_observation = torch.cat([
                self.t_handover_goal,
                self.R_handover_goal,
                self.emotion_global,
                self.emotion_fetch,
                self.body_pose
            ])
            
            status = Status()
            status.header.frame_id = 'replay'
            self.handover_observation = np.array(self.handover_observation)
            self.handover_action = self.handover_wrapper.predict(self.handover_observation, self.reward)

            if self.handover_action == 0:
                status.button_square = 1
                status.button_l1 = 1
            elif self.handover_action == 1:
                status.button_triangle = 1
                status.button_l1 = 1
            elif self.handover_action == 2:
                status.button_circle = 1
                status.button_l1 = 1
            self.status_pub.publish(status)


if __name__ == '__main__':
    try:
        predictor = RLinfer()
        predictor.run_rl_infer()
    except rospy.ROSInterruptException:
        pass