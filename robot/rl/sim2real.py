"""
sim2real.py -- joint-angle conversion between the MuJoCo arm (robot/rl scene) and the real
SO-101 in the teammate's frame (robot_control degrees). Table from fit_sim2real.py.

    sim_deg = sign * real_deg + offset        order: pan, lift, elbow, wrist_flex, wrist_roll

Trust level (see fit_sim2real.py for the numbers):
  * pan / lift / elbow / wrist_flex : solid -- wrist origin agrees to 3 mm mean, 8 mm max over 60 poses.
  * wrist_roll sign  : solid -- the two roll axes are exactly anti-parallel (dot = -1.000).
  * wrist_roll OFFSET: NOT verified. It comes from the tool-point phase and assumes the real hand is
    mounted on link5 like the simulated one. It says sim roll 0 == real roll -170 deg (outside the
    real [-27, +17] limits). Check the real hand's thumb direction at a known roll before trusting it.
"""
import json
import os

import numpy as np

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim2real_map.json")
_M = json.load(open(_PATH, encoding="utf-8"))
SIGN = np.array(_M["sign"], float)
OFFSET = np.array(_M["offset_deg"], float)
YAW_DEG = float(_M["yaw_deg"])
T_M = np.array(_M["t_m"], float)


def real_to_sim_rad(q_real_deg):
    """real (robot_control) degrees [5] -> simulated joint radians [5]."""
    return np.radians(SIGN * np.asarray(q_real_deg, float) + OFFSET)


def sim_to_real_deg(q_sim_rad):
    """simulated joint radians [5] -> real (robot_control) degrees [5]."""
    return (np.degrees(np.asarray(q_sim_rad, float)) - OFFSET) / SIGN


def within_real_limits(q_real_deg, limits):
    """limits = [(lo, hi)] * 5 (Config.arm_joints min_deg/max_deg); returns a bool per joint."""
    q = np.asarray(q_real_deg, float)
    return np.array([lo <= v <= hi for v, (lo, hi) in zip(q, limits)])
