"""
An example of how one might use PyRep to create their RL environments.
In this case, the Franka Panda must reach a randomly placed target.
This script contains examples of:
    - RL environment example.
    - Scene manipulation.
    - Environment resets.
    - Setting joint properties (control loop disabled, motor locked at 0 vel)
"""
from os.path import dirname, join, abspath
from PyRep.pyrep import PyRep
from PyRep.pyrep.robots.parallel_manipulators.pm_4ppps import PM_4PPPS
from PyRep.pyrep.objects.shape import Shape
from PyRep.pyrep.objects.dummy import Dummy
import numpy as np

SCENE_FILE = join(dirname(abspath(__file__)), 'cabin_docking_20210709.ttt')

# 固定藏段可以做位置调整
POSITION_MIN = [-0.05, -0.05, 0.05]
POSITION_MAX = [0.05, 0.05, 0.05]
EPISODES = 5
EPISODE_LENGTH = 200


class PM4PPPSPegInHoleEnv(object):
    def __init__(self):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()
        self.agent = PM_4PPPS()
        self.agent.set_control_loop_enabled(True)
        self.agent.set_motor_locked_at_zero_velocity(False)
        self.stationary_cabin = Shape('cabin_static_front_phy')
        self.stationary_cabin_tip = Dummy('cabin_static_face_center')
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()

    def _get_state(self):
        # 获取舱段中心点的位置、速度，以及视觉触感器获取的图像
        return np.concatenate([self.agent.get_joint_positions(),
                               self.agent.get_joint_velocities(),
                               self.stationary_cabin_tip.get_position()])

    def reset(self):
        # 设置舱段初始位姿，随机给定
        pos = list(np.random.uniform(POSITION_MIN, POSITION_MAX))
        self.stationary_cabin.set_position(pos)
        self.agent.set_joint_positions(self.initial_joint_positions)
        return self._get_state()

    def step(self, action):
        self.agent.set
        self.agent.set_joint_target_velocities(action)  # Execute action on arm
        self.pr.step()  # Step the physics simulation
        ax, ay, az = self.agent_ee_tip.get_position()
        tx, ty, tz = self.stationary_cabin_tip.get_position()
        # Reward is negative distance to target
        reward = -np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
        return reward, self._get_state()

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


class Agent(object):
    def act(self, state):
        del state
        return list(np.random.uniform(-1.0, 1.0, size=(7,)))

    def learn(self, replay_buffer):
        del replay_buffer
        pass