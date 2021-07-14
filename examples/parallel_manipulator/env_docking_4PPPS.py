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
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='[%(filename)s] [%(module)s] [%(funcName)s] [line:%(lineno)d] %(levelname)s %(message)s')

logger = logging.getLogger(__name__)


SCENE_FILE = join(dirname(abspath(__file__)), 'cabin_docking_20210713_widergap.ttt')

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
        # self.agent.set
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


pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
# pr.set_simulation_timestep(0.0001)
pr.start()
robot = PM_4PPPS()
waypoints = [Dummy('waypoint{}'.format(i+1)) for i in range(2)]
tip = Dummy('PM_4PPPS_front_center')

print('Planning path to the stationary cabin ...')
path = robot.get_path(position=waypoints[0].get_position(),
                      quaternion=waypoints[0].get_quaternion())
# path.visualize()  # Let's see what the path looks like
# pr.step()
print('Executing plan ...')
done = False
while not done:
    done = path.step()
    pr.step()
    logger.debug('distance: {}'.format(tip.get_pose()-waypoints[0].get_pose()))
# path.clear_visualization()

print('Insert the cabin ...')
path = robot.get_path(position=waypoints[1].get_position(),
                      quaternion=waypoints[1].get_quaternion())
# path.visualize()  # Let's see what the path looks like
# pr.step()
print('Executing plan ...')
done = False
while not done:
    done = path.step()
    pr.step()
    logger.debug('distance: {}'.format(tip.get_pose() - waypoints[1].get_pose()))
# path.clear_visualization()

print('Done ...')
input('Press enter to finish ...')
pr.stop()
pr.shutdown()

