from PyRep.pyrep.backend import sim, utils
from PyRep.pyrep.objects.object import Object
from PyRep.pyrep.objects.dummy import Dummy
from PyRep.pyrep.robots.configuration_paths.parallel_manipulator_configuration_path import (
    ParallelManipulatorConfigurationPath)
from PyRep.pyrep.robots.robot_component import RobotComponent
from PyRep.pyrep.objects.cartesian_path import CartesianPath
from PyRep.pyrep.errors import ConfigurationError, ConfigurationPathError, IKError
from PyRep.pyrep.const import ConfigurationPathAlgorithms as Algos
from PyRep.pyrep.const import PYREP_SCRIPT_TYPE
from typing import List, Union
import numpy as np
import warnings


class ParallelManipulator(RobotComponent):
    """
    Base class representing a parallel manipulator.
    对于并联机器人而言，直接控制关节角度不能很好的实现机器人末端的控制，因为可能会出现机构运动冗余的问题。
    但是，总可以选出一些特定的关节，进行运动学控制，实现机器人的正运动学控制。
    所以，在这个类的实现里，分别实现并联机器人的正运动学，以及逆运动学
    """

    def __init__(self,
                 count: int,
                 name: str,
                 num_limbs: int,
                 num_joints: int,
                 real_joint_names: [],
                 virtual_joint_names: [],
                 base_name: str = None,
                 max_velocity=0.4,
                 max_acceleration=1,
                 max_jerk=1):
        """
        count: is used for when we have multiple copies of parallel_manipulators
        name: name of the robot
        num_limbs: number of limbs
        num_joints: number of independent controlled joints
        joint names: must be given, since the joints in parallel manipulator is not serially connected
        base_name: the base name of the robot
        max_velocity: max joint velocity for revolute joints: deg/s
                                         for prismatic joints: m/s
        max_acceleration: 最大加速度，单位依照前面的
        max_jerk: 最大加加速度
        """
        super().__init__(count, name, real_joint_names + virtual_joint_names, base_name)

        # 运动规划参数
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk

        # 运动规划的handle，主要是ik group的handle
        suffix = '' if count == 0 else '#%d' % (count - 1)
        # 这里直接调用了PyRep中Object的init方法，获取Dummy的handle
        self._ik_target = Dummy('%s_target%s' % (name, suffix))
        self._ik_tip = Dummy('%s_tip%s' % (name, suffix))
        self._ik_limb_tip = [Dummy('SPJ_center_tip_{}'.format(i+1)) for i in range(4)]
        self._ik_limb_target = [Dummy('SPJ_center_target_{}'.format(i+1)) for i in range(4)]
        self._ik_group = sim.simGetIkGroupHandle('%s_ik%s' % (name, suffix))
        self._collision_collection = sim.simGetCollectionHandle(
            '%s_collection%s' % (name, suffix))

    def set_ik_element_properties(self,
                                  tip: Dummy,
                                  constraint_x=True,
                                  constraint_y=True,
                                  constraint_z=True,
                                  constraint_alpha_beta=True,
                                  constraint_gamma=True) -> None:
        constraints = 0
        if constraint_x:
            constraints |= sim.sim_ik_x_constraint
        if constraint_y:
            constraints |= sim.sim_ik_y_constraint
        if constraint_z:
            constraints |= sim.sim_ik_z_constraint
        if constraint_alpha_beta:
            constraints |= sim.sim_ik_alpha_beta_constraint
        if constraint_gamma:
            constraints |= sim.sim_ik_gamma_constraint
        sim.simSetIkElementProperties(
            ikGroupHandle=self._ik_group,
            tipDummyHandle=tip.get_handle(),
            constraints=constraints,
            precision=None,
            weight=None,
        )

    def set_ik_group_properties(self,
                                resolution_method='pseudo_inverse',
                                max_iterations: int = 1,
                                dls_damping: float = 0.1) -> None:
        try:
            res_method = {'pseudo_inverse': sim.sim_ik_pseudo_inverse_method,
                          'damped_least_squares': sim.sim_ik_damped_least_squares_method,
                          'jacobian_transpose': sim.sim_ik_jacobian_transpose_method}[resolution_method]
        except KeyError:
            raise Exception('Invalid resolution method,'
                            'Must be one of ["pseudo_inverse" | "damped_least_squares" | "jacobian_transpose"]')
        sim.simSetIkGroupProperties(
            ikGroupHandle=self._ik_group,
            resolutionMethod=res_method,
            maxIterations=max_iterations,
            damping=dls_damping
        )

    def solve_ik_via_sampling(self,
                              position: Union[List[float], np.ndarray],
                              euler: Union[List[float], np.ndarray] = None,
                              quaternion: Union[List[float], np.ndarray] = None,
                              ignore_collisions: bool = False,
                              trials: int = 300,
                              max_configs: int = 1,
                              distance_threshold: float = 0.65,
                              max_time_ms: int = 10,
                              relative_to: Object = None
                              ) -> np.ndarray:
        """Solves an IK group and returns the calculated joint values.

        This IK method performs a random searches for manipulator configurations
        that matches the given end-effector pose in space. When the tip pose
        is close enough then IK is computed in order to try to bring the
        tip onto the target. This is the method that should be used when
        the start pose is far from the end pose.

        We generate 'max_configs' number of samples within X number of 'trials',
        before ranking them according to angular distance.

        Must specify either rotation in euler or quaternions, but not both!

        :param position: The x, y, z position of the target.
        :param euler: The x, y, z orientation of the target (in radians).
        :param quaternion: A list containing the quaternion (x,y,z,w).
        :param ignore_collisions: If collision checking should be disabled.
        :param trials: The maximum number of attempts to reach max_configs.
        :param max_configs: The maximum number of configurations we want to
            generate before sorting them.
        :param distance_threshold: Distance indicating when IK should be
            computed in order to try to bring the tip onto the target.
        :param max_time_ms: Maximum time in ms spend searching for
            each configuation.
        :param relative_to: Indicates relative to which reference frame we want
            the target pose. Specify None to retrieve the absolute pose,
            or an Object relative to whose reference frame we want the pose.
        :raises: ConfigurationError if no joint configuration could be found.

        :return: 'max_configs' number of joint configurations, ranked according
            to angular distance.
        """
        if not ((euler is None) ^ (quaternion is None)):
            raise ConfigurationError(
                'Specify either euler or quaternion values, but not both.')

        prev_pose = self._ik_target.get_pose()
        self._ik_target.set_position(position, relative_to)
        if euler is not None:
            self._ik_target.set_orientation(euler, relative_to)
        elif quaternion is not None:
            self._ik_target.set_quaternion(quaternion, relative_to)

        handles = [j.get_handle() for j in self.joints]
        cyclics, intervals = self.get_joint_intervals()
        low_limits, max_limits = list(zip(*intervals))
        # If there are huge intervals, then limit them
        low_limits = np.maximum(low_limits, -np.pi * 2).tolist()
        max_limits = np.minimum(max_limits, np.pi * 2).tolist()

        collision_pairs = []
        if not ignore_collisions:
            collision_pairs = [self._collision_collection, sim.sim_handle_all]

        metric = joint_options = None
        valid_joint_positions = []
        for i in range(trials):
            config = sim.simGetConfigForTipPose(
                self._ik_group, handles, distance_threshold, int(max_time_ms),
                metric, collision_pairs, joint_options, low_limits, max_limits)
            if len(config) > 0:
                valid_joint_positions.append(config)
            if len(valid_joint_positions) >= max_configs:
                break

        self._ik_target.set_pose(prev_pose)
        if len(valid_joint_positions) == 0:
            raise ConfigurationError(
                'Could not find a valid joint configuration for desired '
                'end effector pose.')

        if len(valid_joint_positions) > 1:
            current_config = np.array(self.get_joint_positions())
            # Sort based on angular distance
            valid_joint_positions.sort(
                key=lambda x: np.linalg.norm(current_config - x))

        return np.array(valid_joint_positions)

    def get_configs_for_tip_pose(self,
                                 position: Union[List[float], np.ndarray],
                                 euler: Union[List[float], np.ndarray] = None,
                                 quaternion: Union[List[float], np.ndarray] = None,
                                 ignore_collisions=False,
                                 trials=300,
                                 max_configs=60,
                                 relative_to: Object = None
                                 ) -> List[List[float]]:
        """Gets a valid joint configuration for a desired end effector pose.
        Must specify either rotation in euler or quaternions, but not both!
        :param position: The x, y, z position of the target.
        :param euler: The x, y, z orientation of the target (in radians).
        :param quaternion: A list containing the quaternion (x,y,z,w).
        :param ignore_collisions: If collision checking should be disabled.
        :param trials: The maximum number of attempts to reach max_configs
        :param max_configs: The maximum number of configurations we want to
            generate before ranking them.
        :param relative_to: Indicates relative to which reference frame we want
        the target pose. Specify None to retrieve the absolute pose,
        or an Object relative to whose reference frame we want the pose.
        :raises: ConfigurationError if no joint configuration could be found.
        :return: A list of valid joint configurations for the desired
        end effector pose.
        """

        warnings.warn("Please use 'solve_ik_via_sampling' instead.",
                      DeprecationWarning)
        return list(self.solve_ik_via_sampling(
            position, euler, quaternion, ignore_collisions, trials,
            max_configs, relative_to=relative_to))

    def solve_ik_via_jacobian(
            self, position: Union[List[float], np.ndarray],
            euler: Union[List[float], np.ndarray] = None,
            quaternion: Union[List[float], np.ndarray] = None,
            relative_to: Object = None) -> List[float]:
        """Solves an IK group and returns the calculated joint values.

        This IK method performs a linearisation around the current robot
        configuration via the Jacobian. The linearisation is valid when the
        start and goal pose are not too far away, but after a certain point,
        linearisation will no longer be valid. In that case, the user is better
        off using 'solve_ik_via_sampling'.

        Must specify either rotation in euler or quaternions, but not both!

        :param position: The x, y, z position of the target.
        :param euler: The x, y, z orientation of the target (in radians).
        :param quaternion: A list containing the quaternion (x,y,z,w).
        :param relative_to: Indicates relative to which reference frame we want
        the target pose. Specify None to retrieve the absolute pose,
        or an Object relative to whose reference frame we want the pose.
        :return: A list containing the calculated joint values.
        """
        self._ik_target.set_position(position, relative_to)
        if euler is not None:
            self._ik_target.set_orientation(euler, relative_to)
        elif quaternion is not None:
            self._ik_target.set_quaternion(quaternion, relative_to)

        ik_result, joint_values = sim.simCheckIkGroup(
            self._ik_group, [j.get_handle() for j in self.joints])
        if ik_result == sim.sim_ikresult_fail:
            raise IKError('IK failed. Perhaps the distance was between the tip '
                          ' and target was too large.')
        elif ik_result == sim.sim_ikresult_not_performed:
            raise IKError('IK not performed.')
        return joint_values

    def solve_ik(self, position: Union[List[float], np.ndarray],
                 euler: Union[List[float], np.ndarray] = None,
                 quaternion: Union[List[float], np.ndarray] = None,
                 relative_to: Object = None) -> List[float]:
        """Solves an IK group and returns the calculated joint values.

        Must specify either rotation in euler or quaternions, but not both!

        :param position: The x, y, z position of the target.
        :param euler: The x, y, z orientation of the target (in radians).
        :param quaternion: A list containing the quaternion (x,y,z,w).
        :param relative_to: Indicates relative to which reference frame we want
        the target pose. Specify None to retrieve the absolute pose,
        or an Object relative to whose reference frame we want the pose.
        :return: A list containing the calculated joint values.
        """
        warnings.warn("Please use 'solve_ik_via_jacobian' instead.",
                      DeprecationWarning)
        return self.solve_ik_via_jacobian(
            position, euler, quaternion, relative_to)

    def get_path_from_cartesian_path(self, path: CartesianPath
                                     ) -> ParallelManipulatorConfigurationPath:
        """Translate a path from cartesian space, to parallel_manipulator configuration space.

        Note: It must be possible to reach the start of the path via a linear
        path, otherwise an error will be raised.

        :param path: A :py:class:`CartesianPath` instance to be translated to
            a configuration-space path.
        :raises: ConfigurationPathError if no path could be created.

        :return: A path in the parallel_manipulator configuration space.
        """
        handles = [j.get_handle() for j in self.joints]
        _, ret_floats, _, _ = utils.script_call(
            'getPathFromCartesianPath@PyRep', PYREP_SCRIPT_TYPE,
            ints=[path.get_handle(), self._ik_group,
                  self._ik_target.get_handle()] + handles)
        if len(ret_floats) == 0:
            raise ConfigurationPathError(
                'Could not create a path from cartesian path.')
        return ParallelManipulatorConfigurationPath(self, ret_floats)

    def get_linear_path(self, position: Union[List[float], np.ndarray],
                        euler: Union[List[float], np.ndarray] = None,
                        quaternion: Union[List[float], np.ndarray] = None,
                        steps=100,
                        ignore_collisions=False,
                        relative_to: Object = None) -> ParallelManipulatorConfigurationPath:
        """Gets a linear configuration path given a target pose.

        Generates a path that drives a robot from its current configuration
        to its target dummy in a straight line (i.e. shortest path in Cartesian
        space).

        Must specify either rotation in euler or quaternions, but not both!

        :param position: The x, y, z position of the target.
        :param euler: The x, y, z orientation of the target (in radians).
        :param quaternion: A list containing the quaternion (x,y,z,w).
        :param steps: The desired number of path points. Each path point
            contains a robot configuration. A minimum of two path points is
            required. If the target pose distance is large, a larger number
            of steps leads to better results for this function.
        :param ignore_collisions: If collision checking should be disabled.
        :param relative_to: Indicates relative to which reference frame we want
        the target pose. Specify None to retrieve the absolute pose,
        or an Object relative to whose reference frame we want the pose.
        :raises: ConfigurationPathError if no path could be created.

        :return: A linear path in the parallel_manipulator configuration space.
        """
        if not ((euler is None) ^ (quaternion is None)):
            raise ConfigurationPathError(
                'Specify either euler or quaternion values, but not both.')

        prev_pose = self._ik_target.get_pose()
        for i in self._ik_limb_target:
            i.set_parent(self._ik_target)
        robot_objects = sim.simGetObjectsInTree(self._ik_target.get_handle(), sim.sim_handle_all, 0)
        robot_initial_config = sim.simGetConfigurationTree(self._ik_target.get_handle())

        for i in range(len(robot_objects)):
            sim.simResetDynamicObject(robot_objects[i])
        sim.simSetConfigurationTree(robot_initial_config)

        self._ik_target.set_position(position, relative_to)
        if euler is not None:
            self._ik_target.set_orientation(euler, relative_to)
        elif quaternion is not None:
            self._ik_target.set_quaternion(quaternion, relative_to)
        handles = [j.get_handle() for j in self.joints]

        collision_pairs = []
        if not ignore_collisions:
            collision_pairs = [self._collision_collection, sim.sim_handle_all]
        joint_options = None
        ret_floats = sim.generateIkPath(
            self._ik_group, handles, steps, collision_pairs, joint_options)
        sim.simSetConfigurationTree(robot_initial_config)
        self._ik_target.set_pose(prev_pose)
        for i in self._ik_limb_target:
            i.set_parent(self._ik_tip)
        if len(ret_floats) == 0:
            raise ConfigurationPathError('Could not create path.')
        return ParallelManipulatorConfigurationPath(self, ret_floats)

    def get_nonlinear_path(self, position: Union[List[float], np.ndarray],
                           euler: Union[List[float], np.ndarray] = None,
                           quaternion: Union[List[float], np.ndarray] = None,
                           ignore_collisions=False,
                           trials=300,
                           max_configs=1,
                           distance_threshold: float = 0.65,
                           max_time_ms: int = 10,
                           trials_per_goal=1,
                           algorithm=Algos.SBL,
                           relative_to: Object = None
                           ) -> ParallelManipulatorConfigurationPath:
        """Gets a non-linear (planned) configuration path given a target pose.

        A path is generated by finding several configs for a pose, and ranking
        them according to the distance in configuration space (smaller is
        better).

        Must specify either rotation in euler or quaternions, but not both!

        :param position: The x, y, z position of the target.
        :param euler: The x, y, z orientation of the target (in radians).
        :param quaternion: A list containing the quaternion (x,y,z,w).
        :param ignore_collisions: If collision checking should be disabled.
        :param trials: The maximum number of attempts to reach max_configs.
            See 'solve_ik_via_sampling'.
        :param max_configs: The maximum number of configurations we want to
            generate before sorting them. See 'solve_ik_via_sampling'.
        :param distance_threshold: Distance indicating when IK should be
            computed in order to try to bring the tip onto the target.
            See 'solve_ik_via_sampling'.
        :param max_time_ms: Maximum time in ms spend searching for
            each configuation. See 'solve_ik_via_sampling'.
        :param trials_per_goal: The number of paths per config we want to trial.
        :param algorithm: The algorithm for path planning to use.
        :param relative_to: Indicates relative to which reference frame we want
        the target pose. Specify None to retrieve the absolute pose,
        or an Object relative to whose reference frame we want the pose.
        :raises: ConfigurationPathError if no path could be created.

        :return: A non-linear path in the parallel_manipulator configuration space.
        """

        handles = [j.get_handle() for j in self.joints]

        try:
            configs = self.solve_ik_via_sampling(
                position, euler, quaternion, ignore_collisions, trials,
                max_configs, distance_threshold, max_time_ms, relative_to)
        except ConfigurationError as e:
            raise ConfigurationPathError('Could not create path.') from e

        _, ret_floats, _, _ = utils.script_call(
            'getNonlinearPath@PyRep', PYREP_SCRIPT_TYPE,
            ints=[self._collision_collection, int(ignore_collisions),
                  trials_per_goal] + handles,
            floats=configs.flatten().tolist(),
            strings=[algorithm.value])

        if len(ret_floats) == 0:
            raise ConfigurationPathError('Could not create path.')
        return ParallelManipulatorConfigurationPath(self, ret_floats)

    def get_path(self, position: Union[List[float], np.ndarray],
                 euler: Union[List[float], np.ndarray] = None,
                 quaternion: Union[List[float], np.ndarray] = None,
                 ignore_collisions=False,
                 trials=300,
                 max_configs=1,
                 distance_threshold: float = 0.65,
                 max_time_ms: int = 10,
                 trials_per_goal=1,
                 algorithm=Algos.SBL,
                 relative_to: Object = None
                 ) -> ParallelManipulatorConfigurationPath:
        """Tries to get a linear path, failing that tries a non-linear path.

        Must specify either rotation in euler or quaternions, but not both!

        :param position: The x, y, z position of the target.
        :param euler: The x, y, z orientation of the target (in radians).
        :param quaternion: A list containing the quaternion (x,y,z,w).
        :param ignore_collisions: If collision checking should be disabled.
        :param trials: The maximum number of attempts to reach max_configs.
            See 'solve_ik_via_sampling'.
        :param max_configs: The maximum number of configurations we want to
            generate before sorting them. See 'solve_ik_via_sampling'.
        :param distance_threshold: Distance indicating when IK should be
            computed in order to try to bring the tip onto the target.
            See 'solve_ik_via_sampling'.
        :param max_time_ms: Maximum time in ms spend searching for
            each configuation. See 'solve_ik_via_sampling'.
        :param trials_per_goal: The number of paths per config we want to trial.
        :param algorithm: The algorithm for path planning to use.
        :param relative_to: Indicates relative to which reference frame we want
        the target pose. Specify None to retrieve the absolute pose,
        or an Object relative to whose reference frame we want the pose.

        :raises: ConfigurationPathError if neither a linear or non-linear path
            can be created.
        :return: A linear or non-linear path in the parallel_manipulator configuration space.
        """
        try:
            p = self.get_linear_path(position, euler, quaternion,
                                     ignore_collisions=ignore_collisions,
                                     relative_to=relative_to)
            return p
        except ConfigurationPathError:
            pass  # Allowed. Try again, but with non-linear.

        # This time if an exception is thrown, we dont want to catch it.
        p = self.get_nonlinear_path(
            position, euler, quaternion, ignore_collisions, trials, max_configs,
            distance_threshold, max_time_ms, trials_per_goal, algorithm,
            relative_to)
        return p

    def get_tip(self) -> Dummy:
        """Gets the tip of the parallel_manipulator.

        Each parallel_manipulator is required to have a tip for path planning.

        :return: The tip of the parallel_manipulator.
        """
        return self._ik_tip

    def get_target(self) -> Dummy:
        """
        Gets the tip of the parallel manipulator.

        Each parallel_manipulator is required to have a tip for path planning.

        :return: The tip of the parallel manipulator..
        """
        return self._ik_target

    def get_jacobian(self):
        """Calculates the Jacobian.

        :return: the row-major Jacobian matix.
        """
        self._ik_target.set_matrix(self._ik_tip.get_matrix())
        sim.simCheckIkGroup(self._ik_group,
                            [j.get_handle() for j in self.joints])
        jacobian, (rows, cols) = sim.simGetIkGroupMatrix(self._ik_group, 0)
        jacobian = np.array(jacobian).reshape((rows, cols), order='F')
        return jacobian

    def check_parallel_manipulator_collision(self, obj: 'Object' = None) -> bool:
        """Checks whether two entities are colliding.

        :param obj: The other collidable object to check collision against,
            or None to check against all collidable objects. Note that objects
            must be marked as collidable!
        :return: If the object is colliding.
        """
        handle = sim.sim_handle_all if obj is None else obj.get_handle()
        return sim.simCheckCollision(self._collision_collection, handle) == 1

    def set_joint_control_mode(self, joint_idx, mode='position'):
        if mode == 'position':
            # enable the control loop
            self.joints[joint_idx].set_control_loop_enabled(True)
        elif mode == 'force':
            # disable the control loop
            self.joints[joint_idx].set_control_loop_enabled(False)
            # maximize velocity
            inf_velocity = 10000
            self.joints[joint_idx].set_joint_target_velocity(inf_velocity)
        else:
            assert TypeError

    def retrieve_parameters(self, name):
        obj_dict = {'handle': []}
        for i in name:
            _, handle = sim.simGetObjectHandle(i)
            obj_dict['handle'].append(handle)

        emptyBuff = bytearray()
        retInts, retFloats, retStrings, retBuffer = sim.simExtCallScriptFunction(
            'getshapeMassandInertia_test', sim.sim_scripttype_customizationscript,
            obj_dict['handle'], [], [], emptyBuff)
        obj_dict['mass'] = retFloats[0]
        obj_dict['icm'] = np.array(retFloats[1:10]).reshape([3, 3])
        # 注意，这里有一个错误，也就是返回的inertia matrix 本身已经是计算了质量了，不是massless的
        obj_dict['COM'] = np.array(retFloats[10:13])
        obj_dict['inertia_matrix_massless'] = obj_dict['icm'] / obj_dict['mass']
        return obj_dict
