from PyRep.pyrep.backend import sim
from PyRep.pyrep.robots.configuration_paths.configuration_path import (
    ConfigurationPath)
import numpy as np
from typing import List, Optional, Union


class ParallelManipulatorConfigurationPath(ConfigurationPath):
    """A path expressed in joint configuration space.

    Paths are retrieved from an :py:class:`ParallelManipulator`, and are associated with the
    ParallelManipulator that generated the path.

    This class is used for executing motion along a path via the
    Reflexxes Motion Library type II or IV. The Reflexxes Motion Library
    provides instantaneous trajectory generation capabilities for motion
    control systems.
    """

    def __init__(self, PM: 'ParallelManipulator',  # type: ignore
                 path_points: Union[List[float], np.ndarray]):
        self._PM = PM
        self._path_points = np.asarray(path_points)
        self._rml_handle: Optional[int] = None
        self._drawing_handle = None
        self._path_done = False
        self._num_joints = PM.get_joint_count()
        self.last_idx = 0
        self.path_lengths = None

    def __len__(self):
        return len(self._path_points) // self._num_joints

    def __getitem__(self, i):
        path_points = self._path_points.reshape(-1, self._num_joints)
        path_points = path_points[i].flatten()
        return self.__class__(PM=self._PM, path_points=path_points)

    def step(self) -> bool:
        """Makes a step along the trajectory.

        This function steps forward a trajectory generation algorithm from
        Reflexxes Motion Library.
        NOTE: This does not step the physics engine. This is left to the user.

        :return: If the end of the trajectory has been reached.
        """
        if self._path_done:
            raise RuntimeError('This path has already been completed. '
                               'If you want to re-run, then call set_to_start.')
        if self._rml_handle is None:
            self._rml_handle = self._get_rml_handle()
        done = self._step_motion() == 1
        self._path_done = done
        return done

    def set_to_start(self) -> None:
        """Sets the arm to the beginning of this path.
        """
        start_config = self._path_points[:len(self._PM.joints)]
        self._PM.set_joint_positions(start_config)
        self._path_done = False

    def set_to_end(self) -> None:
        """Sets the arm to the end of this path.
        """
        final_config = self._path_points[-len(self._PM.joints):]
        self._PM.set_joint_positions(final_config)

    def visualize(self) -> None:
        """Draws a visualization of the path in the scene.

        The visualization can be removed
        with :py:meth:`ConfigurationPath.clear_visualization`.
        """
        if len(self._path_points) <= 0:
            raise RuntimeError("Can't visualise a path with no points.")

        tip = self._PM.get_tip()
        self._drawing_handle = sim.simAddDrawingObject(
            objectType=sim.sim_drawing_lines, size=3, duplicateTolerance=0,
            parentObjectHandle=-1, maxItemCount=99999,
            ambient_diffuse=[1, 0, 1])
        sim.simAddDrawingObjectItem(self._drawing_handle, None)
        init_angles = self._PM.get_joint_positions()
        self._PM.set_joint_positions(
            self._path_points[0: len(self._PM.joints)])
        prev_point = list(tip.get_position())

        for i in range(len(self._PM.joints), len(self._path_points),
                       len(self._PM.joints)):
            points = self._path_points[i:i + len(self._PM.joints)]
            self._PM.set_joint_positions(points)
            p = list(tip.get_position())
            sim.simAddDrawingObjectItem(self._drawing_handle, prev_point + p)
            prev_point = p

        # Set the arm back to the initial config
        self._PM.set_joint_positions(init_angles)

    def clear_visualization(self) -> None:
        """Clears/removes a visualization of the path in the scene.
        """
        if self._drawing_handle is not None:
            sim.simAddDrawingObjectItem(self._drawing_handle, None)

    def _get_rml_handle(self) -> int:
        dt = sim.simGetSimulationTimeStep()
        limits = np.array(self._PM.get_joint_upper_velocity_limits())
        vel_correction = 1.0
        max_vel = self._PM.max_velocity
        max_accel = self._PM.max_acceleration
        max_jerk = self._PM.max_jerk
        self.path_lengths = self._get_path_point_lengths()
        target_pos_vel = [self.path_lengths[-1], 0]
        previous_q = self._path_points[0:len(self._PM.joints)]

        while True:
            pos_vel_accel = [0, 0, 0]
            rMax = 0
            rml_handle = sim.simRMLPos(
                1, 0.0001, -1, pos_vel_accel,
                [max_vel * vel_correction, max_accel, max_jerk],
                [1], target_pos_vel)
            state = 0
            while state == 0:
                state, pos_vel_accel = sim.simRMLStep(rml_handle, dt, 1)
                if state >= 0:
                    pos = pos_vel_accel[0]
                    for i in range(len(self.path_lengths)-1):
                        if self.path_lengths[i] <= pos <= self.path_lengths[i + 1]:
                            t = (pos - self.path_lengths[i]) / (self.path_lengths[i + 1] - self.path_lengths[i])
                            # For each joint
                            offset = len(self._PM.joints) * i
                            p1 = self._path_points[
                                 offset:offset + self._num_joints]
                            offset = len(self._PM.joints) * (i + 1)
                            p2 = self._path_points[
                                 offset:offset + self._num_joints]
                            dx = p2 - p1
                            qs = p1 + dx * t
                            dq = qs - previous_q
                            previous_q = qs
                            r = np.abs(dq / dt) / limits
                            m = np.max(r)
                            if m > rMax:
                                rMax = m
                            break
            sim.simRMLRemove(rml_handle)
            if rMax > 1.001:
                vel_correction = vel_correction / rMax
            else:
                break
        pos_vel_accel = [0, 0, 0]
        rml_handle = sim.simRMLPos(
            1, 0.0001, -1, pos_vel_accel,
            [max_vel*vel_correction, max_accel, max_jerk], [1], target_pos_vel)
        return rml_handle

    def _step_motion(self) -> int:
        dt = sim.simGetSimulationTimeStep()
        # lengths = self._get_path_point_lengths()
        state, posVelAccel = sim.simRMLStep(self._rml_handle, dt, 1)
        if state >= 0:
            pos = posVelAccel[0]
            for i in range(len(self.path_lengths) - 1):
                if self.path_lengths[i] <= pos <= self.path_lengths[i + 1]:
                    t = (pos - self.path_lengths[i]) / (self.path_lengths[i + 1] - self.path_lengths[i])
                    # For each joint
                    offset = len(self._PM.joints) * i
                    p1 = self._path_points[
                         offset:offset + len(self._PM.joints)]
                    offset = self._PM._num_joints * (i + 1)
                    p2 = self._path_points[
                         offset:offset + len(self._PM.joints)]
                    dx = p2 - p1
                    qs = p1 + dx * t
                    self._PM.set_joint_target_positions(qs)
                    break
        if state == 1:
            sim.simRMLRemove(self._rml_handle)
        return state

    def _get_path_point_lengths(self) -> List[float]:
        path_points = self._path_points
        prev_points = path_points[0:len(self._PM.joints)]
        dists = [0.]
        d = 0
        for i in range(len(self._PM.joints), len(self._path_points),
                       len(self._PM.joints)):
            points = path_points[i:i + len(self._PM.joints)]
            d += np.sqrt(np.sum(np.square(prev_points - points)))
            dists.append(d)
            prev_points = points
        return dists
