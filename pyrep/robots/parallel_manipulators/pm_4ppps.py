import numpy as np
import math
from PyRep.pyrep.backend import sim
from PyRep.pyrep.backend.utils import trans, rot_around_axis, SC, KL, LP, quadprog
from parallel_manipulator import ParallelManipulator
from PyRep.pyrep.objects.force_sensor import ForceSensor


class PM_4PPPS(ParallelManipulator):
    def __init__(self,
                 count: int = 0,
                 name: str = 'PM_4PPPS'):
        """
        """
        self.name = name
        self.joint_names = [[] for _ in range(4)]
        for i in range(4):
            for joint_dir in ['X', 'Y', 'Z']:
                self.joint_names[i].append('PM_4PPPS_PJ_{}_{}'.format(joint_dir, i + 1))
        joint_names_flat = [joint for limb in self.joint_names for joint in limb]
        super().__init__(count, 'PM_4PPPS', num_limbs=4, num_joints=12, joint_names=joint_names_flat)

        self.force_sensor_names = []
        for i in range(4):
            self.force_sensor_names.append('Force_sensor_{}'.format(i + 1))

        self.force_sensors = [ForceSensor(i) for i in self.force_sensor_names]
