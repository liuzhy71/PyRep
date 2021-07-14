from PyRep.pyrep.robots.parallel_manipulators.parallel_manipulator import ParallelManipulator
from PyRep.pyrep.objects.force_sensor import ForceSensor


class PM_4PPPS(ParallelManipulator):
    def __init__(self,
                 count: int = 0,
                 name: str = 'PM_4PPPS'):
        """
        """
        self.name = name
        self.real_joint_names = [[] for _ in range(4)]
        for i in range(4):
            for joint_dir in ['X', 'Y', 'Z']:
                self.real_joint_names[i].append('PM_4PPPS_PJ_{}_{}'.format(joint_dir, i + 1))
        real_joint_names_flat = [joint for limb in self.real_joint_names for joint in limb]
        self.virtual_joint_names = ['PM_4PPPS_virtual_joint_{}'.format(i + 1) for i in range(6)]
        super().__init__(count, 'PM_4PPPS',
                         num_limbs=4,
                         num_joints=12,
                         real_joint_names=real_joint_names_flat,
                         virtual_joint_names=self.virtual_joint_names)

        self.force_sensor_names = []
        for i in range(4):
            self.force_sensor_names.append('Force_sensor_{}'.format(i + 1))

        self.force_sensors = [ForceSensor(i) for i in self.force_sensor_names]
