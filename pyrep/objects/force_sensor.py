from typing import Tuple, List
from PyRep.pyrep.backend import sim
from PyRep.pyrep.objects.object import Object, object_type_to_class
from PyRep.pyrep.const import ObjectType


class ForceSensor(Object):
    """An object able to measure forces and torques that are applied to it.
    """

    def _get_requested_type(self) -> ObjectType:
        return ObjectType.FORCE_SENSOR

    @classmethod   # 类方法，不需要实例化就可以调用，第一个参数是类的cls参数，可以用来调用类的属性，类的方法，实例化对象
    def create(cls, sensor_size=0.01) -> 'ForceSensor':
        options = 0  # force and torque threshold are disabled
        intParams = [0, 0, 0, 0, 0]
        floatParams = [sensor_size, 0, 0, 0, 0]
        handle = sim.simCreateForceSensor(options=0, intParams=intParams,
                                          floatParams=floatParams, color=None)
        return cls(handle)

    def read(self) -> Tuple[List[float], List[float]]:
        """Reads the force and torque applied to a force sensor.

        :return: A tuple containing the applied forces along the
            sensor's x, y and z-axes, and the torques along the
            sensor's x, y and z-axes.
        """
        _, forces, torques = sim.simReadForceSensor(self._handle)
        return forces, torques


object_type_to_class[ObjectType.FORCE_SENSOR] = ForceSensor
