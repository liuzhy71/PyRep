from PyRep.pyrep.robots.arms.arm import Arm


class UR3(Arm):

    def __init__(self, count: int = 0):
        super().__init__(count, 'UR3', 6)
