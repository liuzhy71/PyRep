from PyRep.pyrep.objects.object import Object, object_type_to_class
from PyRep.pyrep.const import ObjectType
from PyRep.pyrep.backend import sim


class Dummy(Object):
    """A point with orientation.

    Dummies are multipurpose objects that can have many different applications.
    """

    # 没有 __init__ 方法，就自动调用父类的 __init__
    @staticmethod  # 使方法成为静态方法，这样在调用的时候就能够不用输入参数调用，或者不实例化调用
    def create(size=0.01) -> 'Dummy':
        """Creates a dummy object and inserts in the scene.

        :param size: The size of the dummy object.
        :return: The newly created Dummy.
        """
        handle = sim.simCreateDummy(size, None)
        return Dummy(handle)

    def _get_requested_type(self) -> ObjectType:
        return ObjectType.DUMMY


object_type_to_class[ObjectType.DUMMY] = Dummy
