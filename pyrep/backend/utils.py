from threading import Lock
from typing import List, Tuple
from PyRep.pyrep.backend import sim
from PyRep.pyrep.objects.object import Object
from PyRep.pyrep.objects.shape import Shape
from PyRep.pyrep.objects.dummy import Dummy
from PyRep.pyrep.objects.cartesian_path import CartesianPath
from PyRep.pyrep.objects.joint import Joint
from PyRep.pyrep.objects.vision_sensor import VisionSensor
from PyRep.pyrep.objects.force_sensor import ForceSensor
from PyRep.pyrep.objects.proximity_sensor import ProximitySensor
from PyRep.pyrep.objects.camera import Camera
from PyRep.pyrep.objects.octree import Octree
import numpy as np
import cvxopt

step_lock = Lock()


def to_type(handle: int) -> Object:
    """Converts an object handle to the correct sub-type.

    :param handle: The internal handle of an object.
    :return: The sub-type of this object.
    """
    t = sim.simGetObjectType(handle)
    if t == sim.sim_object_shape_type:
        return Shape(handle)
    elif t == sim.sim_object_dummy_type:
        return Dummy(handle)
    elif t == sim.sim_object_path_type:
        return CartesianPath(handle)
    elif t == sim.sim_object_joint_type:
        return Joint(handle)
    elif t == sim.sim_object_visionsensor_type:
        return VisionSensor(handle)
    elif t == sim.sim_object_forcesensor_type:
        return ForceSensor(handle)
    elif t == sim.sim_object_proximitysensor_type:
        return ProximitySensor(handle)
    elif t == sim.sim_object_camera_type:
        return Camera(handle)
    elif t == sim.sim_object_octree_type:
        return Octree(handle)
    raise ValueError


def script_call(function_name_at_script_name: str,
                script_handle_or_type: int,
                ints=(), floats=(), strings=(), bytes='') -> (
        Tuple[List[int], List[float], List[str], str]):
    """Calls a script function (from a plugin, the main client application,
    or from another script). This represents a callback inside of a script.

    :param function_name_at_script_name: A string representing the function
        name and script name, e.g. myFunctionName@theScriptName. When the
        script is not associated with an object, then just specify the
        function name.
    :param script_handle_or_type: The handle of the script, otherwise the
        type of the script.
    :param ints: The input ints to the script.
    :param floats: The input floats to the script.
    :param strings: The input strings to the script.
    :param bytes: The input bytes to the script (as a string).
    :return: Any number of return values from the called Lua function.
    """
    return sim.simExtCallScriptFunction(
        function_name_at_script_name, script_handle_or_type, list(ints),
        list(floats), list(strings), bytes)


def _is_in_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        pass
    return False


def rot_around_axis(axis, theta):
    if axis == 'x' or axis == 'X':
        MX = np.array([[1, 0, 0, 0],
                       [0, np.cos(theta), -np.sin(theta), 0],
                       [0, np.sin(theta), np.cos(theta), 0],
                       [0, 0, 0, 1]
                       ])
    elif axis == 'y' or axis == 'Y':
        MX = np.array([[np.cos(theta), 0, np.sin(theta), 0],
                       [0, 1, 0, 0],
                       [-np.sin(theta), 0, np.cos(theta), 0],
                       [0, 0, 0, 1]
                       ])
    elif axis == 'z' or axis == 'Z':
        MX = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                       [np.sin(theta), np.cos(theta), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]
                       ])
    else:
        MX = np.identity(4)
    return MX


def trans(position):
    return np.array([[1, 0, 0, position[0]], [0, 1, 0, position[1]], [0, 0, 1, position[2]], [0, 0, 0, 1]])


def KL(a: np.array, b: np.array) -> np.array:
    return np.dot(a[0:3], b[3:6]) + np.dot(a[3:6], b[0:3])


def LP(a: np.array, b: np.array) -> np.array:
    a = a.squeeze()
    b = b.squeeze()
    return np.concatenate([np.cross(a[0:3], b[0:3]), np.cross(a[0:3], b[3:6]) - np.cross(b[0:3], a[3:6])]).reshape(-1,
                                                                                                                   1)


def SC(a: np.array, b: np.array, mode):
    a = a.squeeze()
    b = b.squeeze()
    if mode == 'origin':
        return np.concatenate([np.cross(a[0:3], b[0:3]),
                               np.cross(a[0:3], b[3:6]) + np.cross(a[3:6], b[0:3])]).reshape(-1, 1)
    elif mode == 'dual':
        return np.concatenate([np.cross(a[0:3], b[0:3]) + np.cross(a[3:6], b[3:6]),
                               np.cross(a[0:3], b[3:6])]).reshape(-1, 1)
    else:
        assert TypeError


def quadprog(H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None):
    """
    Input: Numpy arrays, the format follows MATLAB quadprog function:
    https://www.mathworks.com/help/optim/ug/quadprog.html
    Output: Numpy array of the solution
    """
    n_var = H.shape[1]

    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(f.T, tc='d')

    if L is not None or k is not None:
        assert (k is not None and L is not None)
        if lb is not None:
            L = np.vstack([L, -np.eye(n_var)])
            k = np.vstack([k, -lb])

        if ub is not None:
            L = np.vstack([L, np.eye(n_var)])
            k = np.vstack([k, ub])

        L = cvxopt.matrix(L, tc='d')
        k = cvxopt.matrix(k, tc='d')

    if Aeq is not None or beq is not None:
        assert (Aeq is not None and beq is not None)
        Aeq = cvxopt.matrix(Aeq, tc='d')
        beq = cvxopt.matrix(beq, tc='d')
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq)

    return np.array(sol['x'])

