from PyRep.pyrep.robots.parallel_manipulators import parallel_manipulator
import numpy as np
import math
from PyRep.pyrep.backend import sim
from PyRep.pyrep.backend.utils import *


class PM_4PPPS(object):
    def __init__(self, clientID_in=-1, name=None):
        """

        :type clientID_in: int
        """
        # initialize own parameters
        self.clientID = clientID_in
        self.name = name
        self.simulation_param = {'dt': self.get_simulation_timestep()}
        _, self.handle = sim.simxGetObjectHandle(clientID_in, self.name, sim.simx_opmode_blocking)

        # constant parameters of the rigid bodies
        self.mobile_platform = self.retrieve_parameters(['mobile_platform_phy'])
        self.link_y = self.retrieve_parameters(['L_Y_phy_1'])
        self.link_z = self.retrieve_parameters(['L_Z_phy_1'])
        self.rod = self.retrieve_parameters(['PM_4PPPS_rod_phy_1', 'rod_high_phy_1'])
        self.lx = 0.25
        self.ly = 0.25
        self.delta = np.concatenate([np.concatenate([np.zeros([3, 3]), np.identity(3)], axis=1),
                                     np.concatenate([np.identity(3), np.zeros([3, 3])], axis=1)], axis=0)
        self.spj_pos0 = np.array([[self.lx, -self.ly, 0, 1], [self.lx, self.ly, 0, 1],
                                  [-self.lx, self.ly, 0, 1], [-self.lx, -self.ly, 0, 1]]).T
        self.pcabin0 = np.array([0, 0, 0, 1])
        # FD constants
        self.fmax = 160 * np.ones([12, 1])
        self.fmin = -160 * np.ones([12, 1])
        self.maxvel = 1
        self.eta1 = 0.1
        self.eta2 = 0.05
        self.elas_xy = 3.443752e-3
        self.elas_z = 2.2421e-5
        self.s01 = np.array([0, 0, 0, 1, 0, 0]).reshape(-1, 1)
        self.s12 = np.array([0, 0, 0, 0, 1, 0]).reshape(-1, 1)
        self.s23 = np.array([0, 0, 0, 0, 0, 1]).reshape(-1, 1)

        # IK desire value
        self.s34 = np.zeros([6, 4])
        self.s45 = np.zeros([6, 4])
        self.s56 = np.zeros([6, 4])
        self.saux = [[] for _ in range(4)]
        self.q = np.zeros([4, 3])
        self.qd = np.zeros([4, 6])
        self.qdd = np.zeros([4, 3])
        self.r0plh = None
        self.r0pl = None
        self.v0pl = None
        self.a0pl = None

        # FK variable
        self.s34_measure = np.zeros([6, 4])
        self.s45_measure = np.zeros([6, 4])
        self.s56_measure = np.zeros([6, 4])
        self.saux_measure = [[] for _ in range(4)]
        self.spj_pos = np.zeros([4, 3])
        self.q_measure = np.zeros([4, 3])
        self.qd_measure = np.zeros([4, 6])
        self.qd_measure_old = np.zeros([4, 6])
        self.qdd_measure = np.zeros([4, 3])
        self.major_joint_force_measure = np.zeros(6)
        self.redundant_joint_force_measure = np.zeros(6)
        self.r0pl_measure = None
        self.cabin_p = np.array([0, 0, 0, 1])
        self.cabin_v = np.zeros(3)
        self.cabin_a = np.zeros(3)
        self.cabin_angle = np.zeros(3)
        self.cabin_omega = np.zeros(3)
        self.cabin_alpha = np.zeros(3)

        # FD variables
        self.pr_old = np.zeros([12, 1])
        self.force = None

        # retrieve all the handles of the joints
        self.joint_names = [[] for _ in range(4)]
        for i in range(4):
            for joint_dir in ['X', 'Y', 'Z']:
                self.joint_names[i].append('PM_4PPPS_PJ_{}_{}'.format(joint_dir, i + 1))

        self.joint_handles = np.zeros([4, 3])
        for i in range(len(self.joint_names)):
            for j in range(len(self.joint_names[0])):
                _, self.joint_handles[i, j] = sim.simxGetObjectHandle(clientID_in, self.joint_names[i][j],
                                                                      sim.simx_opmode_blocking)
        self.joint_handles = self.joint_handles.astype(int)
        self.major_joint_idx = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 2)]
        self.redundant_joint_idx = [(i, j) for i in range(4) for j in range(3) if (i, j) not in self.major_joint_idx]

        # self.major_joint_handles = [self.joint_handles[i] for i in self.major_joint_idx]
        #
        # self.redundant_joint_handles = [handle.tolist() for handle in np.nditer(self.joint_handles)
        #                                 if handle not in self.major_joint_handles]

        self.major_joint_forces = np.zeros(6)
        self.redundant_joint_forces = np.zeros(6)

        self.force_sensor_names = []
        for i in range(4):
            self.force_sensor_names.append('Force_sensor_{}'.format(i + 1))

        self.force_sensor_handles = np.zeros(4)
        self.force_sensor_value = []
        for i in range(4):
            _, self.force_sensor_handles[i] = sim.simxGetObjectHandle(clientID_in, self.force_sensor_names[i],
                                                                      sim.simx_opmode_blocking)

    def get_simulation_timestep(self):
        emptyBuff = bytearray()
        code = "timestep = sim.getSimulationTimeStep()\n" \
               "return tostring(timestep)"
        res, retInts, retFloats, retStrings, _ = sim.simxCallScriptFunction(self.clientID, self.name,
                                                                            sim.sim_scripttype_customizationscript,
                                                                            'executeCode_function', [], [],
                                                                            [code],
                                                                            emptyBuff,
                                                                            sim.simx_opmode_blocking)
        return float(retStrings[0])

    def retrieve_parameters(self, name):
        obj_dict = {'handle': []}
        for i in name:
            _, handle = sim.simxGetObjectHandle(self.clientID, i, sim.simx_opmode_blocking)
            obj_dict['handle'].append(handle)

        emptyBuff = bytearray()
        res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(
            self.clientID, self.name, sim.sim_scripttype_customizationscript, 'getshapeMassandInertia_test',
            obj_dict['handle'], [], [], emptyBuff, sim.simx_opmode_blocking)
        obj_dict['mass'] = retFloats[0]
        obj_dict['icm'] = np.array(retFloats[1:10]).reshape([3, 3])
        # 注意，这里有一个错误，也就是返回的inertia matrix 本身已经是计算了质量了，不是massless的
        obj_dict['COM'] = np.array(retFloats[10:13])
        obj_dict['inertia_matrix_massless'] = obj_dict['icm'] / obj_dict['mass']
        return obj_dict

    def joint_control(self, joint_handle, mode='position', value=None):
        if mode == 'position':
            # enable the control loop
            sim.simxSetObjectIntParameter(self.clientID, joint_handle, 2001, 1, sim.simx_opmode_blocking)
            sim.simxSetJointTargetPosition(self.clientID, joint_handle, value, sim.simx_opmode_blocking)
        elif mode == 'force':
            # disable the control loop
            sim.simxSetObjectIntParameter(self.clientID, joint_handle, 2001, 0, sim.simx_opmode_blocking)
            # maximize velocity
            inf_velocity = 10000
            sim.simxSetJointTargetVelocity(self.clientID, joint_handle, inf_velocity, sim.simx_opmode_blocking)
            sim.simxSetJointMaxForce(self.clientID, joint_handle, value, sim.simx_opmode_blocking)
        else:
            assert TypeError

    # 给定末端相对与参考位姿的位置,角度,在参考坐标系下的速度,加速度,角度,角速度,输出当前关节需要达到的位置,速度,加速度
    # 垃圾课题,天天造轮子,如果你看到这段代码,你就知道你的人生毁了
    def IK(self, p: np.ndarray, v: np.ndarray, a: np.ndarray, angle: np.ndarray, omega: np.ndarray, alpha: np.ndarray):
        angleX = angle[0]
        angleY = angle[1]
        angleZ = angle[2]

        self.r0plh = np.matmul(np.matmul(rot_around_axis('x', angleX), rot_around_axis('y', angleY)),
                               rot_around_axis('z', angleZ))
        self.r0pl = self.r0plh[0:3, 0:3]
        t0pl = np.matmul(trans(p), self.r0plh)
        # 球铰位置想对于参考点的位置

        self.v0pl = np.concatenate([omega, v])

        spj_pos = np.matmul(t0pl, self.spj_pos0)
        jp_M = spj_pos - self.spj_pos0
        self.q = jp_M[0:3, :].T
        pcabin = np.matmul(t0pl, self.pcabin0)
        rspj = spj_pos - pcabin.reshape(-1, 1)

        sp34 = np.array([1, 0, 0]).T
        sp45 = np.array([0, math.cos(angleX), math.sin(angleX)]).T
        sp56 = np.array([math.sin(angleY), -math.cos(angleY) * math.sin(angleX), math.cos(angleY) * math.cos(angleX)]).T

        for i in range(4):
            self.s34[:, i] = np.concatenate([sp34, np.cross(rspj[0:3, i], sp34)])
            self.s45[:, i] = np.concatenate([sp45, np.cross(rspj[0:3, i], sp45)])
            self.s56[:, i] = np.concatenate([sp56, np.cross(rspj[0:3, i], sp56)])

        for i in range(4):
            self.saux[i].append(np.concatenate([np.array([1, 0, 0]), np.cross(rspj[0:3, i], np.array([1, 0, 0]))]))
            self.saux[i].append(np.concatenate([np.array([0, 1, 0]), np.cross(rspj[0:3, i], np.array([0, 1, 0]))]))
            self.saux[i].append(np.concatenate([np.array([0, 0, 1]), np.cross(rspj[0:3, i], np.array([0, 0, 1]))]))

        for i in range(4):
            for j in range(3):
                self.qd[i, j] = KL(self.saux[i][j], self.v0pl)

        J_link = np.hstack(
            [self.s01, self.s12, self.s23, self.s34[:, 0].reshape(-1, 1), self.s45[:, 0].reshape(-1, 1),
             self.s56[:, 0].reshape(-1, 1)])
        qd_limb = np.matmul(np.linalg.inv(J_link), self.v0pl)
        for i in range(4):
            self.qd[i, 3] = qd_limb[3]
            self.qd[i, 4] = qd_limb[4]
            self.qd[i, 5] = qd_limb[5]

        LPL = np.zeros([6, 4])
        for i in range(4):
            LPL[:, i] = (
                    LP(self.qd[i, 0] * self.s01, self.qd[i, 1] * self.s12 + self.qd[i, 2] * self.s23 + self.qd[i, 3]
                       * self.s34[:, i].reshape(-1, 1) + self.qd[i, 4] * self.s45[:, i].reshape(-1, 1) + self.qd[
                           i, 5] * self.s56[:, i].reshape(-1, 1)) +
                    LP(self.qd[i, 1] * self.s12,
                       self.qd[i, 2] * self.s23 + self.qd[i, 3] * self.s34[:, i].reshape(-1, 1) + self.qd[i, 4]
                       * self.s45[:, i].reshape(-1, 1) + (self.qd[i, 5] * self.s56[:, i]).reshape(-1, 1)) +
                    LP(self.qd[i, 2] * self.s23,
                       self.qd[i, 3] * self.s34[:, i].reshape(-1, 1) + self.qd[i, 4] * self.s45[:, i].reshape(-1, 1)
                       + self.qd[i, 5] * self.s56[:, i].reshape(-1, 1)) +
                    LP(self.qd[i, 3] * self.s34[:, i].reshape(-1, 1),
                       self.qd[i, 4] * self.s45[:, i].reshape(-1, 1)
                       + self.qd[i, 5] * self.s56[:, i].reshape(-1, 1)) +
                    LP(self.qd[i, 4] * self.s45[:, i].reshape(-1, 1),
                       self.qd[i, 5] * self.s56[:, i].reshape(-1, 1))).squeeze()

        self.a0pl = np.concatenate([alpha, a - np.cross(omega, v)]).reshape(-1, 1)

        for i in range(4):
            for j in range(3):
                LieS = KL(self.saux[i][j], LPL[:, i])
                self.qdd[i, j] = KL(self.saux[i][j], self.a0pl) - LieS

    def full_dynamics(self, eplcm):
        pr = None
        LL = [[] for _ in range(4)]
        for i in range(4):
            LL[i].append(np.zeros([6, 1]))
            LL[i].append(LP(self.qd[i, 0] * self.s01, self.qd[i, 1] * self.s12))
            LL[i].append(LP(self.qd[i, 0] * self.s01, self.qd[i, 1] * self.s12 + self.qd[i, 2] * self.s23) +
                         LP(self.qd[i, 1] * self.s12, self.qd[i, 2] * self.s23))
        _, gravity = sim.simxGetArrayParameter(self.clientID, sim.sim_arrayparam_gravity, sim.simx_opmode_blocking)
        gravity = -np.array(gravity)
        a0l = [[] for _ in range(4)]
        for i in range(4):
            a0l[i].append(self.qdd[i, 0] * self.s01 + LL[i][0])
            a0l[i].append(self.qdd[i, 0] * self.s01 + self.qdd[i, 1] * self.s12 + LL[i][1])
            a0l[i].append(self.qdd[i, 0] * self.s01 + self.qdd[i, 1] * self.s12 + self.qdd[i, 2] * self.s23 + LL[i][2])

        v0l = [[] for _ in range(4)]
        for i in range(4):
            v0l[i].append(self.qd[i, 0] * self.s01)
            v0l[i].append(self.qd[i, 0] * self.s01 + self.qd[i, 1] * self.s12)
            v0l[i].append(self.qd[i, 0] * self.s01 + self.qd[i, 1] * self.s12 + self.qd[i, 2] * self.s23)

        iplcm_r = np.matmul(np.matmul(self.r0pl, self.mobile_platform['icm']), self.r0pl)
        iplO = np.concatenate([np.concatenate([iplcm_r, np.zeros([3, 3])], axis=1),
                               np.concatenate([np.zeros([3, 3]), self.mobile_platform['mass'] * np.eye(3)], axis=1)],
                              axis=0)
        i0plO = np.matmul(iplO, self.a0pl) + SC(self.v0pl, np.matmul(iplO, self.v0pl), 'dual')
        w0plO = np.concatenate([np.array([0, 0, 0]), self.mobile_platform['mass'] * gravity]).reshape(-1, 1)

        e0plO = eplcm
        f0plO = -np.matmul(self.delta, (i0plO + w0plO + e0plO))

        ilcm = []
        i0lO = [[] for _ in range(4)]
        wlcm = []
        w0lO = [[] for _ in range(4)]
        f0lO = [[] for _ in range(4)]
        for obj in [self.link_y, self.link_z, self.rod]:
            ilcm.append(np.concatenate([np.concatenate([obj['icm'], np.zeros([3, 3])], axis=1),
                                        np.concatenate([np.zeros([3, 3]), obj['mass'] * np.eye(3)], axis=1)], axis=0))
            wlcm.append(np.concatenate([np.array([0, 0, 0]), obj['mass'] * gravity]).reshape(-1, 1))
        for j in range(3):
            for i in range(4):
                i0lO[i].append(np.matmul(ilcm[j], a0l[i][j]) + SC(v0l[i][j], np.matmul(ilcm[j], v0l[i][j]), 'dual'))
                w0lO[i].append(wlcm[j])
                f0lO[i].append(-np.matmul(self.delta, (i0lO[i][j] + w0lO[i][j])))

        JauxMT = np.concatenate([self.saux[0][0], self.saux[0][1], self.saux[0][2],
                                 self.saux[1][0], self.saux[1][2], self.saux[2][2]]).reshape(6, 6)

        JauxMT_pinv = np.linalg.pinv(JauxMT)
        H6 = np.linalg.pinv(np.matmul(JauxMT, self.delta))
        H = [[] for _ in range(4)]
        H[0].append(np.concatenate([self.s01, np.zeros([6, 5])], axis=1))
        H[0].append(np.concatenate([self.s01, self.s12, np.zeros([6, 4])], axis=1))
        H[0].append(np.concatenate([self.s01, self.s12, self.s23, np.zeros([6, 3])], axis=1))
        H[1].append(np.concatenate([np.zeros([6, 3]), self.s01, np.zeros([6, 2])], axis=1))
        H22 = np.dot(np.dot(self.s12, self.saux[1][1].reshape(1, -1)), np.linalg.pinv(JauxMT))
        H[1].append(np.concatenate([np.zeros([6, 3]), self.s01, np.zeros([6, 2])], axis=1) + H22)
        H[1].append(np.concatenate([np.zeros([6, 3]), self.s01, self.s23, np.zeros([6, 1])], axis=1) + H22)

        H31 = np.dot(np.dot(self.s01, self.saux[2][0].reshape(1, -1)), JauxMT_pinv)
        H32 = np.dot(np.dot(self.s12, self.saux[2][1].reshape(1, -1)), JauxMT_pinv)
        H[2].append(H31)
        H[2].append(H31 + H32)
        H[2].append(np.concatenate([np.zeros([6, 5]), self.s23], axis=1) + H31 + H32)

        H41 = np.dot(np.dot(self.s01, self.saux[3][0].reshape(1, -1)), JauxMT_pinv)
        H42 = np.dot(np.dot(self.s12, self.saux[3][1].reshape(1, -1)), JauxMT_pinv)
        H43 = np.dot(np.dot(self.s23, self.saux[3][2].reshape(1, -1)), JauxMT_pinv)
        H[3].append(H41)
        H[3].append(H41 + H42)
        H[3].append(H41 + H42 + H43)

        Jaux2 = np.concatenate([self.saux[1][1], self.saux[2][0], self.saux[2][1],
                                self.saux[3][0], self.saux[3][1], self.saux[3][2]]).reshape(6, 6).T
        IG = np.concatenate([np.eye(6), np.dot(Jaux2.T, JauxMT_pinv).T], axis=1)

        ForceB = np.zeros([6, 1])
        for k in range(6):
            ForceB[k] = -KL(f0plO.squeeze(), H6[:, k])
            for i in range(4):
                for j in range(3):
                    ForceB[k] = ForceB[k] - KL(f0lO[i][j].squeeze(), H[i][j][:, k])

        # Force0 = np.dot(np.linalg.pinv(IG) * ForceB).T

        prmin = -self.maxvel * np.ones([12, 1])
        prmax = self.maxvel * np.ones([12, 1])

        W = np.diag((self.fmax - self.fmin).squeeze()) / 2

        JT = np.concatenate([IG[:, 0: 4], IG[:, 6].reshape(-1, 1), IG[:, 4].reshape(-1, 1),
                             IG[:, 7: 9], IG[:, 5].reshape(-1, 1), IG[:, 9: 12]], axis=1)
        J1T = np.dot(JT, W)
        ForceBp = ForceB - np.dot(JT, (self.fmax + self.fmin)) / 2

        sen = self.cal_sensitivity()
        sen = np.abs(sen)
        sen[sen < 0.001] = 0.001

        N11 = np.diag([sen[0], sen[1], sen[2], sen[3], 0, sen[4], 0, 0, sen[5], 0, 0, 0])
        N12 = np.diag(
            [self.elas_xy, self.elas_xy, self.elas_z, self.elas_xy, 0, self.elas_z, 0, 0, self.elas_z, 0, 0, 0])
        N1 = 1000 * np.dot(N11, N12)
        # N2diag = [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1]
        run = 1
        if all(self.pr_old == np.zeros([12, 1])):
            Aeq = J1T
            beq = ForceBp
            Hm = 2 * N1
            f = np.zeros([1, 12])
            pr = quadprog(H=Hm, f=f, L=np.zeros([12, 12]), k=np.zeros([12, 1]), Aeq=Aeq, beq=beq, lb=prmin, ub=prmax)
            self.pr_old = pr
        else:
            while run:
                Wh = np.diag((2 * np.sign(1 / 2 - 1 / 2 * np.sign(self.pr_old - prmin))
                              + 2 * np.sign(
                            1 / 2 + 1 / 2 * np.sign(
                                self.pr_old - prmax)) + 2 * self.eta2).squeeze()) + 2 * N1 * self.eta1
                wh_inv = np.linalg.pinv(Wh)

                d = -2 * prmin * np.sign(1 / 2 - 1 / 2 * np.sign(self.pr_old - prmin)) - 2 * prmax * np.sign(
                    1 / 2 + 1 / 2 * np.sign(self.pr_old - prmax)) - 2 * self.eta2 * self.pr_old
                pr = np.dot(wh_inv,
                            np.dot(-J1T.T,
                                   np.dot(np.linalg.pinv(np.dot(np.dot(-J1T, wh_inv), -J1T.T)),
                                          (ForceBp + np.dot(np.dot(J1T, wh_inv), d)))) - d)
                if np.where(pr.squeeze() > 1)[0].size != 0 and min(prmin) <= -0.7 and max(prmax) >= 0.7:
                    prmin = 0.99 * prmin
                    prmax = 0.99 * prmax
                else:
                    run = 0
            self.pr_old = pr
        self.force = pr * ((self.fmax - self.fmin) / 2) + (self.fmax + self.fmin) / 2
        # self.major_joint_forces = [self.force[0], self.force[1], self.force[2],
        #                            self.force[3], self.force[5], self.force[8]]
        # self.redundant_joint_forces = [self.force[i] for i not in ]

    def step(self):
        """
        这个函数在主要是负责发送同步信号，同时更新机器人状态信息存储器.
        需要更新的信息包括：关节位置，关节速度，关节加速度，关节输出力矩，力传感器数据
        :return:
        """
        self.qd_measure_old = self.qd_measure
        sim.simxSynchronousTrigger(self.clientID)
        emptyBuff = bytearray()

        for i in self.major_joint_idx:
            self.q_measure[i] = sim.simxGetJointPosition(self.clientID, self.joint_handles[i], sim.simx_opmode_blocking)
            code = "velocity = sim.getJointVelocity({})\n".format(self.joint_handles[i]) + \
                   "return tostring(velocity)"
            res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(self.clientID, self.name,
                                                                                        sim.sim_scripttype_childscript,
                                                                                        'executeCode_function', [], [],
                                                                                        [code],
                                                                                        emptyBuff,
                                                                                        sim.simx_opmode_blocking)
            self.qd_measure[i] = float(retStrings[0])
            self.major_joint_force_measure[i] = sim.simxGetJointForce(self.clientID, self.joint_handles[i],
                                                                      sim.simx_opmode_blocking)

        for i in self.redundant_joint_idx:
            self.q_measure[i] = sim.simxGetJointPosition(self.clientID, self.joint_handles[i], sim.simx_opmode_blocking)
            code = "velocity = sim.getJointVelocity({})\n".format(self.joint_handles[i]) + \
                   "return tostring(velocity)"
            res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(self.clientID, self.name,
                                                                                        sim.sim_scripttype_childscript,
                                                                                        'executeCode_function', [], [],
                                                                                        [code],
                                                                                        emptyBuff,
                                                                                        sim.simx_opmode_blocking)
            self.qd_measure[i] = float(retStrings[0])
            self.major_joint_force_measure[i] = sim.simxGetJointForce(self.clientID, self.joint_handles[i],
                                                                      sim.simx_opmode_blocking)
        self.qdd_measure = (self.qd_measure - self.qd_measure_old) / self.simulation_param['dt']

        for i in range(4):
            returnCode, state, forceVector, torqueVector = sim.simxReadForceSensor(self.clientID,
                                                                                   self.force_sensor_handles[i],
                                                                                   sim.simx_opmode_blocking)
            self.force_sensor_value.append(forceVector)

    def forward_kinematics(self):
        q11 = self.q_measure[0, 0]
        q12 = self.q_measure[0, 1]
        q13 = self.q_measure[0, 2]
        q21 = self.q_measure[1, 0]
        q23 = self.q_measure[1, 2]
        q33 = self.q_measure[2, 2]
        q22 = -2 * self.ly + q12 + np.sqrt(4 * self.ly ** 2 - q11 ** 2 - q13 ** 2 + 2 * q11 * q21 - q21 ** 2
                                           + 2 * q13 * q23 - q23 ** 2)

        q31 = (1 / (4 * self.ly ** 2 - (q13 - q23) ** 2)) * (8 * self.lx * self.ly ** 2 - 2 * self.lx * q13 ** 2
                                                             + 4 * self.ly ** 2 * q21 - q13 ** 2 * q21
                                                             + 4 * self.lx * q13 * q23 + q11 * q13 * q23
                                                             + q13 * q21 * q23 - 2 * self.lx * q23 ** 2
                                                             - q11 * q23 ** 2 - 2 * np.sqrt(
                    4 * self.ly ** 2 - q11 ** 2 - q13 ** 2 + 2 * q11 * q21 - q21 ** 2 + 2 * q13 * q23 - q23 ** 2) *
                                                             np.sqrt((-self.lx ** 2) * (-4 * self.ly ** 2 + (
                                                                     q13 - q23) ** 2) - self.ly ** 2 * (
                                                                             q23 - q33) ** 2) - q11 * q13 * q33 +
                                                             q13 * q21 * q33 + q11 * q23 * q33 - q21 * q23 * q33)

        q32 = (1 / ((2 * self.ly + q13 - q23) * (2 * self.ly - q13 + q23))) * (-8 * self.ly ** 3 + 4 * self.ly ** 2 * (
                q12 + np.sqrt(4 * self.ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2)) +
                                                                               2 * self.ly * (q13 - q23) ** 2 - (
                                                                                       q13 - q23) * (
                                                                                       q12 * (q13 - q23) + np.sqrt(
                                                                                   4 * self.ly ** 2 - (
                                                                                           q11 - q21) ** 2 - (
                                                                                           q13 - q23) ** 2) * (
                                                                                               q13 - q33)) +
                                                                               2 * (-q11 + q21) * np.sqrt(
                    self.lx ** 2 * (2 * self.ly + q13 - q23) * (2 * self.ly - q13 + q23) -
                    self.ly ** 2 * (q23 - q33) ** 2))
        self.spj_pos[:, 0] = self.spj_pos0[0:3, 0] + np.array([q11, q12, q13])
        self.spj_pos[:, 1] = self.spj_pos0[0:3, 1] + np.array([q21, q22, q23])
        self.spj_pos[:, 2] = self.spj_pos0[0:3, 2] + np.array([q31, q32, q33])

        Cm = (self.spj_pos[:, 0] + self.spj_pos[:, 2]) / 2
        self.cabin_p = Cm - self.pcabin0
        q4 = 2 * Cm - self.spj_pos[:, 1] - self.spj_pos0[0:3, 3]
        self.spj_pos[:, 3] = self.spj_pos0[0:3] + q4
        xp_unnormalized = self.spj_pos[:, 0] - self.spj_pos[:, 3]
        xp = xp_unnormalized / np.linalg.norm(xp_unnormalized)
        yp_unnormalized = self.spj_pos[:, 1] - self.spj_pos[:, 0]
        yp = yp_unnormalized / np.linalg.norm(yp_unnormalized)
        zp = np.cross(xp, yp)
        self.r0pl_measure = np.array([xp, yp, zp]).T
        self.cabin_angle[0] = np.arctan(self.r0pl_measure[2, 2] / self.r0pl_measure[1, 2])
        self.cabin_angle[1] = np.arcsin(self.r0pl_measure[0, 2])
        self.cabin_angle[2] = np.arctan(self.r0pl_measure[1, 2] / self.r0pl_measure[0, 0])

        r = self.spj_pos - self.cabin_p.reshape(-1, 1)

        sp34 = np.array([1, 0, 0]).T
        sp45 = np.array([0, math.cos(self.cabin_angle[0]), math.sin(self.cabin_angle[0])]).T
        sp56 = np.array([math.sin(self.cabin_angle[1]), -math.cos(self.cabin_angle[1]) *
                         math.sin(self.cabin_angle[0]),
                         math.cos(self.cabin_angle[1]) * math.cos(self.cabin_angle[0])]).T

        for i in range(4):
            self.s34_measure[:, i] = np.concatenate([sp34, np.cross(r[0:3, i], sp34)])
            self.s45_measure[:, i] = np.concatenate([sp45, np.cross(r[0:3, i], sp45)])
            self.s56_measure[:, i] = np.concatenate([sp56, np.cross(r[0:3, i], sp56)])

        for i in range(4):
            self.saux_measure[i].append(np.concatenate([np.array([1, 0, 0]), np.cross(r[0:3, i], np.array([1, 0, 0]))]))
            self.saux_measure[i].append(np.concatenate([np.array([0, 1, 0]), np.cross(r[0:3, i], np.array([0, 1, 0]))]))
            self.saux_measure[i].append(np.concatenate([np.array([0, 0, 1]), np.cross(r[0:3, i], np.array([0, 0, 1]))]))

        Jaux = [self.saux_measure[idx[0]][idx[1]] for idx in self.major_joint_idx]
        qdm = [self.qd_measure[i] for i in self.major_joint_idx]
        v0pl = np.matmul(np.linalg.inv(np.matmul(Jaux, self.delta)), qdm)
        self.cabin_v = v0pl[0:3]
        self.cabin_omega = v0pl[3:6]

        qdmm = [self.qdd_measure[i] for i in self.major_joint_idx]

        J_link = np.hstack(
            [self.s01, self.s12, self.s23, self.s34_measure[:, 0].reshape(-1, 1), self.s45_measure[:, 0].reshape(-1, 1),
             self.s56_measure[:, 0].reshape(-1, 1)])
        qd_limb = np.matmul(np.linalg.inv(J_link), v0pl)
        for i in range(4):
            self.qd_measure[i, 3] = qd_limb[3]
            self.qd_measure[i, 4] = qd_limb[4]
            self.qd_measure[i, 5] = qd_limb[5]

        LPL = np.zeros([6, 4])
        for i in range(4):
            LPL[:, i] = (
                    LP(self.qd_measure[i, 0] * self.s01,
                       self.qd_measure[i, 1] * self.s12 + self.qd_measure[i, 2] * self.s23 + self.qd_measure[i, 3]
                       * self.s34[:, i].reshape(-1, 1) + self.qd_measure[i, 4] * self.s45[:, i].reshape(-1, 1) +
                       self.qd_measure[
                           i, 5] * self.s56[:, i].reshape(-1, 1)) +
                    LP(self.qd_measure[i, 1] * self.s12,
                       self.qd_measure[i, 2] * self.s23 + self.qd_measure[i, 3] * self.s34[:, i].reshape(-1, 1) +
                       self.qd_measure[i, 4]
                       * self.s45[:, i].reshape(-1, 1) + (self.qd_measure[i, 5] * self.s56[:, i]).reshape(-1, 1)) +
                    LP(self.qd_measure[i, 2] * self.s23,
                       self.qd_measure[i, 3] * self.s34[:, i].reshape(-1, 1) + self.qd_measure[i, 4] * self.s45[:,
                                                                                                       i].reshape(-1, 1)
                       + self.qd_measure[i, 5] * self.s56[:, i].reshape(-1, 1)) +
                    LP(self.qd_measure[i, 3] * self.s34[:, i].reshape(-1, 1),
                       self.qd_measure[i, 4] * self.s45[:, i].reshape(-1, 1)
                       + self.qd_measure[i, 5] * self.s56[:, i].reshape(-1, 1)) +
                    LP(self.qd_measure[i, 4] * self.s45[:, i].reshape(-1, 1),
                       self.qd_measure[i, 5] * self.s56[:, i].reshape(-1, 1))).squeeze()
        # 一个想不明白的问题，就是我用主关节计算了副关节的位置，但是副关节的位置本身也可以获得，那我这个模型的意义何在。
        # 提出一个假设，那就是在仿真模型中，实际的变形会传导到力传感的位姿变化，错误，只要力传感器向动平台传递力就会抖动。
        # 那么由于力传感器的抖动，实际上关节能够到达的位置也需要调整，所以最后的力位混合控制部分就是效果不好了
        #

    def redundant_joint_forces(self):
        #
        pass

    def cal_sensitivity(self):
        q11 = self.q[0][0]
        q12 = self.q[0][1]
        q13 = self.q[0][2]
        q21 = self.q[1][0]
        q23 = self.q[1][2]
        q33 = self.q[2][2]

        lx = self.lx
        ly = self.ly

        s11 = 1 / 2 * ((((q11 - q21) * (4 * ly ** 2 - (q13 - q23) * (q13 - q33)) + 2 * np.sqrt(
            4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) * np.sqrt(lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) -
                                                                         ly ** 2 * (q23 - q33) ** 2)) * (
                                8 * ly ** 3 - 4 * ly ** 2 * (
                                2 * q12 + np.sqrt(4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2)) -
                                2 * ly * (q13 - q23) ** 2 + 2 * q12 * (q13 - q23) ** 2 + 2 * (q11 - q21) * np.sqrt(
                            lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) - ly ** 2 * (q23 - q33) ** 2) +
                                np.sqrt(4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) * (q13 - q23) * (
                                        q13 - q33))) / (np.sqrt(
            4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) * (2 * ly + q13 - q23) ** 2 * (
                                                                2 * ly - q13 + q23) ** 2) + (
                               1 / ((-4 * ly ** 2 + (q13 - q23) ** 2) ** 2)) * (
                               -4 * ly ** 2 + (2 * (-q11 + q21) * np.sqrt(
                           lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) - ly ** 2 * (q23 - q33) ** 2)) / np.sqrt(
                           4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) + (q13 - q23) * (q13 -
                                                                                               2 * q23 + q33)) * (
                               -4 * ly ** 2 * (2 * lx + q11 + q21) + 2 * np.sqrt(
                           4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) * np.sqrt(
                           lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) -
                           ly ** 2 * (q23 - q33) ** 2) + (q13 - q23) * (
                                       2 * lx * (q13 - q23) + q21 * (q13 - q33) + q11 * (q13 - 2 * q23 + q33))))

        s12 = 2 * q12 + (-8 * ly ** 3 + 4 * ly ** 2 * np.sqrt(4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) +
                         2 * ly * (q13 - q23) ** 2 + 2 * (-q11 + q21) * np.sqrt(
                    lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) - ly ** 2 * (q23 - q33) ** 2) -
                         np.sqrt(4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) * (q13 - q23) * (q13 - q33)) / (
                      (2 * ly + q13 - q23) * (2 * ly - q13 + q23))

        s13 = 1 / 4 * (8 * (
                q12 + (-8 * ly ** 3 + 4 * ly ** 2 * np.sqrt(4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) +
                       2 * ly * (q13 - q23) ** 2 + 2 * (-q11 + q21) * np.sqrt(
                    lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) - ly ** 2 * (q23 - q33) ** 2) -
                       np.sqrt(4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) * (q13 - q23) * (q13 - q33)) / (
                        2 * (2 * ly + q13 - q23) * (2 * ly - q13 + q23))) * ((-8 * ly ** 3 +
                                                                              4 * ly ** 2 * np.sqrt(
                    4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) +
                                                                              2 * ly * (q13 - q23) ** 2 +
                                                                              2 * (-q11 + q21) * np.sqrt(
                    lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) - ly ** 2 * (q23 - q33) ** 2) -
                                                                              np.sqrt(
                                                                                  4 * ly ** 2 - (q11 - q21) ** 2 - (
                                                                                          q13 - q23) ** 2) * (
                                                                                      q13 - q23) * (
                                                                                      q13 - q33)) / (2 * (
                2 * ly + q13 - q23) * (
                                                                                                             2 * ly - q13 + q23) ** 2) - (
                                                                                     -8 * ly ** 3 + 4 * ly ** 2 * np.sqrt(
                                                                                 4 * ly ** 2 - (
                                                                                         q11 - q21) ** 2 - (
                                                                                         q13 - q23) ** 2) +
                                                                                     2 * ly * (q13 - q23) ** 2 +
                                                                                     2 * (-q11 + q21) * np.sqrt(
                                                                                 lx ** 2 * (4 * ly ** 2 - (
                                                                                         q13 - q23) ** 2) - ly ** 2 * (
                                                                                         q23 - q33) ** 2) -
                                                                                     np.sqrt(4 * ly ** 2 - (
                                                                                             q11 - q21) ** 2 - (
                                                                                                     q13 - q23) ** 2) * (
                                                                                             q13 -
                                                                                             q23) * (
                                                                                             q13 - q33)) / (
                                                                                     2 * (
                                                                                     2 * ly + q13 - q23) ** 2 * (
                                                                                             2 * ly - q13 +
                                                                                             q23)) + (
                                                                                     4 * ly * (q13 - q23) - (
                                                                                     4 * ly ** 2 * (
                                                                                     q13 - q23)) / np.sqrt(
                                                                                 4 * ly ** 2 - (
                                                                                         q11 - q21) ** 2 - (
                                                                                         q13 - q23) ** 2) -
                                                                                     np.sqrt(4 * ly ** 2 - (
                                                                                             q11 - q21) ** 2 - (
                                                                                                     q13 - q23) ** 2) * (
                                                                                             q13 - q23) - (
                                                                                             2 * lx ** 2 * (
                                                                                             -q11 + q21) * (
                                                                                                     q13 - q23)) / np.sqrt(

                                                                                 lx ** 2 * (4 * ly ** 2 - (
                                                                                         q13 - q23) ** 2) - ly ** 2 * (
                                                                                         q23 - q33) ** 2) -
                                                                                     np.sqrt(4 * ly ** 2 - (
                                                                                             q11 - q21) ** 2 - (
                                                                                                     q13 - q23) ** 2) * (
                                                                                             q13 -
                                                                                             q33) + ((
                                                                                                             q13 - q23) ** 2 * (
                                                                                                             q13 - q33)) / np.sqrt(
                                                                                 4 * ly ** 2 - (
                                                                                         q11 - q21) ** 2 - (
                                                                                         q13 - q23) ** 2)) / (
                                                                                     2 * (2 * ly + q13 -
                                                                                          q23) * (
                                                                                             2 * ly - q13 + q23))) + 2 * (
                               q13 + q33) + (
                               1 / ((-4 * ly ** 2 + (q13 - q23) ** 2) ** 2)) *
                       2 * (2 * lx * (q13 - q23) + (2 * lx + q11 + q21) * (q13 - q23) - (
                        2 * lx ** 2 * np.sqrt(
                    4 * ly ** 2 - (q11 - q21) ** 2 - (
                            q13 - q23) ** 2) * (
                                q13 - q23)) / np.sqrt(
                    lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) - ly ** 2 * (q23 - q33) ** 2) - (
                                    2 * (q13 - q23) * np.sqrt(
                                lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) - ly ** 2 * (q23 - q33) ** 2)) / np.sqrt(
                    4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) + q21 * (q13 - q33) +
                            q11 * (q13 - 2 * q23 + q33)) * (-4 * ly ** 2 * (2 * lx + q11 + q21) +
                                                            2 * np.sqrt(
                            4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) * np.sqrt(
                            lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) -
                            ly ** 2 * (q23 - q33) ** 2) + (q13 - q23) * (2 * lx * (q13 - q23) +
                                                                         q21 * (q13 - q33) + q11 * (
                                                                                 q13 - 2 * q23 + q33))) - (
                               1 / ((-4 * ly ** 2 + (q13 - q23) ** 2) ** 3)) * 4 * (q13 - q23) * (
                               -4 * ly ** 2 * (2 * lx + q11 + q21) +
                               2 * np.sqrt(4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) * np.sqrt(
                           lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) - ly ** 2 * (q23 - q33) ** 2) + (
                                       q13 - q23) * (2 * lx * (q13 - q23) +
                                                     q21 * (q13 - q33) + q11 * (q13 - 2 * q23 + q33))) ** 2)

        s21 = 1 / 2 * (((-8 * ly ** 3 +
                         4 * ly ** 2 * (2 * q12 + np.sqrt(
                    4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2)) +
                         2 * ly * (q13 - q23) ** 2 - (q13 - q23) * (2 * q12 * (q13 - q23) +
                                                                    np.sqrt(4 * ly ** 2 - (q11 - q21) ** 2 - (
                                                                            q13 - q23) ** 2) * (q13 -
                                                                                                q33)) +
                         2 * (-q11 + q21) * np.sqrt(
                    lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) -
                    ly ** 2 * (q23 - q33) ** 2)) * ((q11 -
                                                     q21) * (4 * ly ** 2 - (q13 - q23) * (q13 - q33)) +
                                                    2 * np.sqrt(
                    4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) * np.sqrt(
                    lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) - ly ** 2 * (q23 - q33) ** 2))) / (np.sqrt(
            4 * ly ** 2 - (q11 - q21) ** 2 - (
                    q13 - q23) ** 2) * (
                                                                                                          2 * ly + q13 -
                                                                                                          q23) ** 2 * (
                                                                                                          2 * ly - q13 + q23) ** 2) + (

                               1 / ((-4 * ly ** 2 + (q13 - q23) ** 2) ** 2)) * (-4 * ly ** 2 + (
                2 * (q11 - q21) * np.sqrt(
            lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) - ly ** 2 * (
                    q23 - q33) ** 2)) / np.sqrt(
            4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) + (q13 - q23) * (q13 -
                                                                                q33)) * (
                               -4 * ly ** 2 * (2 * lx + q11 + q21) +
                               2 * np.sqrt(4 * ly ** 2 - (q11 - q21) ** 2 - (
                               q13 - q23) ** 2) * np.sqrt(
                           lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) -
                           ly ** 2 * (q23 - q33) ** 2) + (q13 - q23) * (2 * lx * (q13 - q23) +
                                                                        q21 * (q13 - q33) + q11 * (
                                                                                q13 - 2 * q23 + q33))))

        s23 = 1 / 4 * (8 * (
                q12 + (-8 * ly ** 3 + 4 * ly ** 2 * np.sqrt(4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) +
                       2 * ly * (q13 - q23) ** 2 + 2 * (-q11 + q21) * np.sqrt(
                    lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) - ly ** 2 * (q23 - q33) ** 2) -
                       np.sqrt(4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) * (q13 - q23) * (q13 - q33)) / (
                        2 * (2 * ly + q13 - q23) * (2 * ly - q13 + q23))) * (
                               -((-8 * ly ** 3 + 4 * ly ** 2 * np.sqrt(
                                   4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) + 2 * ly * (q13 - q23) ** 2 +
                                  2 * (-q11 + q21) * np.sqrt(
                                           lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) - ly ** 2 * (
                                                   q23 - q33) ** 2) -
                                  np.sqrt(4 * ly ** 2 - (q11 - q21) ** 2 - (
                                          q13 - q23) ** 2) * (q13 -
                                                              q23) * (
                                          q13 - q33)) / (
                                         2 * (2 * ly + q13 - q23) * (
                                         2 * ly - q13 +
                                         q23) ** 2)) + (-8 * ly ** 3 +
                                                        4 * ly ** 2 * np.sqrt(
                                   4 * ly ** 2 - (q11 - q21) ** 2 - (
                                           q13 - q23) ** 2) +
                                                        2 * ly * (
                                                                q13 - q23) ** 2 +
                                                        2 * (
                                                                -q11 + q21) * np.sqrt(
                                   lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) - ly ** 2 * (q23 - q33) ** 2) -
                                                        np.sqrt(
                                                            4 * ly ** 2 - (
                                                                    q11 - q21) ** 2 - (
                                                                    q13 - q23) ** 2) * (
                                                                q13 -
                                                                q23) * (
                                                                q13 - q33)) / (
                                       2 * (2 * ly + q13 - q23) ** 2 * (
                                       2 * ly - q13 +
                                       q23)) + (-4 * ly * (q13 - q23) + (
                               4 * ly ** 2 * (q13 - q23)) / np.sqrt(
                           4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) + ((-q11 +
                                                                                  q21) * (2 * lx ** 2 * (
                               q13 - q23) - 2 * ly ** 2 * (q23 - q33))) / np.sqrt(
                           lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) - ly ** 2 * (q23 - q33) ** 2) +
                                                np.sqrt(4 * ly ** 2 - (
                                                        q11 - q21) ** 2 - (
                                                                q13 - q23) ** 2) * (
                                                        q13 -
                                                        q33) - ((
                                                                        q13 - q23) ** 2 * (
                                                                        q13 - q33)) / np.sqrt(
                                   4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2)) / (2 * (2 * ly + q13 -
                                                                                               q23) * (
                                                                                                  2 * ly - q13 + q23))) + (
                               1 / ((-4 * ly ** 2 + (q13 - q23) ** 2) ** 2)) *
                       2 * (-2 * lx * (q13 - q23) + (-2 * lx - 2 * q11) * (q13 - q23) + (
                        np.sqrt(4 * ly ** 2 - (q11 - q21) ** 2 - (q13 -
                                                                  q23) ** 2) * (
                                2 * lx ** 2 * (q13 - q23) - 2 * ly ** 2 * (q23 - q33))) / np.sqrt(
                    lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) - ly ** 2 * (q23 - q33) ** 2) + (
                                    2 * (q13 - q23) * np.sqrt(
                                lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) - ly ** 2 * (q23 - q33) ** 2)) / np.sqrt(
                    4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) - q21 * (q13 - q33) -
                            q11 * (q13 - 2 * q23 + q33)) * (-4 * ly ** 2 * (2 * lx + q11 + q21) +
                                                            2 * np.sqrt(
                            4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) * np.sqrt(
                            lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) -
                            ly ** 2 * (q23 - q33) ** 2) + (q13 - q23) * (2 * lx * (q13 - q23) +
                                                                         q21 * (q13 - q33) + q11 * (
                                                                                 q13 - 2 * q23 + q33))) + (
                               1 / ((-4 * ly ** 2 + (q13 - q23) ** 2) ** 3)) *
                       4 * (q13 - q23) * (-4 * ly ** 2 * (2 * lx + q11 + q21) +
                                          2 * np.sqrt(4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) * np.sqrt(
                            lx ** 2 * (4 * ly ** 2 - (q13 - q23) ** 2) -
                            ly ** 2 * (q23 - q33) ** 2) + (q13 - q23) * (2 * lx * (q13 - q23) +
                                                                         q21 * (q13 - q33) + q11 * (
                                                                                 q13 - 2 * q23 + q33))) ** 2)

        s33 = 1 / 2 * (q33 + (2 * ly * np.sqrt(4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) * (-q13 + q23)
                              * np.sqrt(lx ** 2 * (2 * ly + q13 - q23) * (2 * ly - q13 + q23) - ly ** 2 *
                                        (q23 - q33) ** 2) - (q13 - q23) * (2 * (lx + q11) * (q11 - q21) -
                                                                           2 * q12 * np.sqrt(4 * ly ** 2 -
                                                                                             (q11 - q21) ** 2 - (
                                                                                                     q13 - q23) ** 2) + (
                                                                                   q13 - q23) * (
                                                                                   2 * q13 - q33)) * np.sqrt(
                    lx ** 2 * (2 * ly + q13 - q23) * (2 * ly - q13 + q23) -
                    ly ** 2 * (q23 - q33) ** 2) + 4 * ly ** 3 * (q11 - q21) * (q23 - q33) -
                              4 * ly ** 2 * (-q12 * q21 * q23 + lx * np.sqrt(
                    4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) * q23 -
                                             2 * q13 * np.sqrt(lx ** 2 * (2 * ly + q13 - q23) * (2 * ly - q13 + q23) -
                                                               ly ** 2 * (q23 - q33) ** 2) + q11 * (q12 + np.sqrt(
                            4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2)) * (q23 - q33) +
                                             q12 * q21 * q33 -
                                             lx * np.sqrt(4 * ly ** 2 - (q11 - q21) ** 2 - (q13 - q23) ** 2) * q33 +
                                             np.sqrt(lx ** 2 * (2 * ly + q13 - q23) * (2 * ly - q13 + q23) -
                                                     ly ** 2 * (q23 - q33) ** 2) * q33)) / (
                               (2 * ly + q13 - q23) * (2 * ly - q13 + q23) * np.sqrt(
                           lx ** 2 * (2 * ly + q13 - q23) * (2 * ly - q13 + q23) -
                           ly ** 2 * (q23 - q33) ** 2)))

        return np.array([s11, s12, s13, s21, s23, s33])

    def plan(self):
        """
        顶层设计：
            使用强化学习设计一个路径点
        :return:jrs
        """
        pass

    def control(self):
        """
        整体的控制流程应该是，n
        1. 首先根据当前的位置、速度、加速度做内力消除。
            内力消除：两种思路，保证末端位姿固定的条件下，调整受力或者调整位置。调整位置更合适。也就是动力学模型修正
            这部分需要考虑么
            调整力仿佛也是可行的。因为我是可以根据力模型进行计算的
        2. 根据给出的下一个way points做主关节位置控制，副关节力控制
            稳定后进行 控制模型切换:
                暂停仿真，读取主关节输出力矩，读取副关节位置
                根据FD模型，设置主关节目标力矩，设置副关节目标位置
                恢复仿真
            稳定后继续切换
        3. 根据下一个way point，循环12
        """
        # 判断是否已经稳定
        # 这个部分需要研究一下PID自整定的问题。
        # 给定目标函数，也就是末端的位姿，然后可以生成一系列guiding data,生成初步的扰动

        # 切换控制模式
        sim.simxPauseSimulation(self.clientID, True)
        for idx in self.major_joint_idx:
            handle = self.joint_handles[idx]
            self.joint_control(handle, 'force', self.force[idx[0] * 3 + idx[1]])
        for idx in self.redundant_joint_idx:
            handle = self.joint_handles[idx]
            self.joint_control(handle, 'position', self.q[idx])
        sim.simxPauseSimulation(self.clientID, False)

        # 判断是否稳定

        # 切换控制模式

    def pid_auto_tuning(self):
        # 使用DDPG进行自稳调节
        pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import time

    print('Program started')
    sim.simxFinish(-1)  # just in case, close all opened connections
    while True:
        clientID = sim.simxStart('127.0.0.1', 25001, True, True, 5000, 5)  # Connect to CoppeliaSim
        if clientID >= -1:
            break
        else:
            time.sleep(1)
            print('Please run the simulation on CoppeliaSim!')

    print('Connected to remote API server')

    # set simulation step time
    tstep = 0.005  # for better control accuracy
    sim.simxSetFloatingParameter(clientID, sim.sim_floatparam_simulation_time_step, tstep, sim.simx_opmode_blocking)

    PM = PM_4PPPS(clientID, name='PM_4PPPS')
    PM.IK(np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
          np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), )
