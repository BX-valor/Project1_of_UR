"""
acrobot.py - Acrobot 动力学模型

Acrobot（双摆系统）动力学实现，基于拉格朗日方程。
状态: x = [theta1, theta2, theta1_dot, theta2_dot]
控制: u (施加在肘关节的力矩)
"""

import numpy as np
from scipy.integrate import solve_ivp


class Acrobot:
    """
    Acrobot 动力学系统
    
    二连杆、二自由度平面机器人，肩关节被动，肘关节主动驱动。
    """
    
    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0,
                 lc1=0.5, lc2=0.5, I1=None, I2=None, g=9.81,
                 underactuated=True):
        """
        初始化 Acrobot 参数
        
        参数:
            m1, m2: 连杆质量 [kg]
            l1, l2: 连杆长度 [m]
            lc1, lc2: 质心到关节距离 [m]
            I1, I2: 转动惯量 [kg*m^2]，None 则使用细杆模型
            g: 重力加速度 [m/s^2]
            underactuated: 是否为欠驱动模式
        """
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.lc1 = lc1 if lc1 is not None else l1 / 2
        self.lc2 = lc2 if lc2 is not None else l2 / 2
        # 等效转动惯量（绕关节）= 绕质心转动惯量 + 质心平移项 (m*lc^2)
        self.I1 = I1 if I1 is not None else m1 * lc1**2 + m1 * l1**2 / 12
        self.I2 = I2 if I2 is not None else m2 * lc2**2 + m2 * l2**2 / 12
        self.g = g
        self.underactuated = underactuated
        
    def get_manipulator_matrices(self, x):
        """
        计算操作手方程中的各矩阵
        M(q)*q_ddot + C(q,q_dot)*q_dot + G(q) = B*u
        
        参数:
            x: 状态向量 [theta1, theta2, theta1_dot, theta2_dot]
        
        返回:
            M: 质量矩阵 (2x2)
            M_inv: M 的逆矩阵 (2x2)
            C: 科氏力/离心力矩阵 (2x2)
            G: 重力项向量 (2,)
            B: 输入矩阵 (2x1)
        """
        q1, q2, q1_dot, q2_dot = x
        
        # 质量矩阵 M(q)
        M11 = (self.I1 + self.I2 + self.m2 * self.l1**2 +
               2 * self.m2 * self.l1 * self.lc2 * np.cos(q2))
        M12 = self.I2 + self.m2 * self.l1 * self.lc2 * np.cos(q2)
        M22 = self.I2
        
        M = np.array([[M11, M12],
                      [M12, M22]])
        M_inv = np.linalg.inv(M)
        
        # 科氏力矩阵 C(q, q_dot)
        # 标准形式满足 M_dot - 2C 为反对称矩阵
        h = -self.m2 * self.l1 * self.lc2 * np.sin(q2)
        C = np.array([[h * q2_dot, h * (q1_dot + q2_dot)],
                      [-h * q1_dot, 0]])
        
        # 重力项 G(q)
        G1 = (-(self.m1 * self.lc1 + self.m2 * self.l1) * self.g * np.sin(q1)
              - self.m2 * self.lc2 * self.g * np.sin(q1 + q2))
        G2 = -self.m2 * self.lc2 * self.g * np.sin(q1 + q2)
        G = np.array([G1, G2])
        
        # 输入矩阵 B
        if self.underactuated:
            B = np.array([[0], [1]])
        else:
            B = np.array([[1, 0], [0, 1]])
            
        return M, M_inv, C, G, B
    
    def dynamics(self, x, u):
        """
        计算状态导数 x_dot = f(x) + g(x)*u
        
        参数:
            x: 状态向量 [theta1, theta2, theta1_dot, theta2_dot]
            u: 控制力矩（标量）
        
        返回:
            x_dot: 状态导数 [theta1_dot, theta2_dot, theta1_ddot, theta2_ddot]
        """
        q1, q2, q1_dot, q2_dot = x
        q_dot = np.array([q1_dot, q2_dot])
        
        M, M_inv, C, G, B = self.get_manipulator_matrices(x)
        
        # q_ddot = M^{-1} * (B*u - C*q_dot - G)
        if self.underactuated:
            q_ddot = M_inv @ (B.flatten() * u - C @ q_dot - G)
        else:
            q_ddot = M_inv @ (B @ np.atleast_1d(u) - C @ q_dot - G)
        
        return np.array([q1_dot, q2_dot, q_ddot[0], q_ddot[1]])
    
    def energy(self, x):
        """
        计算系统总能量 E = KE + PE
        
        参数:
            x: 状态向量
        
        返回:
            E: 系统总能量 [J]
        """
        q1, q2, q1_dot, q2_dot = x
        q_dot = np.array([q1_dot, q2_dot])
        
        M, _, _, _, _ = self.get_manipulator_matrices(x)
        
        # 动能: 0.5 * q_dot^T * M * q_dot
        KE = 0.5 * q_dot.T @ M @ q_dot
        
        # 势能（以悬挂点为参考，竖直向上为势能零点）
        PE = (-(self.m1 * self.lc1 + self.m2 * self.l1) * self.g * np.cos(q1)
              - self.m2 * self.lc2 * self.g * np.cos(q1 + q2))
        
        return KE + PE
    
    def desired_energy(self):
        """
        目标平衡点的能量（两连杆竖直向上）
        
        返回:
            E_des: 目标能量 [J]
        """
        return ((self.m1 * self.lc1 + self.m2 * self.l1) * self.g +
                self.m2 * self.lc2 * self.g)
    
    def kinetic_energy(self, x):
        """
        计算系统动能
        """
        q1, q2, q1_dot, q2_dot = x
        q_dot = np.array([q1_dot, q2_dot])
        M, _, _, _, _ = self.get_manipulator_matrices(x)
        return 0.5 * q_dot.T @ M @ q_dot
    
    def potential_energy(self, x):
        """
        计算系统势能
        """
        q1, q2 = x[0], x[1]
        return (-(self.m1 * self.lc1 + self.m2 * self.l1) * self.g * np.cos(q1)
                - self.m2 * self.lc2 * self.g * np.cos(q1 + q2))
    
    def get_cartesian_positions(self, x):
        """
        获取连杆端点的笛卡尔坐标
        
        参数:
            x: 状态向量
        
        返回:
            (x0, y0): 基座坐标
            (x1, y1): 第一连杆末端/第二连杆起点
            (x2, y2): 第二连杆末端
        """
        q1, q2 = x[0], x[1]
        
        x0, y0 = 0.0, 0.0
        x1 = self.l1 * np.sin(q1)
        y1 = -self.l1 * np.cos(q1)
        x2 = x1 + self.l2 * np.sin(q1 + q2)
        y2 = y1 - self.l2 * np.cos(q1 + q2)
        
        return (x0, y0), (x1, y1), (x2, y2)


if __name__ == "__main__":
    # 简单测试：验证能量守恒（无控制输入时）
    import matplotlib.pyplot as plt
    
    plant = Acrobot()
    x0 = np.array([0.5, 0.3, 0.0, 0.0])  # 初始状态
    
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 1000)
    
    sol = solve_ivp(lambda t, x: plant.dynamics(x, 0), t_span, x0, t_eval=t_eval, method='RK45')
    
    energies = [plant.energy(sol.y[:, i]) for i in range(sol.y.shape[1])]
    
    plt.figure(figsize=(10, 4))
    plt.plot(sol.t, energies)
    plt.xlabel('Time [s]')
    plt.ylabel('Energy [J]')
    plt.title('Energy Conservation Test (u=0)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../figures/energy_conservation_test.png', dpi=150)
    plt.show()
    
    print(f"能量漂移: {np.max(energies) - np.min(energies):.6f} J")
    print("acrobot.py 测试完成！")
