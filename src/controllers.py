"""
controllers.py - 控制器实现

包含:
- EnergyShapingController: Spong 能量成型控制器（Swing-up 阶段）
- LQRController: LQR 控制器（Balance 阶段）
- HybridController: 混合控制器（切换逻辑）
"""

import numpy as np
from scipy.linalg import solve_continuous_are


class EnergyShapingController:
    """
    Spong 能量成型控制器（Swing-up 阶段）
    
    基于论文: Spong, M.W. "Swing up control of the acrobot." IEEE ICRA, 1994.
    
    控制律结合能量泵入和 PD 稳定项，通过 Collocated PFL 转换为实际力矩。
    """
    
    def __init__(self, plant, k_e=50.0, k_p=5.0, k_d=1.0, u_max=10.0):
        """
        参数:
            plant: Acrobot 实例
            k_e: 能量增益
            k_p: PD 比例增益
            k_d: PD 微分增益
            u_max: 最大控制力矩 [Nm]
        """
        self.plant = plant
        self.k_e = k_e
        self.k_p = k_p
        self.k_d = k_d
        self.u_max = u_max
        self.E_des = plant.desired_energy()
        
    def compute_control(self, x):
        """
        计算控制输入 u
        
        参数:
            x: [theta1, theta2, theta1_dot, theta2_dot]
        
        返回:
            u: 控制力矩（标量）
        """
        q1, q2, q1_dot, q2_dot = x
        
        # 当前能量
        E = self.plant.energy(x)
        E_tilde = E - self.E_des
        
        # 能量泵入项: -k_e * theta2_dot * (E - E_des)
        # 基于 Spong 1994 原始控制律，通过 PFL 转换保持稳定性
        energy_term = -self.k_e * q2_dot * E_tilde
        
        # PD 稳定项（保持第二关节在目标位置附近）
        pd_term = -self.k_p * q2 - self.k_d * q2_dot
        
        # 虚拟控制输入 v = theta2_ddot_desired
        v = energy_term + pd_term
        
        # 通过 Collocated PFL 转换为实际力矩
        M, M_inv, C, G, B = self.plant.get_manipulator_matrices(x)
        
        q_dot = np.array([q1_dot, q2_dot])
        
        # 从 M*q_ddot + C*q_dot + G = B*u 推导:
        # q_ddot = M_inv @ (B*u - C*q_dot - G)
        # theta2_ddot = M_inv[1,0]*(B[0]*u - ...) + M_inv[1,1]*(B[1]*u - ...)
        # 对于 Acrobot, B = [0, 1]^T:
        # theta2_ddot = M_inv[1,1] * u - M_inv[1,:] @ (C*q_dot + G)
        # 令 theta2_ddot = v, 解出 u:
        # u = (v + M_inv[1,:] @ (C*q_dot + G)) / M_inv[1,1]
        
        u = (v + M_inv[1, :] @ (C @ q_dot + G)) / M_inv[1, 1]
        
        # 控制力矩限幅
        u = np.clip(u, -self.u_max, self.u_max)
        
        return float(u)


class LQRController:
    """
    LQR 控制器（Balance 阶段）
    
    在平衡点处线性化，求解代数 Riccati 方程得到最优反馈增益。
    """
    
    def __init__(self, A, B, Q, R, x_eq, u_max=10.0):
        """
        参数:
            A, B: 线性化系统矩阵
            Q: 状态代价矩阵 (4x4)
            R: 控制代价矩阵 (标量或 1x1)
            x_eq: 平衡点状态
            u_max: 最大控制力矩 [Nm]
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R if np.isscalar(R) else np.atleast_2d(R)
        self.x_eq = x_eq
        self.u_max = u_max
        
        # 求解代数 Riccati 方程
        self.P = solve_continuous_are(A, B, Q, self.R)
        
        # 计算 LQR 增益 K
        self.K = np.linalg.inv(self.R) @ self.B.T @ self.P
        
        # 验证闭环稳定性
        A_cl = A - B @ self.K
        eigenvalues = np.linalg.eigvals(A_cl)
        print(f"LQR 闭环特征值: {eigenvalues}")
        assert all(np.real(eigenvalues) < 0), "LQR 未稳定化系统！"
        
        print(f"LQR 增益 K: {self.K.flatten()}")
    
    def compute_control(self, x):
        """
        计算控制输入
        
        参数:
            x: 当前状态
        
        返回:
            u: 控制力矩（标量）
        """
        x_tilde = x - self.x_eq
        u = -self.K @ x_tilde
        
        # 限幅
        return float(np.clip(u.item(), -self.u_max, self.u_max))
    
    def cost_to_go(self, x):
        """
        计算 Cost-to-Go: J(x) = x_tilde^T * P * x_tilde
        """
        x_tilde = x - self.x_eq
        return float(x_tilde @ self.P @ x_tilde)


class HybridController:
    """
    混合控制器：Energy Shaping + LQR 切换
    
    基于 LQR Cost-to-Go 设计切换准则，带状态约束和滞后机制防止频繁切换。
    """
    
    def __init__(self, swing_up_controller, lqr_controller,
                 switching_threshold=50.0, hysteresis_factor=2.0,
                 state_constraints=None):
        """
        参数:
            swing_up_controller: EnergyShapingController 实例
            lqr_controller: LQRController 实例
            switching_threshold: Cost-to-Go 切换阈值
            hysteresis_factor: 滞后因子
            state_constraints: 状态约束字典，可选键: 'max_dq1', 'max_dq2', 'max_q2'
                              用于确保切换时状态足够接近平衡点
        """
        self.swing_up = swing_up_controller
        self.lqr = lqr_controller
        self.threshold = switching_threshold
        self.hysteresis = switching_threshold * hysteresis_factor
        self.active_controller = "SWING_UP"
        self.switch_history = []
        
        # 默认状态约束
        if state_constraints is None:
            state_constraints = {}
        self.max_dq1 = state_constraints.get('max_dq1', 3.0)
        self.max_dq2 = state_constraints.get('max_dq2', 5.0)
        self.max_q2 = state_constraints.get('max_q2', np.pi)
        
    def _can_switch_to_lqr(self, x, cost_to_go):
        """
        判断是否可以切换到 LQR
        
        条件:
        1. Cost-to-Go 低于阈值
        2. 状态速度和角度在允许范围内
        """
        if cost_to_go >= self.threshold:
            return False
        
        q1, q2, q1_dot, q2_dot = x
        
        # 速度约束
        if abs(q1_dot) > self.max_dq1:
            return False
        if abs(q2_dot) > self.max_dq2:
            return False
        
        # 第二关节角度约束（避免转太多圈）
        if abs(q2) > self.max_q2:
            return False
        
        return True
        
    def compute_control(self, x, t=None):
        """
        计算控制输入并处理切换逻辑
        """
        # 计算 LQR cost-to-go
        cost_to_go = self.lqr.cost_to_go(x)
        
        # 切换逻辑（带滞后和状态约束）
        if self.active_controller == "SWING_UP":
            if self._can_switch_to_lqr(x, cost_to_go):
                self.active_controller = "LQR"
                if t is not None:
                    self.switch_history.append((t, "SWING_UP -> LQR", cost_to_go))
                print(f"[t={t:.3f}s] Switched to LQR (cost={cost_to_go:.4f}, q2={x[1]:.3f}, dq1={x[2]:.3f}, dq2={x[3]:.3f})")
        else:  # LQR 模式
            if cost_to_go > self.hysteresis:
                self.active_controller = "SWING_UP"
                if t is not None:
                    self.switch_history.append((t, "LQR -> SWING_UP", cost_to_go))
                print(f"[t={t:.3f}s] Switched to Swing-up (cost={cost_to_go:.4f})")
        
        # 计算控制输入
        if self.active_controller == "SWING_UP":
            u = self.swing_up.compute_control(x)
        else:
            u = self.lqr.compute_control(x)
            
        return u, self.active_controller
    
    def get_switch_history(self):
        """
        获取切换历史记录
        """
        return self.switch_history


class PassiveController:
    """
    被动控制器（u=0），用于对比测试
    """
    
    def compute_control(self, x):
        return 0.0
