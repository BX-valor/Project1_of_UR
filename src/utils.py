"""
utils.py - 工具函数

包含数值线性化、仿真评估指标等通用工具函数。
"""

import numpy as np


def linearize_system(plant, x_eq, u_eq=0, eps=1e-6):
    """
    在平衡点处数值线性化
    
    计算 A = df/dx 和 B = df/du 在平衡点处的值
    
    参数:
        plant: Acrobot 实例
        x_eq: 平衡点状态
        u_eq: 平衡点控制输入
        eps: 数值微分步长
    
    返回:
        A: 状态矩阵 (nxn)
        B: 输入矩阵 (nx1)
    """
    n = len(x_eq)
    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    
    # 计算 f(x_eq, u_eq)
    f_eq = plant.dynamics(x_eq, u_eq)
    
    # A 矩阵: df/dx
    for i in range(n):
        x_perturb = x_eq.copy()
        x_perturb[i] += eps
        f_perturb = plant.dynamics(x_perturb, u_eq)
        A[:, i] = (f_perturb - f_eq) / eps
    
    # B 矩阵: df/du
    B[:, 0] = (plant.dynamics(x_eq, u_eq + eps) - f_eq) / eps
    
    return A, B


def compute_settling_time(t, x, x_eq, threshold=0.05, use_relative=True):
    """
    计算调节时间（进入阈值范围内且不再离开）
    
    参数:
        t: 时间数组
        x: 状态轨迹 (n_states, n_time)
        x_eq: 目标状态
        threshold: 误差阈值
        use_relative: True 则 threshold 为相对于初始误差的比例，False 则为绝对阈值
    
    返回:
        settling_time: 调节时间，若未稳定则返回 None
    """
    errors = np.linalg.norm(x - x_eq[:, np.newaxis], axis=0)
    
    if use_relative:
        initial_error = errors[0]
        threshold_value = threshold * initial_error
    else:
        threshold_value = threshold
    
    within_threshold = errors < threshold_value
    if not np.any(within_threshold):
        return None
    
    outside_indices = np.where(~within_threshold)[0]
    if len(outside_indices) == 0:
        return t[0]
    
    last_outside = outside_indices[-1]
    
    # 鲁棒性改进：允许末端有短暂波动。
    # 如果最后 5% 时间内 >=80% 数据点在阈值内，则认为系统已稳定。
    n_tail = max(1, int(0.05 * len(t)))
    if last_outside >= len(t) - n_tail:
        # 检查最后 5% 时间内的稳定比例
        tail_within_ratio = np.sum(within_threshold[-n_tail:]) / n_tail
        if tail_within_ratio >= 0.8:
            # 找到最后一个在阈值外的点（排除末端噪声）
            # 从后往前找，跳过末端在阈值内的连续段
            effective_last_outside = last_outside
            for i in range(last_outside, -1, -1):
                if within_threshold[i]:
                    effective_last_outside = i
                else:
                    break
            # 实际上我们需要找到真正的最后一个 outside
            # 重新搜索，忽略最后 5% 的噪声
            effective_last_outside = -1
            for i in range(len(t) - n_tail):
                if not within_threshold[i]:
                    effective_last_outside = i
            if effective_last_outside >= 0 and effective_last_outside < len(t) - 1:
                return t[effective_last_outside + 1]
            return None
        else:
            return None
    
    return t[last_outside + 1]


def compute_overshoot(t, x, x_eq, idx=0):
    """
    计算超调量（针对指定状态分量）
    
    参数:
        t: 时间数组
        x: 状态轨迹
        x_eq: 目标状态
        idx: 状态分量索引
    
    返回:
        overshoot: 超调百分比
    """
    x_i = x[idx, :]
    x_target = x_eq[idx]
    initial = x_i[0]
    
    if abs(initial - x_target) < 1e-10:
        return 0.0
    
    # 看是否越过目标值
    if initial < x_target:
        max_val = np.max(x_i)
        if max_val > x_target:
            return (max_val - x_target) / abs(x_target - initial) * 100
    else:
        min_val = np.min(x_i)
        if min_val < x_target:
            return (x_target - min_val) / abs(x_target - initial) * 100
    
    return 0.0


def evaluate_performance(t, x, u, x_eq):
    """
    综合评估控制性能
    
    返回:
        dict: 包含调节时间、超调、最大力矩、稳态误差、能量消耗等指标
    """
    u = np.array(u)
    
    # 调节时间（基于总状态误差）
    settling_t = compute_settling_time(t, x, x_eq, threshold=0.05)
    
    # 超调（theta1）
    overshoot = compute_overshoot(t, x, x_eq, idx=0)
    
    # 最大力矩
    max_torque = np.max(np.abs(u)) if len(u) > 0 else 0
    
    # 平均力矩
    avg_torque = np.mean(np.abs(u)) if len(u) > 0 else 0
    
    # 稳态误差（最后 10% 时间的平均误差）
    n_steady = max(1, int(0.1 * x.shape[1]))
    steady_error = np.mean(np.linalg.norm(x[:, -n_steady:] - x_eq[:, np.newaxis], axis=0))
    
    # 总能量消耗（力矩平方积分）
    if len(u) > 1:
        energy_consumption = np.trapezoid(u**2, t[:len(u)])
    else:
        energy_consumption = 0
    
    # 切换次数（如果有模式信息则在外部计算）
    
    return {
        'settling_time': settling_t,
        'overshoot': overshoot,
        'max_torque': max_torque,
        'avg_torque': avg_torque,
        'steady_error': steady_error,
        'energy_consumption': energy_consumption
    }


def wrap_angle(angle):
    """
    将角度 wrapping 到 [-pi, pi] 范围
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi
