"""
计算筹码分布（使用自适应截断高斯方法）
2025.05.24 修正版
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar

def solve_truncated_gaussian(L, H, M):
    """
    同时求解截断高斯分布的均值和标准差
    使得截断后的均值等于观测值M
    """
    # 初始猜测：标准差为价格区间的20%
    price_range = H - L
    sigma0 = max(0.2 * price_range, 1e-5)
    mu0 = M  # 初始均值设为观测均值
    
    # 定义方程：截断后的均值应等于M
    def equations(params):
        mu, sigma = params
        a = (L - mu) / sigma
        b = (H - mu) / sigma
        Z = norm.cdf(b) - norm.cdf(a)
        if Z < 1e-10:
            return [0, 0]  # 避免除零错误
        
        # 截断后的均值公式
        truncated_mean = mu + sigma * (norm.pdf(a) - norm.pdf(b)) / Z
        # 返回与观测值M的差异
        return [truncated_mean - M, sigma - sigma0]  # 保持标准差相对稳定
    
    # 使用数值优化求解
    try:
        sol = root_scalar(
            lambda x: equations([x, sigma0])[0],
            bracket=[L, H],
            method='brentq'
        )
        mu = sol.root
    except:
        mu = M  # 求解失败时使用观测均值
    
    return mu, sigma0

def calc_current_chip(min_price, max_price, vwap, price_step=0.01):
    """
    计算单日筹码分布
    """

    L = min_price
    H = max_price

    # 处理无效价格区间
    if L >= H:
        return [[(L+H)/2, 1]]
    
    # 计算成交均价
    M = vwap
    
    # 获取自适应参数
    mu, sigma = solve_truncated_gaussian(L, H, M)
    
    # 生成价格网格
    prices = np.arange(L, H + price_step, price_step)
    
    # 计算截断高斯分布
    a = (L - mu) / sigma
    b = (H - mu) / sigma
    denom = norm.cdf(b) - norm.cdf(a)
    
    # 避免除零错误
    if denom < 1e-10:
        pdf_values = np.ones_like(prices) / len(prices)
    else:
        pdf_values = norm.pdf(prices, mu, sigma) / denom
    
    # 归一化并分配筹码
    total = np.sum(pdf_values)
    if total < 1e-10:
        chip_distribution = 1.0 * (prices == prices[len(prices)//2])
    else:
        chip_distribution = pdf_values / total * 1.0
    

    data = list(zip(prices, chip_distribution))
    # 返回价格-筹码分布对
    #avg = 0
    #for item in data:
    #    avg += item[0] * item[1]
    
    #print(avg)

    #print('当日筹码分布:')
    #print(data)

    return data


def calc_current_chip_original(min_price, max_price, vwap, close_price=None, price_step=0.01):

    #vwap 是成交均价
    if abs(min_price - max_price) < 0.000001: # 开盘涨停或者跌停
        chip_list = [[min_price,1]]
        return chip_list
    
    #vwap = amount / volume 
    #print("vwap",vwap)

    prices = np.arange(min_price, max_price + price_step, price_step)
    # 建议用收盘价和VWAP加权，软件常见权重 0.7/0.3
    if close_price is not None:
        peak_pos = 0.6 * vwap + 0.4 * close_price
    else:
        peak_pos = vwap
    # 限制顶点在范围内
    peak_pos = max(min_price, min(max_price, peak_pos))
    chip_dist = np.zeros_like(prices)
    left_idx = np.argmin(np.abs(prices - min_price))
    peak_idx = np.argmin(np.abs(prices - peak_pos))
    right_idx = np.argmin(np.abs(prices - max_price))
    # 防止极端情况
    if peak_idx == left_idx or peak_idx == right_idx:
        peak_idx = (left_idx + right_idx) // 2
    # 左侧
    for i in range(left_idx, peak_idx+1):
        chip_dist[i] = (prices[i] - min_price) / (peak_pos - min_price + 1e-8)
    # 右侧
    for i in range(peak_idx, right_idx+1):
        chip_dist[i] = (max_price - prices[i]) / (max_price - peak_pos + 1e-8)
    # 归一化，使总面积等于成交量
    chip_dist = chip_dist * price_step
    chip_dist = (chip_dist / np.sum(chip_dist)) * 1.0
    # 计算平均成本
    avg_cost = np.sum(prices * chip_dist) / np.sum(chip_dist)
    #print(avg_cost)

    data = list(zip(prices, chip_dist))
    #print('当日筹码分布:')
    #print(data)
    return data