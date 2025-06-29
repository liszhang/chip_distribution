import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

def calculate_chip_metrics(chip_data, close_price):
    """
    :param chip_data: 筹码分布数据 [[price1, percent1], [price2, percent2], ...]
    :param close_price: 当前收盘价
    :return: 包含所有指标的字典
    """
    # 转换为numpy数组便于计算
    chips = np.array(chip_data)
    prices = chips[:, 0]
    percents = chips[:, 1]
    
    # 1. 平均成本（加权平均）
    avg_cost = np.sum(prices * percents) / np.sum(percents)
    
    # 2. 收盘获利比例（价格<=收盘价的筹码占比）
    profit_ratio = np.sum(percents[prices <= close_price]) / np.sum(percents)
    
    # 3. 成本区间计算（90%和70%）
    def get_cost_interval(target_ratio):
        sorted_idx = np.argsort(prices)
        sorted_prices = prices[sorted_idx]
        sorted_percents = percents[sorted_idx]
        cum_percent = np.cumsum(sorted_percents) / np.sum(percents)
        
        # 找到区间下限（5%位置）
        low_idx = np.where(cum_percent >= (1-target_ratio)/2)[0][0]
        # 找到区间上限（95%位置）
        high_idx = np.where(cum_percent >= 1-(1-target_ratio)/2)[0][0]
        
        return [sorted_prices[low_idx], sorted_prices[high_idx]]
    
    interval_90 = get_cost_interval(0.9)
    interval_70 = get_cost_interval(0.7)
    
    # 4. 筹码集中度（价格在70%区间内的筹码占比）
    concentration_70 = np.sum(
        percents[(prices >= interval_70[0]) & (prices <= interval_70[1])]
    ) / np.sum(percents)
    
    return {
        'avg_cost': avg_cost,
        'profit_ratio': profit_ratio,
        'interval_90': interval_90,
        'interval_70': interval_70,
        'concentration_70': concentration_70
    }

