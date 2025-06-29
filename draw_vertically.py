import matplotlib.pyplot as plt
import numpy as np

def draw_chip_distribution(code, data,type = 'non_recurve'):
    """
    绘制筹码分布图（竖过来的）
    """
    # 分离价格和概率
    prices = np.array([point[0] for point in data])
    probabilities = np.array([point[1] for point in data])
    
    total_prob = np.sum(probabilities)
    
    # 设置阈值 - 过滤掉小于最大概率0.1%的值
    threshold = 0.00025
    mask = probabilities > threshold

    prices = prices[mask]
    probabilities = probabilities[mask]



    #print(probabilities)
    
    # 计算价格间隔（用于设置条形宽度）
    price_diff = np.diff(prices)
    if len(price_diff) > 0:
        bar_height = np.min(price_diff) * 0.8  # 使用最小间隔的80%作为条宽
    else:
        bar_height = 0.01  # 默认值
    
    # 创建画布
    plt.figure(figsize=(8, 10))
    
    # 使用条形图绘制筹码分布
    plt.barh(prices, probabilities, 
             height=bar_height, 
             color='#4CAF50', 
             edgecolor='#2E7D32',
             linewidth=0.5,
             alpha=0.8)
    
    # 设置坐标轴标签
    plt.xlabel('probability', fontsize=12)
    plt.ylabel('price', fontsize=12)
    plt.title(f'{code} chip distribution', fontsize=14)
    
    # 设置网格线
    plt.grid(True, linestyle='--', alpha=0.3, axis='x')
    
    # 设置坐标轴范围
    plt.xlim(0, max(probabilities)*1.1)
    plt.ylim(min(prices) - bar_height, max(prices) + bar_height)
    
    # 优化布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(type + f'_{code}_筹码分布.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，避免内存泄漏