"""
非递归方式进行计算筹码
这个计算是基于赋权的股票数据
2025.06.15
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict
from current_chip_distribution import calc_current_chip,calc_current_chip_original
from draw_vertically import draw_chip_distribution
from chip_stat import calculate_chip_metrics
#from multi_process import parallel_aggregate_chips

stock_path = '/home/ubuntu/Data_Center/stock_daily_simple/stock_data_new'

def compute_survival_ratio(df):
    # 功能：计算股票筹码的存活率
    # 逻辑：从数据尾部向前倒推，基于换手率计算持有者留存比例
    # 公式：当前筹码存活率 = 当日的换手率 × 当日的存活率
    # Tips：交易首日的换手率是 1.0，表示初始状态
    
    #        date   换手率  survival_ratio
    #0 2023-01-01  1.00         0.45220
    #1 2023-01-02  0.20         0.56525
    #2 2023-01-03  0.15         0.66500
    #3 2023-01-04  0.30         0.95000
    #4 2023-01-05  0.05         1.00000

    # sum(换手率 * survival_ratio) = 1.0 # 这是一个很重要的验证点

    survival = [1.0]  # 初始化最后一行=1
    for t in df['换手率'].values[-1:0:-1]:  # 只遍历最后一行到第二行
        survival.append(survival[-1] * (1 - t))
    df['survival_ratio'] = list(reversed(survival))


def calc_chip_everyday(df):
    # 计算每一天的筹码分布

    price_step = 0.01
    df.insert(len(df.columns), '筹码比例', '')
    df['筹码比例'] = df['筹码比例'].astype(object)

    for index, row in tqdm(df.iterrows()):
        chip_list_current = calc_current_chip_original( 
                row['最低价'], row['最高价'], row['交易均价'], row['收盘价'],price_step)
        
        df.at[index, '筹码比例'] = chip_list_current

def calc_chip_everyday_fast(df):
    # 计算每一天的筹码分布
    price_step = 0.01
    chip_ratio_list = []
    for row in tqdm(df.itertuples(), total=len(df)):
        chip_list_current = calc_current_chip_original(
            row.最低价, row.最高价, row.交易均价, row.收盘价, price_step
        )
        chip_ratio_list.append(chip_list_current)
    # 直接用新 list 赋值，不要用 df.at/index
    df['筹码比例'] = chip_ratio_list

def aggregate_chips(df):
    #从历史每一天的筹码分布去积累最终的筹码分布
    total = defaultdict(float)
    for _, row in df.iterrows():
        for price_weight_tuple in row['筹码比例']:
            # 关键步骤：价格×100转为整数，避免浮点误差
            price_float = price_weight_tuple[0]
            weight = price_weight_tuple[1]
            price_int = int(round(price_float * 100))
            
            # 累加计算（在整数域操作）
            # 需要注意: 当前筹码存活率 = 当日的换手率 × 当日的存活率
            total[price_int] += weight * row['survival_ratio']*row['换手率']
    
    # 转换回原始价格并归一化
    sum_weights = sum(total.values())
    #print(f"Total weights sum: {sum_weights}")
    chip_dict =  {p/100.0: w/sum_weights for p, w in total.items()}

    return(sorted([(price, weight) for price, weight in chip_dict.items()], key=lambda x: x[0]))



def aggregate_chips_fast(df):
    total = defaultdict(float)
    for row in df.itertuples():
        for price_weight_tuple in row.筹码比例:
            price_int = int(round(price_weight_tuple[0] * 100))
            total[price_int] += price_weight_tuple[1] * row.survival_ratio * row.换手率
    sum_weights = sum(total.values())
    chip_dict = {p/100.0: w/sum_weights for p, w in total.items()}
    return sorted(chip_dict.items())
    
def chip_distribution(code, target_date):
    
    df = pd.read_csv(stock_path + '/%s.csv' % code, parse_dates=['交易日期'], on_bad_lines='warn')
    #print(df.head())

    df = df.loc[df['交易日期'] <= target_date]
    print(df.head())
    print(df.shape)

    start_time = time.time()  # 记录开始时间
    compute_survival_ratio(df)
    end_time1 = time.time()   # 记录结束时间

    #calc_chip_everyday(df)
    calc_chip_everyday_fast(df)
    end_time2 = time.time()   # 记录结束时间

    df.loc[0, '换手率'] = 1.0 # 这里需要注意一下，第一行的换手率是 1.0



    #a = aggregate_chips(df)
    a = aggregate_chips_fast(df)
    end_time3 = time.time()   # 记录结束时间

    print(end_time1 - start_time, end_time2 - end_time1, end_time3 - end_time2
)


    target_price = df.loc[df['交易日期'] == target_date, '收盘价'].iloc[0]
    
    #b = calculate_chip_metrics(a, target_price)
    #print(b)
    
    #draw_chip_distribution(code, a)

    #print(a)

    # sum_value = 0
    # for i in range(len(df)):
    #     if i == 0:
    #         sum_value = df.iloc[i]['survival_ratio']
    #     else:
    #         sum_value += df.iloc[i]['survival_ratio'] * df.iloc[i]['换手率']
    # print(f"Total survival ratio sum: {sum_value}")


if __name__ == '__main__':
    code_list = ['sh601872']
    #code_list = ['sh601872','sz301631','bj920082','sz002445','sz300347','sh600713','sh603192','sh603171']
    #code = 'sz301631'
    #code = 'sh601872'

    #code = 'bj920082'
    target_date = '2025-06-25'

    for code in code_list:
        start_time = time.time()  # 记录开始时间
        chip_distribution(code, target_date)
        end_time = time.time()   # 记录结束时间
        
        elapsed = end_time - start_time  # 计算耗时（秒）
        print(f"【{code}】耗时: {elapsed:.4f} 秒")

        