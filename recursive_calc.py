"""
计算筹码分布
这个计算是基于赋权的股票数据
2025.06.12

"""
import time
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar
from tqdm import tqdm
import matplotlib.pyplot as plt
from draw_vertically import draw_chip_distribution
from current_chip_distribution import calc_current_chip,calc_current_chip_original
from chip_stat import calculate_chip_metrics

stock_path = '/home/ubuntu/Data_Center/stock_daily_simple/stock_data_new'

def calc_chip_distribution(chip_list_current, chip_list_previous, t_rate, price_step = 0.01):
    '''
    该方法为计算当前周期筹码分布
    params: chip_list_current: 当周期筹码分布
    params: chip_list_previous: 前一周期筹码分布
    params: t_rate: 换手率
    params: price_step: 价格步长
    return: chip_list: 筹码分布，列表格式
    '''
    # 将筹码分布转换为DataFrame

    df_current = pd.DataFrame(chip_list_current, columns=[
                              '价格', '筹码比例_current']).sort_values(by='价格')
    df_previous = pd.DataFrame(chip_list_previous, columns=[
                               '价格', '筹码比例_previous']).sort_values(by='价格')

    # 将价格转换为整数，防止浮点数计算误差
    df_current['价格'] = (df_current['价格']/price_step).round().astype(int)
    df_previous['价格'] = (df_previous['价格']/price_step).round().astype(int)

    df_previous = df_previous.groupby(
        '价格')['筹码比例_previous'].sum().reset_index()

    if t_rate > 1:
        t_rate = 1.0

    # 计算当周期筹码分布
    df_current['筹码比例_current'] = df_current['筹码比例_current']*(t_rate)
    df_previous['筹码比例_previous'] = df_previous['筹码比例_previous']*(1-t_rate)

    df = pd.merge(left=df_current, right=df_previous, on='价格', how='outer')
    df.fillna(0, inplace=True)
    df['筹码比例'] = df['筹码比例_current'] + df['筹码比例_previous']

    # 将价格恢复至小数状态
    df['价格'] = df['价格']*price_step

    # 格式化数据
    df.drop(columns=['筹码比例_current', '筹码比例_previous'], inplace=True)
    df.sort_values(by='价格', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 返回筹码分布
    chip_list = df.values.tolist()
    return chip_list



def insert_chip(df, price_step=0.01, min_chip=10):
    '''
    params: df: 行情和财务的混合数据
    params: min_chip: 最小筹码比例, 如果筹码比例小于10^(-min_chip), 则删除该价格, 并重新计算筹码比例, 使筹码总和为1
    return: df: 增加筹码分布的股票数据
    '''
    #price_step = 0.01
    min_chip = 10**(-min_chip)
    df.insert(len(df.columns), '筹码比例', '')
    df['筹码比例'] = df['筹码比例'].astype(object)


    for index, row in tqdm(df.iterrows()):
        if index == 0 :
            chip_list_previous = [[row['开盘价'], 0]]###
            
        if index >= 1:
            chip_list_previous = df.loc[index-1, '筹码比例']
            

        #chip_list_current = calc_current_chip(
        #    row['最低价'], row['最高价'], row['交易均价'], price_step)
        
        chip_list_current = calc_current_chip_original( 
            row['最低价'], row['最高价'], row['交易均价'], row['收盘价'],price_step)
        
        chip_list_current = calc_chip_distribution(
            chip_list_current, chip_list_previous, row['换手率'], price_step)

        # 删除筹码比例小于min_chip的价格
        chip_list_current = [
            chip for chip in chip_list_current if chip[1] >= min_chip]

        # 重新计算筹码比例，使其总和为1
        total_ratio = sum(ratio for price, ratio in chip_list_current)

        chip_list_current = [[price, ratio / total_ratio]
                             for price, ratio in chip_list_current]

        df.at[index, '筹码比例'] = chip_list_current

    return df



def chip_distribution(code, target_date):
    
    df = pd.read_csv(stock_path + '/%s.csv' % code, parse_dates=['交易日期'], on_bad_lines='warn')
    print(df.head())

    df = df.loc[df['交易日期'] <= target_date]
    print(df.tail())

    df = insert_chip(df, 0.01, 6)
    data = df.loc[df['交易日期'] == target_date, '筹码比例'].iloc[0]

    target_price = df.loc[df['交易日期'] == target_date, '收盘价'].iloc[0]
    
    #result = calculate_chip_metrics(data, target_price)
    #print(result)

    #draw_chip_distribution(code, data,'recurve')



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