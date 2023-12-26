import os
import pandas as pd
import pandas_alive
import plotly.express as px
import PIL
from tqdm import tqdm
from pandas import DataFrame


LABLE_INDEX_MAP = {'frame':0, 'listen':1, 'shelter':2, 'neutrality':3, 'phone':4, 'stand':5, 'write':6}


def info_analyse_to_dataframe_old(infer_info_path, outputpath='result_test11.csv'):
    df = DataFrame(
        columns=['frame', 'listen', 'shelter', 'neutrality', 'phone', 'stand', 'write'],
    )
    with open(infer_info_path, 'r') as f:
        lines = f.readlines(500000)  # 一行数据的格式如 'frame 1, 1 listen, 1 shelter, 1 neutrality, 1 phone, 1 stand, 1 write, \n'
        second_count = 1
        for line in tqdm(lines):
            data = [0] * len(LABLE_INDEX_MAP)
            line_list = line.strip(' ').split(',')
            frame_idx = line_list[0].strip(' ').split(' ')[1]
            if int(frame_idx) % 25 != 0:
                continue
            data[LABLE_INDEX_MAP['frame']] = second_count
            second_count += 1
            for record in line_list[1:-1]:
                if 'listen' in record:  # 注意yolo会对类别标签做单复数处理 需要转换
                    count, _= record.strip(' ').split(' ')
                    data[LABLE_INDEX_MAP['listen']] = count
                elif 'shelter' in record:
                    count, _= record.strip(' ').split(' ')
                    data[LABLE_INDEX_MAP['shelter']] = count
                elif 'neutrality' in record:
                    count, _= record.strip(' ').split(' ')
                    data[LABLE_INDEX_MAP['neutrality']] = count
                elif 'phone' in record:
                    count, _= record.strip(' ').split(' ')
                    data[LABLE_INDEX_MAP['phone']] = count
                elif 'stand' in record:
                    count, _= record.strip(' ').split(' ')
                    data[LABLE_INDEX_MAP['stand']] = count
                elif 'write' in record:
                    count, _= record.strip(' ').split(' ')
                    data[LABLE_INDEX_MAP['write']] = count
            df.loc[frame_idx] = data
    # df.set_index(['frame'], inplace=True)
    df.to_csv(outputpath, sep=',', index=False, header=True)


def info_analyse_to_dataframe(infer_info_path, outputpath='result_test_12_25_stand.csv', stand_only=False):
    df = DataFrame(
        columns=['timesteps', 'action', 'count'],
    )
    with open(infer_info_path, 'r') as f:
        lines = f.readlines(50000)
        for line in tqdm(lines):
            action_info = line.strip(' ').split(',')
            timestep, actions = action_info[0], action_info[1:-1]  # 末尾为换行符
            timestep = timestep.strip(' ').split(' ')[-1]
            m, s = divmod(int(timestep), 60)
            if s < 10:
                s = '0' + str(s)
            h, m = divmod(m, 60)
            timestep = str(h) + ':' + str(m) + ':' + str(s)
            for item in actions:
                data = [0] * len(df.columns)
                data[0] = timestep
                item = item.strip(' ')
                count, action = item.split(' ')
                if 'listen' in action:
                    action = 'listen'
                elif 'shelter' in action:
                    action = 'shelter'
                elif 'neutrality' in action:
                    action = 'neutrality'
                elif 'phone' in action:
                    action = 'phone'
                elif 'stand' in action:
                    action = 'stand'
                elif 'write' in action:
                    action = 'write'
                if stand_only:
                    if action != 'stand':
                        continue
                data[1], data[2] = action, count
                df.loc[len(df)] = data
    df.to_csv(outputpath, sep=',', index=False, header=True)

def analyse_visualization(data_path):
    df = pd.read_csv(data_path)
    df = df.drop(['frame'], axis=1)
    print(df)
    df.index = df.index.strftime("%S")
    # df.index = pd.to_datetime(df.index, format="%S")
    df.index = pd.to_datetime(df.index)
    df.index = df.dt.strftime("%S")
    print(df)
    print('start!')
    # df.plot_animated(
    # 'electricity-generated-australia.gif',  #保存gif名称
    # # period_fmt="%d/%m/%Y",  #动态更新图中时间戳
    # title='Australian Electricity Sources 1980-2018',  #标题
    # perpendicular_bar_func='mean',  #添加均值辅助线
    # # period_summary_func=current_total,  #汇总
    # cmap='Set1',  #定义调色盘
    # n_visible=6,  #柱子显示数
    # orientation='h',)  #柱子方向
    
    # animated_line_chart = df.diff().fillna(0).plot_animated(kind='line')
    # animated_pie_chart = df.plot_animated(kind="pie", 
    #                                       rotatelabels=True, 
    #                                     #   period_label={'x':0.1,'y':0.1},
    #                                     #   tick_label_size=3,
    #                                       period_fmt="%S",
    #                                       )
    # animated_bar_chart = df.plot_animated(kind="race",
    #                                       n_visible=6, 
    #                                       orientation='h', 
    #                                       period_fmt="%S",
    #                                       )
    # pandas_alive.animate_multiple_plots('example-bar-and-line-chart.gif', [animated_bar_chart, animated_pie_chart],)
    print('done!')


if __name__ == "__main__":
    # infer_info_path = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/test_10_24/verbose.txt'
    infer_info_path = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/test_12_25/verbose.txt'
    info_analyse_to_dataframe(infer_info_path)
    
    # analyse_visualization('/workspace/cv-docker/joey04.li/datasets/yolov8-0927/result_test11.csv')
    # source = pd.read_csv("/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/result_test_11_15.csv")
    # source['timesteps'] = pd.to_datetime(source['timesteps'], format='%H:%M:%S')
    # source['timesteps'] = pd.to_timedelta(source['timesteps'])
    # source['timesteps'] = source['timesteps'].strftime("%H:%M:%S")
    # source['timesteps'] = source['timesteps'].dt.time
    # source.to_csv('result_test_11_15.csv', sep=',', index=False, header=True)