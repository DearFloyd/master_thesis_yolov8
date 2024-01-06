### gradio-4.3.0 ###
import gradio as gr
import numpy as np
import argparse, warnings
import os
import altair as alt
import pandas as pd
from ultralytics import YOLO
from vega_datasets import data
from analyse import infer_bar_chart, infer_multi_line, mean_ratio


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/train/yolov8s-C2f-EMSC-attention-head--164bs-500ep/weights/best.pt', help='training model path')
    parser.add_argument('--source', type=str, default='/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/src_img/split_output6.mp4', help='source directory for images or videos')
    parser.add_argument('--conf', type=float, default=0.35, help='object confidence threshold for detection')
    parser.add_argument('--iou', type=float, default=0.6, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--mode', type=str, default='predict', choices=['predict', 'track'], help='predict mode or track mode')
    parser.add_argument('--project', type=str, default='runs/detect', help='project name')
    parser.add_argument('--name', type=str, default='split_output6', help='experiment name (project/name)')
    parser.add_argument('--show', action="store_true", help='show results if possible')
    parser.add_argument('--save_verbose', type=str, default='/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output6/verbose.txt', help='save detail predict verbose results as .txt file')
    parser.add_argument('--save_txt', action="store_true", help='save results as .txt file')
    parser.add_argument('--save_conf', action="store_true", help='save results with confidence scores')
    parser.add_argument('--show_labels', action="store_true", default=True, help='show object labels in plots')
    parser.add_argument('--show_conf', action="store_true", default=False, help='show object confidence scores in plots')
    parser.add_argument('--vid_stride', type=int, default=25, help='video frame-rate stride')
    parser.add_argument('--line_width', type=int, default=1, help='line width of the bounding boxes')
    parser.add_argument('--visualize', action="store_true", help='visualize model features')
    parser.add_argument('--augment', action="store_true", help='apply image augmentation to prediction sources')
    parser.add_argument('--agnostic_nms', action="store_true", help='class-agnostic NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--retina_masks', action="store_true", help='use high-resolution segmentation masks')
    parser.add_argument('--boxes', action="store_true", default=True, help='Show boxes in segmentation predictions')
    parser.add_argument('--save', action="store_true", default=True, help='save result')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', choices=['botsort.yaml', 'bytetrack.yaml', 'deepocsort.yaml', 'hybirdsort.yaml', 'ocsort.yaml'], help='tracker type, [botsort.yaml, bytetrack.yaml, deepocsort.yaml, hybirdsort.yaml, ocsort.yaml]')
    parser.add_argument('--reid_weight', type=str, default='/workspace/cv-docker/joey04.li/datasets/yolo_tracking/examples/weights/osnet_x1_0_imagenet.pt', help='if tracker have reid, add reid model path')
    parser.add_argument('--exist-ok', default=True, help='existing project/name ok, do not increment')
    
    return parser.parse_known_args()[0]


def transformer_opt(opt):
    opt = vars(opt)
    del opt['source']
    del opt['weight']
    del opt['save_verbose']
    return opt


class YOLOV8(YOLO):
    '''
    weigth:model path
    '''
    def __init__(self, weight='', task=None) -> None:
        super().__init__(weight, task)


def yolov8_detect():
    opt = parse_opt()
    
    model = YOLOV8(weight=opt.weight)
    verbose_path = opt.save_verbose
    if opt.mode == 'predict':
        model.predict(source=opt.source, verbose_path=verbose_path, **transformer_opt(opt))


def make_plot_00_15(plot_type):
    infer_info_path = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output1/verbose.txt'
    outputpath_mark_line = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output1/mark_line.csv'
    outputpath_mark_bar = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output1/mark_bar.csv'
    outputpath_radial = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output1/radial.csv'

    if not os.path.exists(outputpath_mark_line):
        infer_multi_line(infer_info_path, outputpath_mark_line)
    if not os.path.exists(outputpath_mark_bar):
        infer_bar_chart(infer_info_path, outputpath_mark_bar)
    if not os.path.exists(outputpath_radial):
        mean_ratio(infer_info_path, outputpath_radial)
    
    if plot_type == "mutiline":
        source = pd.read_csv(outputpath_mark_line)
        return alt.Chart(source, title="0-15分钟分布").mark_line().encode(
            x='timesteps',
            y='count',
            color='action',
        ).properties(width=1000, height=500)

    elif plot_type == "bar_chart":
        source = pd.read_csv(outputpath_mark_bar)
        return alt.Chart(source, title="0-15分钟状态比例").mark_bar(
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3
        ).encode(
            x='timesteps',
            y='count():Q',
            color='state',
        ).properties(width=1300, height=500)  # 需要记得修改altair库中的/altair/vegalite/data.py中的max_rows: int 从5000到50000
    
    elif plot_type == "radial":
        mean_ratio_radio_source = pd.read_csv(outputpath_radial)
        base = alt.Chart(mean_ratio_radio_source).encode(
                theta=alt.Theta("count:Q", stack=True),
                # radius=alt.Radius("state", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
                color="state:N",
            )
        c1 = base.mark_arc(outerRadius=160, stroke="#fff")
        c2 = base.mark_text(radius=190, size=10).encode(text="count:Q")
        return c1 + c2


def make_plot_15_30(plot_type):
    infer_info_path = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output2/verbose.txt'
    outputpath_mark_line = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output2/mark_line.csv'
    outputpath_mark_bar = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output2/mark_bar.csv'
    outputpath_radial = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output2/radial.csv'

    if not os.path.exists(outputpath_mark_line):
        infer_multi_line(infer_info_path, outputpath_mark_line)
    if not os.path.exists(outputpath_mark_bar):
        infer_bar_chart(infer_info_path, outputpath_mark_bar)
    if not os.path.exists(outputpath_radial):
        mean_ratio(infer_info_path, outputpath_radial)

    if plot_type == "mutiline":
        source = pd.read_csv(outputpath_mark_line)
        return alt.Chart(source, title="15-30分钟分布").mark_line().encode(
            x='timesteps',
            y='count',
            color='action',
        ).properties(width=1000, height=500)
    
    elif plot_type == "bar_chart":
        source = pd.read_csv(outputpath_mark_bar)
        return alt.Chart(source, title="15-30分钟状态比例").mark_bar(
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3
        ).encode(
            x='timesteps',
            y='count():Q',
            color='state',
        ).properties(width=1300, height=500)
    
    elif plot_type == "radial":
        mean_ratio_radio_source = pd.read_csv(outputpath_radial)
        base = alt.Chart(mean_ratio_radio_source).encode(
                theta=alt.Theta("count:Q", stack=True),
                # radius=alt.Radius("state", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
                color="state:N",
            )
        c1 = base.mark_arc(outerRadius=160, stroke="#fff")
        c2 = base.mark_text(radius=190, size=10).encode(text="count:Q")
        return c1 + c2
    

def make_plot_30_45(plot_type):
    infer_info_path = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output3/verbose.txt'
    outputpath_mark_line = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output3/mark_line.csv'
    outputpath_mark_bar = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output3/mark_bar.csv'
    outputpath_radial = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output3/radial.csv'

    if not os.path.exists(outputpath_mark_line):
        infer_multi_line(infer_info_path, outputpath_mark_line)
    if not os.path.exists(outputpath_mark_bar):
        infer_bar_chart(infer_info_path, outputpath_mark_bar)
    if not os.path.exists(outputpath_radial):
        mean_ratio(infer_info_path, outputpath_radial)

    if plot_type == "mutiline":
        source = pd.read_csv(outputpath_mark_line)
        return alt.Chart(source, title="30-45分钟分布").mark_line().encode(
            x='timesteps',
            y='count',
            color='action',
        ).properties(width=1000, height=500)
    
    elif plot_type == "bar_chart":
        source = pd.read_csv(outputpath_mark_bar)
        return alt.Chart(source, title="30-45分钟状态比例").mark_bar(
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3
        ).encode(
            x='timesteps',
            y='count():Q',
            color='state',
        ).properties(width=1300, height=500)
    
    elif plot_type == "radial":
        mean_ratio_radio_source = pd.read_csv(outputpath_radial)
        base = alt.Chart(mean_ratio_radio_source).encode(
                theta=alt.Theta("count:Q", stack=True),
                # radius=alt.Radius("state", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
                color="state:N",
            )
        c1 = base.mark_arc(outerRadius=160, stroke="#fff")
        c2 = base.mark_text(radius=190, size=10).encode(text="count:Q")
        return c1 + c2
    

def make_plot_45_60(plot_type):
    infer_info_path = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output4/verbose.txt'
    outputpath_mark_line = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output4/mark_line.csv'
    outputpath_mark_bar = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output4/mark_bar.csv'
    outputpath_radial = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output4/radial.csv'

    if not os.path.exists(outputpath_mark_line):
        infer_multi_line(infer_info_path, outputpath_mark_line)
    if not os.path.exists(outputpath_mark_bar):
        infer_bar_chart(infer_info_path, outputpath_mark_bar)
    if not os.path.exists(outputpath_radial):
        mean_ratio(infer_info_path, outputpath_radial)

    if plot_type == "mutiline":
        source = pd.read_csv(outputpath_mark_line)
        return alt.Chart(source, title="45-60分钟分布").mark_line().encode(
            x='timesteps',
            y='count',
            color='action',
        ).properties(width=1000, height=500)
    
    elif plot_type == "bar_chart":
        source = pd.read_csv(outputpath_mark_bar)
        return alt.Chart(source, title="45-60分钟状态比例").mark_bar(
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3
        ).encode(
            x='timesteps',
            y='count():Q',
            color='state',
        ).properties(width=1300, height=500)
    
    elif plot_type == "radial":
        mean_ratio_radio_source = pd.read_csv(outputpath_radial)
        base = alt.Chart(mean_ratio_radio_source).encode(
                theta=alt.Theta("count:Q", stack=True),
                # radius=alt.Radius("state", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
                color="state:N",
            )
        c1 = base.mark_arc(outerRadius=160, stroke="#fff")
        c2 = base.mark_text(radius=190, size=10).encode(text="count:Q")
        return c1 + c2


def make_plot_60_75(plot_type):
    infer_info_path = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output5/verbose.txt'
    outputpath_mark_line = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output5/mark_line.csv'
    outputpath_mark_bar = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output5/mark_bar.csv'
    outputpath_radial = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output5/radial.csv'

    if not os.path.exists(outputpath_mark_line):
        infer_multi_line(infer_info_path, outputpath_mark_line)
    if not os.path.exists(outputpath_mark_bar):
        infer_bar_chart(infer_info_path, outputpath_mark_bar)
    if not os.path.exists(outputpath_radial):
        mean_ratio(infer_info_path, outputpath_radial)

    if plot_type == "mutiline":
        source = pd.read_csv(outputpath_mark_line)
        return alt.Chart(source, title="60-75分钟分布").mark_line().encode(
            x='timesteps',
            y='count',
            color='action',
        ).properties(width=1000, height=500)
    
    elif plot_type == "bar_chart":
        source = pd.read_csv(outputpath_mark_bar)
        return alt.Chart(source, title="60-75分钟状态比例").mark_bar(
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3
        ).encode(
            x='timesteps',
            y='count():Q',
            color='state',
        ).properties(width=1300, height=500)
    
    elif plot_type == "radial":
        mean_ratio_radio_source = pd.read_csv(outputpath_radial)
        base = alt.Chart(mean_ratio_radio_source).encode(
                theta=alt.Theta("count:Q", stack=True),
                # radius=alt.Radius("state", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
                color="state:N",
            )
        c1 = base.mark_arc(outerRadius=160, stroke="#fff")
        c2 = base.mark_text(radius=190, size=10).encode(text="count:Q")
        return c1 + c2
    

def make_plot_75_90(plot_type):
    infer_info_path = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output6/verbose.txt'
    outputpath_mark_line = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output6/mark_line.csv'
    outputpath_mark_bar = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output6/mark_bar.csv'
    outputpath_radial = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/split_output6/radial.csv'

    if not os.path.exists(outputpath_mark_line):
        infer_multi_line(infer_info_path, outputpath_mark_line)
    if not os.path.exists(outputpath_mark_bar):
        infer_bar_chart(infer_info_path, outputpath_mark_bar)
    if not os.path.exists(outputpath_radial):
        mean_ratio(infer_info_path, outputpath_radial)


    if plot_type == "mutiline":
        source = pd.read_csv(outputpath_mark_line)
        return alt.Chart(source, title="75-90分钟分布").mark_line().encode(
            x='timesteps',
            y='count',
            color='action',
        ).properties(width=1000, height=500)
    
    elif plot_type == "bar_chart":
        source = pd.read_csv(outputpath_mark_bar)
        return alt.Chart(source, title="75-90分钟状态比例").mark_bar(
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3
        ).encode(
            x='timesteps',
            y='count():Q',
            color='state',
        ).properties(width=1300, height=500)
    
    elif plot_type == "radial":
        mean_ratio_radio_source = pd.read_csv(outputpath_radial)
        base = alt.Chart(mean_ratio_radio_source, title="75-90分钟片段").encode(
                theta=alt.Theta("count:Q", stack=True),
                # radius=alt.Radius("state", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
                color="state:N",
            )
        c1 = base.mark_arc(outerRadius=160, stroke="#fff")
        c2 = base.mark_text(radius=190, size=10).encode(text="count:Q")
        return c1 + c2
    

if __name__ == "__main__":

    with gr.Blocks() as demo:
        button = gr.Radio(label="Plot type",
                          choices=['mutiline',
                                #    'bar_chart',
                                   'radial'], value='radial')
        plot1 = gr.Plot(label="Plot")
        plot2 = gr.Plot(label="Plot")
        plot3 = gr.Plot(label="Plot")
        plot4 = gr.Plot(label="Plot")
        plot5 = gr.Plot(label="Plot")
        plot6 = gr.Plot(label="Plot")

        button.change(make_plot_00_15, inputs=button, outputs=[plot1])
        # button.change(make_plot_15_30, inputs=button, outputs=[plot2])
        # button.change(make_plot_30_45, inputs=button, outputs=[plot3])
        # button.change(make_plot_45_60, inputs=button, outputs=[plot4])
        # button.change(make_plot_60_75, inputs=button, outputs=[plot5])
        button.change(make_plot_75_90, inputs=button, outputs=[plot6])

        demo.load(make_plot_00_15, inputs=[button], outputs=[plot1])
        # demo.load(make_plot_15_30, inputs=[button], outputs=[plot2])
        # demo.load(make_plot_30_45, inputs=[button], outputs=[plot3])
        # demo.load(make_plot_45_60, inputs=[button], outputs=[plot4])
        # demo.load(make_plot_60_75, inputs=[button], outputs=[plot5])
        demo.load(make_plot_75_90, inputs=[button], outputs=[plot6])
        #Blocks特有组件，设置所有子组件按水平排列
        with gr.Row():
            image_input = gr.Image(sources='upload')
            image_output = gr.Image()
        image_button = gr.Button("Flip", variant="primary")  # 这一行放在gr.Row()同级 这样Flip按钮就是垂直放在子组件下面
        # image_button.click(flip_image, inputs=image_input, outputs=image_output)
    demo.launch()
