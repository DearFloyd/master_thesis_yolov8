import gradio as gr
import numpy as np
import time
import altair as alt
import pandas as pd


def greet1(name):
    return "Hello " + name + "!"

def greet2(name, is_morning, temperature):
    salutation = "Good morning" if is_morning else "Good evening"
    greeting = f"{salutation} {name}. It is {temperature} degrees today"
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius, 2)

def sepia(input_img):
    
    # 处理图像
    sepia_filter = np.array([
        [0.393, 0.769, 0.189], 
        [0.349, 0.686, 0.168], 
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img

def flip_text(x):
    return x[::-1]

def flip_image(x):
    return np.fliplr(x)

def detect(img):
    return
    if isinstance(img,str):
        img = get_url_img(img) if img.startswith('http') else Image.open(img).convert('RGB')
    result = model.predict(source=img)
    if len(result[0].boxes.boxes)>0:
        vis = plots.plot_detection(img,boxes=result[0].boxes.boxes,
                     class_names=class_names, min_score=0.2)
    else:
        vis = img

    return vis



if __name__ == "__main__":
    # 接口创建函数
    # fn设置处理函数，inputs设置输入接口组件，outputs设置输出接口组件
    # fn,inputs,outputs都是必填函数
    # demo = gr.Interface(fn=greet1, inputs="text", outputs="text")
    # demo = gr.Interface(
    # fn=greet2,
    # # 按照处理程序设置输入组件
    # inputs=["text", "checkbox", gr.Slider(0, 100)],
    # # 按照处理程序设置输出组件
    # outputs=["text", "number"],)

    # demo = gr.Interface(sepia, gr.Image(shape=(200, 200)), "image")

    # # 多级tab组件
    # with gr.Blocks() as demo:
    #     #用markdown语法编辑输出一段话
    #     gr.Markdown("Flip text or image files using this demo.")
    #     # 设置tab选项卡
    #     with gr.Tab("Flip Text"):
    #         #Blocks特有组件，设置所有子组件按垂直排列
    #         #垂直排列是默认情况，不加也没关系
    #         with gr.Column():
    #             text_input = gr.Textbox()
    #             text_output = gr.Textbox()
    #             text_button = gr.Button("Flip")
    #     with gr.Tab("Flip Image"):
    #         #Blocks特有组件，设置所有子组件按水平排列
    #         with gr.Row():
    #             image_input = gr.Image()
    #             image_output = gr.Image()
    #         image_button = gr.Button("Flip")  # 这一行放在gr.Row()同级 这样Flip按钮就是垂直放在子组件下面
    #     #设置折叠内容
    #     with gr.Accordion("Open for More!"):
    #         gr.Markdown("Look at me...")
    #     text_button.click(flip_text, inputs=text_input, outputs=text_output)
    #     image_button.click(flip_image, inputs=image_input, outputs=image_output)
    # demo.launch()   

    
    # 目标检测
    with gr.Blocks() as demo:
        gr.Markdown("# 视频解析可视化")

        with gr.Tab("视频解析"):
            in_img = gr.Image(source='upload', type='pil', label='Input Image')
            in_video = gr.Video(source='upload', label='Input Video')
            button = gr.Button("执行检测", variant="primary")

            gr.Markdown("## 预测输出")
            out_img = gr.Image(type='pil')
            button.click(detect,
                        inputs=in_img, 
                        outputs=out_img)

        with gr.Tab("单图展示"):
            #Blocks特有组件，设置所有子组件按垂直排列
            #垂直排列是默认情况，不加也没关系
            with gr.Column():
                text_input = gr.Textbox()
                text_output = gr.Textbox()
                text_button = gr.Button("Flip", variant="primary")
            text_button.click(flip_text, inputs=text_input, outputs=text_output)

        with gr.Tab("大数据分析"):
            #Blocks特有组件，设置所有子组件按水平排列
            with gr.Row():
                image_input = gr.Image(source='upload')
                image_output = gr.Image()
            image_button = gr.Button("Flip", variant="primary")  # 这一行放在gr.Row()同级 这样Flip按钮就是垂直放在子组件下面
            image_button.click(flip_image, inputs=image_input, outputs=image_output)
        
        with gr.Tab("选择测试图片"):
            files = ['people.jpeg','coffee.jpeg','cat.jpeg']
            drop_down = gr.Dropdown(choices=files,value=files[0])
            button = gr.Button("执行检测",variant="primary")
            gr.Markdown("## 预测输出")
            out_img = gr.Image(type='pil')
            
            button.click(detect,
                        inputs=drop_down, 
                        outputs=out_img)
        
        with gr.Tab("输入图片链接"):
            default_url = 'https://t7.baidu.com/it/u=3601447414,1764260638&fm=193&f=GIF'
            url = gr.Textbox(value=default_url)
            button = gr.Button("执行检测",variant="primary")
            
            gr.Markdown("## 预测输出")
            out_img = gr.Image(type='pil')
            
            button.click(detect,
                        inputs=url, 
                        outputs=out_img)
            
        with gr.Tab("上传本地图片"):
            input_img = gr.Image(type='pil')
            button = gr.Button("执行检测",variant="primary")
            
            gr.Markdown("## 预测输出")
            out_img = gr.Image(type='pil')
            
            button.click(detect,
                        inputs=input_img, 
                        outputs=out_img)

    gr.close_all() 
    demo.queue(concurrency_count=5)
    demo.launch()
