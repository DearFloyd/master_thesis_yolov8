import gradio as gr
import numpy as np
import time 


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
    demo = gr.Interface(sepia, gr.Image(shape=(200, 200)), "image")
    
    demo.launch()   
