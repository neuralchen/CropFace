import time
import gradio as gr
 
def long_running_task(progress_bar):
    for i in range(100):
        time.sleep(0.1)  # 模拟耗时操作
        progress_bar.set(i + 1)  # 更新进度条的值
 
interface = gr.Interface(long_running_task, "progress_bar")
interface.launch()