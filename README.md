# Face Crop工具使用说明
 
## 安装
1. 从百度云盘上下载软件：
百度云盘链接：[百度云 (密码：sjtu)](https://pan.baidu.com/s/1F88z-xWA16h82fU1yVN6BA?pwd=sjtu)

2. 解压```crop_face.zip```，***注意解压的文件夹目录不要有中文！！！！***

3. 进入目录```evn/```

4. 双击安装 python ```python-3.10.11-amd64.exe```

    <img src="./docs/installation.PNG"/>

    ***必须勾选红色方框内的选项***

5. 安装好以后测试```python```是否被正确安装

    找到软件根目录下的```test_python.bat```脚本，鼠标左键双击运行它，如果能进入类似于如下画面，说明```python```安装成功，安装窗口即可直接关闭

    <img src="./docs/python.PNG"/>

6. 安装软件的依赖

    安装软件的依赖，找到软件根目录下的```installation.bat```脚本，鼠标右键点击它，选择```以管理员身份运行```:
    <img src="./docs/installation1.png"/>

    等待它安装完成，如果没有报错，即说明安装完成，安装完成后可以关闭安装窗口，如果报错请保留这个界面的截图，方便后续定位错误原因:
    <img src="./docs/finish.png"/>

## 说明
- "批量标注人脸"并不会保存人脸图片，他只是初步扫描整个视频的视频帧的清晰度，筛选符合清晰度、大小标准的人脸。为了加速扫描视频读取使用了跳帧读取，但是结果日志中记录的是每一帧标准的frame index。同时，采用了极小的人脸检测尺寸以加速扫描过程。


## 使用

双击```run.bat```启动软件，***请注意硬盘要留有足够的空间来存放标注结果，推荐40GB的空间***
<img src="./docs/software.png"/>    

#### 功能简介

- "视频目录"：存放视频的目录，需要批次处理的视频全放在这个目录下。
- "选择目录"按钮：点击这个按钮，即可弹出选择路径的窗口，选择存放视频的目录，点击"选择文件夹"按钮即可选定目录：
    <img src="./docs/select_dir.PNG"/> 
- "存放结果目录"：这个目录是用来存放处理后的结果的，它必须与"视频目录"不一致，建议新建一个带日期的文件夹来专门存放。
- "截取大小"：它决定要将截取出来的人脸存放的分辨率大小，在我们的数据标注中统一设置为"1024"。
- "对齐模式"：它决定要用什么方式截取出人脸，一般是人脸五官距离图像边缘的距离以及五官的标准位置。在我们的数据标注中统一设置为"ffhq"。
- "扫描格式"：视频数据的视频格式，在我们的数据标注中统一设置为"MP4"。
- "批量标注人脸"按钮：注意一定要是"视频目录"与"存放结果目录"要事先设置好，点击这个按钮后，软件将会把"视频目录"中的所有视频标注，标注结果将存放在"存放结果目录"下的以视频名建立的新文件夹中。

    批量标注结果：
    <img src="./docs/results.PNG"/>

    单个视频标注结果，视频标注的结果被存储在```2K1A9663_face_log.txt```这个txt文件中，***请注意千万不要修改这个文件！！！***
    <img src="./docs/single_result.PNG"/>



- "最小尺寸"：决定要从视频中保留的图像大小比例，它是"截取大小"数值的倍数，在我们的数据标注中统一选择"0.8"。

- "格式"：决定人脸图片被保存为何种图片格式，在我们的数据标注中统一选择"png"。
- "扩图倍数"：直接剪裁人脸会导致人脸质量降低，因此需要对人脸做一次扩图，扩图越高截出来的人脸质量越高，但会导致软件变慢，在我们的数据标注中统一选择"16"。
- "旋转"：如果拍摄视频不是人像正常直立，比如：倒像。那需要根据旋转的方向决定这个旋转值。
- "批量保存人脸"：当批量标注人脸结束后，不要关闭当前处理的界面，直接点击这个按钮即可开始保存标注的人脸。

    软件保存的某一个视频的人脸剪裁结果：
    <img src="./docs/saving.PNG"/>

- "停止处理"按钮：这个按钮即停止目前正在处理的过程，请注意软件没有设置断点处理，因此点击停止后，再需要处理原过程，软件会从头开始再处理一遍，因此谨慎使用！


# 人脸清洗

### 标准人脸例子

<img src="./docs/face_standard.PNG"/>

### 需要清理的人脸

- 人脸模糊的图像，非常不锐利， 明显带有失焦的特征

    <img src="./docs/blur1.png"/>
    <img src="./docs/blur2.png"/>
    <img src="./docs/blur3.png"/>
    <img src="./docs/blur4.png"/>


- 五官大部分被遮挡
    <img src="./docs/zhedang.png"/>
    <img src="./docs/zhedang1.png"/>
    <img src="./docs/zhedang2.png"/>