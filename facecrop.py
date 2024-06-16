#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: face_crop_video.py
# Created Date: Tuesday February 1st 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 28th April 2022 10:04:25 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################


import  os
import  cv2
import  sys
import  json
import  glob
import  shutil
from    tqdm import tqdm
import  tkinter
from    tkinter.filedialog import askdirectory

from   utilities.json_config import readConfig

import threading
import tkinter as tk
import tkinter.ttk as ttk

from sklearn.metrics import mean_squared_error
from insightface_func.face_detect_crop_multi_highresolution import Face_detect_crop

class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.configure(state="disabled")
        self.widget.see(tk.END)
    
    def flush(self):
        pass

#############################################################
# Main Class
#############################################################

class Application(tk.Frame):


    def __init__(self, master=None):
        tk.Frame.__init__(self, master,bg='black')
        # self.font_size = 16
        self.font_list = ("Times New Roman",14)
        self.padx = 5
        self.pady = 5
        self.stop_sign =False
        self.window_init()
    
    def __label_text__(self, usr, root):
        return "User Name:  %s\nWorkspace:  %s"%(usr, root)

    def window_init(self):
        cwd = os.getcwd()
        self.master.title('Face Crop Tool for Video - %s'%cwd)
        # self.master.iconbitmap('./utilities/_logo.ico')
        self.master.geometry("{}x{}".format(700, 700))

        font_list = self.font_list
        
        #################################################################################################
        list_frame    = tk.Frame(self.master)
        list_frame.pack(fill="both", padx=5,pady=5)
        list_frame.columnconfigure(0, weight=1)
        list_frame.columnconfigure(1, weight=1)
        list_frame.columnconfigure(2, weight=1)

        self.img_path = tkinter.StringVar()

        tk.Label(list_frame, text="视频目录:",font=font_list,justify="left")\
                    .grid(row=0,column=0,sticky=tk.EW)
        
        tk.Entry(list_frame, textvariable= self.img_path, font=font_list)\
                    .grid(row=0,column=1,sticky=tk.EW)
        

        tk.Button(list_frame, text = "选择目录", font=font_list,
                    command = self.Select, bg='#F4A460', fg='#F5F5F5')\
                    .grid(row=0,column=2,sticky=tk.EW)
        #################################################################################################
        list_frame1    = tk.Frame(self.master)
        list_frame1.pack(fill="both", padx=5,pady=5)
        list_frame1.columnconfigure(0, weight=1)
        list_frame1.columnconfigure(1, weight=1)
        list_frame1.columnconfigure(2, weight=1)

        self.save_path = tkinter.StringVar()

        tk.Label(list_frame1, text="存放结果目录:",font=font_list,justify="left")\
                    .grid(row=0,column=0,sticky=tk.EW)
        
        tk.Entry(list_frame1, textvariable= self.save_path, font=font_list)\
                    .grid(row=0,column=1,sticky=tk.EW)
        

        tk.Button(list_frame1, text = "选择目录", font=font_list,
                    command = self.Select_Target, bg='#F4A460', fg='#F5F5F5')\
                    .grid(row=0,column=2,sticky=tk.EW)
        
        #################################################################################################
        label_frame    = tk.Frame(self.master)
        label_frame.pack(fill="both", padx=5,pady=5)
        label_frame.columnconfigure(0, weight=1)
        label_frame.columnconfigure(1, weight=1)
        label_frame.columnconfigure(2, weight=1)
        label_frame.columnconfigure(3, weight=1)
        # label_frame.columnconfigure(4, weight=1)

        tk.Label(label_frame, text="截取大小:",font=font_list,justify="left")\
                    .grid(row=0,column=0,sticky=tk.EW)
        
        tk.Label(label_frame, text="对齐模式:",font=font_list,justify="left")\
                    .grid(row=0,column=1,sticky=tk.EW)

        # tk.Label(label_frame, text="Target Format:",font=font_list,justify="left")\
        #             .grid(row=0,column=2,sticky=tk.EW)
        
        tk.Label(label_frame, text="模糊阈值:",font=font_list,justify="left")\
                    .grid(row=0,column=2,sticky=tk.EW)
        

        tk.Label(label_frame, text="帧间隔:",font=font_list,justify="left")\
                    .grid(row=0,column=3,sticky=tk.EW)
        
        #################################################################################################

        test_frame    = tk.Frame(self.master)
        test_frame.pack(fill="both", padx=5,pady=5)
        test_frame.columnconfigure(0, weight=1)
        test_frame.columnconfigure(1, weight=1)
        test_frame.columnconfigure(2, weight=1)
        test_frame.columnconfigure(3, weight=1)
        test_frame.columnconfigure(4, weight=1)

        self.test_var = tkinter.StringVar()

        self.test_com = ttk.Combobox(test_frame, textvariable=self.test_var)
        self.test_com.grid(row=0,column=0,sticky=tk.EW)
        self.test_com["value"] = [256,512,768,1024]
        self.test_com.current(1)

        self.align_var = tkinter.StringVar()
        self.align_com = ttk.Combobox(test_frame, textvariable=self.align_var)
        self.align_com.grid(row=0,column=1,sticky=tk.EW)
        self.align_com["value"] = ["VGGFace","ffhq"]
        self.align_com.current(1)

        self.thredhold = tkinter.StringVar()
        tk.Entry(test_frame, textvariable= self.thredhold, font=font_list)\
                    .grid(row=0,column=3,sticky=tk.EW)
        self.thredhold.set("20")

        self.frame_interv = tkinter.StringVar()
        tk.Entry(test_frame, textvariable= self.frame_interv, font=font_list)\
                    .grid(row=0,column=4,sticky=tk.EW)
        self.frame_interv.set("60")

        #################################################################################################

        sep = ttk.Separator(self.master, orient=tk.HORIZONTAL)  # VERTICAL为竖的分割线
        sep.pack(padx=10, fill=tk.X)  # 我试了一下，去掉fill=X则分隔符不出现

        #################################################################################################

        
        scale_frame    = tk.Frame(self.master)
        scale_frame.pack(fill="both", padx=5,pady=5)
        scale_frame.columnconfigure(0, weight=1)
        scale_frame.columnconfigure(1, weight=1)
        scale_frame.columnconfigure(2, weight=1)
        scale_frame.columnconfigure(3, weight=1)
        scale_frame.columnconfigure(4, weight=1)
        scale_frame.columnconfigure(5, weight=1)
        # label_frame.columnconfigure(2, weight=1)

        tk.Label(scale_frame, text="最小尺寸:",font=font_list,justify="left")\
                    .grid(row=0,column=0,sticky=tk.EW)
        self.min_scale = tkinter.StringVar()

        self.image_scale = ttk.Combobox(scale_frame, textvariable=self.min_scale)
        self.image_scale.grid(row=0,column=1,sticky=tk.EW)
        self.image_scale["value"] = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
        self.image_scale.current(3)

        tk.Label(scale_frame, text="快照尺寸",font=font_list,justify="left")\
                    .grid(row=0,column=2,sticky=tk.EW)
        self.snap_size_var = tkinter.StringVar()

        self.snap_size = ttk.Combobox(scale_frame, textvariable=self.snap_size_var)
        self.snap_size.grid(row=0,column=3,sticky=tk.EW)
        self.snap_size["value"] = [2,4,8,16,32]
        self.snap_size.current(3)


        tk.Label(scale_frame, text="扫描格式",font=font_list,justify="left")\
                    .grid(row=0,column=4,sticky=tk.EW)
        self.suffix_name_var = tkinter.StringVar()

        self.suffix_name = ttk.Combobox(scale_frame, textvariable=self.suffix_name_var)
        self.suffix_name.grid(row=0,column=5,sticky=tk.EW)
        self.suffix_name["value"] = ["MP4","MKV","AVI"]
        self.suffix_name.current(0)
        

        #################################################################################################
        test_frame1    = tk.Frame(self.master)
        test_frame1.pack(fill="both", padx=5,pady=5)
        test_frame1.columnconfigure(0, weight=1)
        test_frame1.columnconfigure(1, weight=12)

        self.crop_ckpt_var = tkinter.StringVar()
        self.crop_ckpt_com = ttk.Combobox(test_frame1, textvariable=self.crop_ckpt_var)
        self.crop_ckpt_com.grid(row=0,column=0,sticky=tk.EW)
        self.crop_ckpt_com["value"] = ["从头开始", "继续上次"]
        self.crop_ckpt_com.current(0)

        test_update_button = tk.Button(test_frame1, text = "批量标注人脸",
                            font=font_list, command = self.Crop, bg='#F4A460', fg='#F5F5F5')
        test_update_button.grid(row=0,column=1,sticky=tk.EW)
        #################################################################################################

        sep = ttk.Separator(self.master, orient=tk.HORIZONTAL)  # VERTICAL为竖的分割线
        sep.pack(padx=10, fill=tk.X)  # 我试了一下，去掉fill=X则分隔符不出现


        #################################################################################################

        

        save_frame    = tk.Frame(self.master)
        save_frame.pack(fill="both", padx=5,pady=5)
        save_frame.columnconfigure(0, weight=1)
        save_frame.columnconfigure(1, weight=1)
        save_frame.columnconfigure(2, weight=1)
        save_frame.columnconfigure(3, weight=1)
        save_frame.columnconfigure(4, weight=1)
        save_frame.columnconfigure(5, weight=1)
        # label_frame.columnconfigure(2, weight=1)

        tk.Label(save_frame, text="格式",font=font_list,justify="left")\
                    .grid(row=0,column=0,sticky=tk.EW)

        self.format_var = tkinter.StringVar()
        self.format_com = ttk.Combobox(save_frame, textvariable=self.format_var)
        self.format_com.grid(row=0,column=1,sticky=tk.EW)
        self.format_com["value"] = ["png","jpg"]
        self.format_com.current(0)

        tk.Label(save_frame, text="扩图倍数",font=font_list,justify="left")\
                    .grid(row=0,column=2,sticky=tk.EW)
        self.affine_size_var = tkinter.StringVar()

        self.affine_size = ttk.Combobox(save_frame, textvariable=self.affine_size_var)
        self.affine_size.grid(row=0,column=3,sticky=tk.EW)
        self.affine_size["value"] = [1,4,8,10,16,20,24,32]
        self.affine_size.current(2)

        tk.Label(save_frame, text="旋转",font=font_list,justify="left")\
                    .grid(row=0,column=4,sticky=tk.EW)
        self.rotat_size_var = tkinter.StringVar()

        self.rotat_size = ttk.Combobox(save_frame, textvariable=self.rotat_size_var)
        self.rotat_size.grid(row=0,column=5,sticky=tk.EW)
        self.rotat_size["value"] = ["None", "90", "180", "270", "config"]
        self.rotat_size.current(0)
        #################################################################################################

        test_frame1    = tk.Frame(self.master)
        test_frame1.pack(fill="both", padx=5,pady=5)
        test_frame1.columnconfigure(0, weight=1)
        test_frame1.columnconfigure(1, weight=12)

        self.save_ckpt_var = tkinter.StringVar()
        self.save_ckpt_com = ttk.Combobox(test_frame1, textvariable=self.save_ckpt_var)
        self.save_ckpt_com.grid(row=0,column=0,sticky=tk.EW)
        self.save_ckpt_com["value"] = ["从头开始", "继续上次"]
        self.save_ckpt_com.current(0)

        crop_button = tk.Button(test_frame1, text = "批量保存人脸",
                            font=font_list, command = self.Save_Face, bg='#F4A460', fg='#F5F5F5')
        crop_button.grid(row=0,column=1,sticky=tk.EW)

        #################################################################################################
        sep = ttk.Separator(self.master, orient=tk.HORIZONTAL)  # VERTICAL为竖的分割线
        sep.pack(padx=10, fill=tk.X)  # 我试了一下，去掉fill=X则分隔符不出现

        scane_frame1    = tk.Frame(self.master)
        scane_frame1.pack(fill="both", padx=5,pady=5)
        scane_frame1.columnconfigure(0, weight=1)
        # test_frame1.columnconfigure(1, weight=1)

        scane_button = tk.Button(scane_frame1, text = "扫描并保存清洗后数据",
                            font=font_list, command = self.ScaneSave, bg='#00FF00', fg='#F5F5F5')
        scane_button.grid(row=0,column=0,sticky=tk.EW)

        #################################################################################################
        sep = ttk.Separator(self.master, orient=tk.HORIZONTAL)  # VERTICAL为竖的分割线
        sep.pack(padx=10, fill=tk.X)  # 我试了一下，去掉fill=X则分隔符不出现

        stop_frame1    = tk.Frame(self.master)
        stop_frame1.pack(fill="both", padx=5,pady=5)
        stop_frame1.columnconfigure(0, weight=1)
        # test_frame1.columnconfigure(1, weight=1)

        stop_button = tk.Button(stop_frame1, text = "停止处理",
                            font=font_list, command = self.Stop, bg='#FF0000', fg='#F5F5F5')
        stop_button.grid(row=0,column=0,sticky=tk.EW)

        #################################################################################################


        text = tk.Text(self.master, wrap="word")
        text.pack(fill="both",expand="yes", padx=5,pady=5)
        

        sys.stdout = TextRedirector(text, "stdout")
        
        self.init_algorithm()
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def init_algorithm(self):
        self.detect = Face_detect_crop(name='antelope', root='./insightface_func/models')
        
    
    # def __scaning_logs__(self):
    def Select(self):
        thread_update = threading.Thread(target=self.select_task)
        thread_update.start()
    
    def select_task(self):
        
        path = askdirectory()
        if os.path.isdir(path):
            print("Selected source videos directory: %s"%path)
            self.img_path.set(path)
    
    def Select_Target(self):
        thread_update = threading.Thread(target=self.select_target_task)
        thread_update.start()
    
    def select_target_task(self):
        path = askdirectory()
        if os.path.isdir(path):
            print("Selected target directory: %s"%path)
            self.save_path.set(path)
    def Save_Face(self):
        thread_update = threading.Thread(target=self.save_task)
        thread_update.start()
    # save face task
    def save_task(self):
        self.stop_sign=False
        path        = self.img_path.get()
        tg_path     = self.save_path.get()

        if tg_path == "":
            print("Select target path first!")
            return
        rotat       = self.rotat_size_var.get()
        suffix_name = self.suffix_name_var.get()

        affine_size = self.affine_size_var.get()
        affine_size = int(affine_size)
        blur_t      = self.thredhold.get()
        tg_format   = self.format_com.get()
        blur_t      = float(blur_t)

        if rotat == "None":
            rotat = False
        elif rotat == "90":
            rotat = cv2.ROTATE_90_CLOCKWISE
        elif rotat == "180":
            rotat = cv2.ROTATE_180
        elif rotat == "270":
            rotat = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif rotat == "config":
            file_name = os.path.join(path,"video_metainfo.json")
            print(file_name)
            if not os.path.exists(file_name):
                print("Add a video_metainfo.json in root path!")
                return
            rotat_tab = readConfig(file_name)


        print("target path: ",tg_path)
        print("Blurry thredhold %f"%blur_t)

        videos = glob.glob(os.path.join(path,"*.%s"%suffix_name), recursive=False)
        videos = sorted(videos)
        init_detect     = False
        for index_video, i_video in enumerate(videos):
            print("Processing video: %s..........."%i_video)
            if self.stop_sign:
                print("Stop the process!")
                return
            
            basepath    = os.path.splitext(os.path.basename(i_video))[0]
            tg_path_i   = os.path.join(tg_path,basepath)
            file_log_i  = os.path.join(tg_path_i,"%s_face_log.txt"%basepath)

            if rotat == "config":
                i_rotat     = rotat_tab[basepath]
                if i_rotat == 90:
                    i_rotat = cv2.ROTATE_90_CLOCKWISE
                elif i_rotat == 180:
                    i_rotat = cv2.ROTATE_180
                elif i_rotat == 270:
                    i_rotat = cv2.ROTATE_90_COUNTERCLOCKWISE
            else:
                i_rotat     = rotat

            print("target path: ",tg_path_i)
            if not os.path.exists(tg_path_i):
                os.makedirs(tg_path_i)
            if not os.path.exists(file_log_i):
                print("Mark the video first!\n Process exits without any changes!")
                continue
            with open(file_log_i,'r', encoding="utf-8") as log_reader: # ,encoding="UTF-8"
                result_face = log_reader.readlines()
        
            save_log    = os.path.join(tg_path_i,"%s_face_save_log.txt"%basepath)
            if os.path.exists(save_log):
                print("Video: %s exists! Skip it!"%save_log)
                continue
            frame_face_list = []
            temp_face       = []
            current_frame   = -1
            
            for index, i_line in enumerate(result_face):
                if self.stop_sign:
                    print("Stop the process!")
                    return
                str_line = i_line.split(",")
                if index == 0:
                    # id, target path, file path, blur thredhold, frame ivterval, min scale, snap size, crop size, align mode
                    crop_size   = int(str_line[7])
                    mode        = str_line[8].strip()
                    if mode == "VGGFace":
                        mode = "None"
                    if not init_detect:
                        self.detect.prepare(ctx_id = 0, det_thresh=0.6,\
                            det_size=(640,640), mode = mode, crop_size=crop_size)
                        affine_size = crop_size * affine_size
                        init_detect = True
                    with open(save_log, 'w', encoding="utf-8") as i_file:
                        # id, target path, file path, blur thredhold, frame ivterval, min scale, snap size, crop size, align mode
                        i_file.writelines("%s,%s,%f,%d,%s,%s,%d,%s\n"%(tg_path,path,blur_t,affine_size,rotat,tg_format,crop_size,mode))
                    continue
                if current_frame != int(str_line[1]):
                    current_frame = int(str_line[1])
                    if len(temp_face) >= 2:
                        frame_face_list.append(temp_face)
                        str_1 = str(temp_face[0])
                        for i_str in temp_face[1:]:
                            str_1 += ("||"+",".join(i_str))
                        with open(save_log, 'a+', encoding="utf-8") as i_file:
                            i_file.writelines(str_1)
                    temp_face = [int(str_line[1])]
                if float(str_line[3]) >= blur_t:
                    temp_face.append(str_line[1:])
            
            print("Processing [%d/%d] video, total %d face in video: %s"%(index_video+1 ,len(videos), len(frame_face_list), i_video))
            if len(frame_face_list)<=0:continue
            video       = cv2.VideoCapture(i_video)
            landmark_log= os.path.join(tg_path_i,"%s_saveface_landmark.json"%basepath)
            landmark_list = {}
            for frame_index in tqdm(range(len(frame_face_list))):
                if self.stop_sign:
                    print("Stop the process!")
                    return

                video.set(cv2.CAP_PROP_POS_FRAMES, frame_face_list[frame_index][0])
                # print("processing frame:", frame_face_list[frame_index][0])
                ret, frame = video.read()
                if  ret:
                    if i_rotat:
                        frame = cv2.rotate(frame, i_rotat)
                    detect_results = self.detect.get(frame, transformer_size= affine_size)
                    if detect_results is not None:
                        for item in frame_face_list[frame_index][1:]:
                            index_face = int(item[1])
                            if index_face >= len(detect_results[0]):
                                break
                            face_i = detect_results[0][index_face]
                            landmark_list["%d"%frame_face_list[frame_index][0]]={
                                "frame_index": frame_face_list[frame_index][0],
                                "face_index":index_face,
                                "landmark":detect_results[3][index_face].tolist()
                            }
                            f_path =os.path.join(tg_path_i, 
                                                 str(frame_face_list[frame_index][0]).zfill(6)+"_%d.%s"%(int(item[1]),
                                                                                                         tg_format))
                            cv2.imencode('.%s'%tg_format,face_i)[1].tofile(f_path)
            with open(landmark_log, 'w') as cf:
                configjson  = json.dumps(landmark_list, indent=4)
                cf.writelines(configjson)
            video.release()    
        print("Face saving finished!")
    
    def ScaneSave(self):
        thread_update = threading.Thread(target=self.scane_save_task)
        thread_update.start()
    
    def scane_save_task(self):
        tg_path     = self.save_path.get()
        if tg_path == "":
            print("Select target path first!")
            return
        dirs = glob.glob(os.path.join(tg_path,"*/"), recursive=False)
        # print(dirs)
        log_save_path = os.path.join(tg_path, "clear_data_logs")
        os.makedirs(log_save_path, exist_ok=True)
        total = len(dirs)
        for index, i_dir in enumerate(dirs):
            if self.stop_sign:
                print("Stop the process!")
                return
            if "clear_data_logs" in i_dir:
                total = total -1
                continue
            txt_log = glob.glob(os.path.join(i_dir,"*_face_log.txt"), recursive=False)
            with open(txt_log[0],'r', encoding="utf-8") as log_reader: # ,encoding="UTF-8"
                logs = log_reader.readline()
            # str_line = logs.split(",")

            print("Processing [%d/%d], dir : %s......"%(index+1, total, i_dir))
            basepath_log= os.path.splitext(os.path.basename(txt_log[0]))[0]
            # tg_path_i   = os.path.join(tg_path,basepath)
            file_log_i  = os.path.join(i_dir,"%s_clear.txt"%basepath_log[:-9])
            images      = glob.glob(os.path.join(i_dir,"*.%s"%"png"), recursive=False)
            if len(images) == 0:
                print("[%d/%d], no image found in %s!"%(index, len(dirs), i_dir))
            images = sorted(images)
            num = 1
            with open(file_log_i,'w', encoding="utf-8") as logf: # ,encoding='UTF-8'
                logf.writelines(logs)
                for i_name in images:
                    if self.stop_sign:
                        print("Stop the process!")
                        return
                    basepath    = os.path.splitext(os.path.basename(i_name))[0]
                    logf.writelines("%d,%s\n"%(num, basepath))
                    num += 1
            shutil.copy(file_log_i, os.path.join(log_save_path,"%s_clear.txt"%basepath_log[:-9]))
        print("Scaning process finished!")


    def Stop(self):
        
        thread_update = threading.Thread(target=self.stop_task)
        thread_update.start()
    def stop_task(self):
        self.stop_sign = True
    def Crop(self):
        thread_update = threading.Thread(target=self.crop_task)
        thread_update.start()

    def crop_task(self):
        self.stop_sign = False
        mode        = self.align_com.get()
        if mode == "VGGFace":
            mode = "None"
        
        
        path        = self.img_path.get()
        tg_path     = self.save_path.get()
        blur_t      = self.thredhold.get()
        frame_interv= self.frame_interv.get()
        frame_interv= int(frame_interv)
        affine_size = self.affine_size_var.get()
        affine_size = int(affine_size)
        snap_size   = int(self.snap_size_var.get())
        suffix_name = self.suffix_name_var.get()
        crop_size   = int(self.test_com.get())

        min_scale   = float(self.min_scale.get())
        blur_t      = float(blur_t)
        print("Blurry thredhold %f"%blur_t)
        self.detect.prepare(ctx_id = 0, det_thresh=0.6,\
                        det_size=(640,640),mode = mode,crop_size=crop_size,ratio=min_scale)
        log_file = "./dataset_readme.txt"
        videos = glob.glob(os.path.join(path,"*.%s"%suffix_name), recursive=False)
        if len(videos) == 0:
            videos = glob.glob(os.path.join(path,"*.%s"%suffix_name.lower()), recursive=False) 
        videos = sorted(videos)
        print("Total %d videos are found!"%len(videos))
        for video_index, i_video in enumerate(videos):
            if self.stop_sign:
                print("Stop the process!")
                return
            basepath    = os.path.splitext(os.path.basename(i_video))[0]
            tg_path_i   = os.path.join(tg_path,basepath)
            file_log_i  = os.path.join(tg_path_i,"%s_face_log.txt"%basepath)

            if os.path.exists(file_log_i):
                print("Video: %s had been processed! Skip it!"%i_video)
                continue

            with open(log_file,'a+', encoding="utf-8") as logf: # ,encoding='UTF-8'
                logf.writelines("%s --> %s\n"%(i_video, tg_path_i))

            # print("target path: ",tg_path_i)
            if not os.path.exists(tg_path_i):
                os.makedirs(tg_path_i)
        
            with open(file_log_i, 'w', encoding="utf-8") as i_file:
                # id, target path, file path, blur thredhold, frame ivterval, min scale, snap size, crop size, align mode
                i_file.writelines("0,%s,%s,%f,%d,%f,%d,%d,%s\n"%(tg_path_i,i_video,blur_t,frame_interv,min_scale,snap_size,crop_size,mode))
            face_str = []
            ok_face  = 1
            print("Processing [%d/%d] video: %s..........."%(video_index+1, len(videos), i_video))
            video = cv2.VideoCapture(i_video)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame_index in tqdm(range(frame_count//frame_interv)):
                if self.stop_sign:
                    print("Stop the process!")
                    return
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_index*frame_interv)
                ret, frame = video.read()
                if  ret:
                    detect_results = self.detect.get_snap_bbox(frame)
                    if detect_results is not None:
                        for index, face_i in enumerate(detect_results):
                            img1    = cv2.cvtColor(face_i,cv2.COLOR_BGR2GRAY)
                            out     = mean_squared_error(img1[2:,:],img1[:-2,:])
                            out1    = mean_squared_error(img1[:,2:],img1[:,:-2])
                            score   = min(out,out1)
                            
                            if score < blur_t:
                                
                                # print("Score is %f"%score)
                                continue
                            # cv2.imwrite("%s-score_%.2f.png"%(frame_index, score), face_i)
                            line_str    = "%d,%d,%d,%.2f\n"%(ok_face,frame_index*frame_interv,index,score)
                            ok_face     += 1
                            face_str.append(line_str)
                            if len(face_str) > 200:
                                with open(file_log_i, 'a+', encoding="utf-8") as i_file:
                                    i_file.writelines(face_str)
                                face_str = []

            video.release()
            with open(file_log_i, 'a+', encoding="utf-8") as i_file:
                i_file.writelines(face_str)
        print("Crop process finished!")
    def on_closing(self):
        self.stop_sign = True
        # self.__save_config__()
        self.master.destroy()
    


if __name__ == "__main__":
    app = Application()
    app.mainloop()