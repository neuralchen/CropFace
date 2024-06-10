import os
import cv2
import requests
import json
from tqdm import tqdm
from glob import glob
import base64

def file_base64(file_name):
    with open(file_name,'rb') as fin:
        file_data=fin.read()
        base64_data=base64.b64encode(file_data)
    return base64_data
base_path = "D:/hd_face_dataset/av/midv-214-4k/"

imgs = glob(os.path.join(base_path, "*.png"))
url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
total_result = {}
tq_bar = tqdm(imgs)
for i_img in tq_bar:
    
    id_basename = os.path.splitext(os.path.basename(i_img))[0]
    tq_bar.set_description("Processing %s"%id_basename)
    payload = {'api_key': 'NkMJGKamSurd7tGdLnt9OxyDlWwgFXVU',
            'api_secret': 'HD0DCGrEgyJalMj0N5WqCgP8yAkkCysg',
            #'image_url': 'https://image.so.com/view?q=%E5%8D%83%E7%8E%BA&listsrc=sobox&listsign=c8f2dd5542c5c82c1e1116ad3425cce3&src=360pic_normal&correct=%E5%8D%83%E7%8E%BA&ancestor=list&cmsid=7cd680ddc9676663d354b03c861bf133&cmran=0&cmras=6&cn=0&gn=0&kn=50&fsn=130&adstar=0&clw=262#id=e8bfc493703e8be5669f2c133326a43f&currsn=0&ps=121&pc=121',
            'return_landmark': 2,
            #'image_file':files,
            'image_base64': file_base64(i_img),
            'return_attributes': "gender,age,eyestatus,facequality,headpose,blur"
            }
    r = requests.post(url, data=payload)
    data = json.loads(r.text)
    total_result[id_basename] = data
# %%
# print request content,you can also use r.+tab to see more things.

# img = cv2.imread(img_name)
# vis = img.copy()
# draw face rectangle
# cv2.rectangle(vis, (left, top), (left+width, top+height),(0, 255, 0), 1)

with open(os.path.join(base_path,"faceplusplus_detected_results.json"), 'w') as cf:
    configjson  = json.dumps(total_result, indent=4)
    cf.writelines(configjson)

# # %%
# # draw face landmarks
# for j in (0, len(data['faces']) - 1):
#     for i in data['faces'][j]['landmark']:
#         cor = data['faces'][j]['landmark'][i]
#         x = cor["x"]
#         y = cor["y"]
#         cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
# # %%
# # save image with landmarks
# cv2.imwrite("%s.png"%id_basename, vis)