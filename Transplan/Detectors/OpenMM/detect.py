# Some basic setup:
from tabnanny import check
import os
from tqdm import tqdm
import glob
import cv2
# Setup detectron2 logger
import pandas as pd
# from mmdet.apis import init_detector, inference_detector
# import mmcv
from Libs import *
from Utils import *

# choose to run on CPU to GPU


# model_weight_url = ""
# if "model_temp_280758.pkl" not in os.listdir("./Detectors/detectron2/"):
#   os.system(f"wget {model_weight_url} -O ./Detectors/detectron2/model_temp_280758.pkl")
# model_weight_path = "./Detectors/detectron2/model_temp_280758.pkl"

def detect(args,*oargs):
  setup(args)
  env_name = args.Detector
  exec_path = "./Detectors/OpenMM/run.py"
  conda_pyrun(env_name, exec_path, args)

def df(args):
  file_path = args.DetectionDetectorPath
  num_header_lines = 3
  data = {}
  data["fn"], data["class"], data["score"], data["x1"], data["y1"], data["x2"], data["y2"] = [], [], [], [], [], [], []
  with open(file_path, "r+") as f:
    lines = f.readlines()
    for line in lines[num_header_lines::]:
      splits = line.split()
      fn , clss, score, x1, y1, x2, y2 = int(splits[0]), int(splits[1]), float(splits[2]), float(splits[3]), float(splits[4]), float(splits[5]), float(splits[6])
      data["fn"].append(fn)
      data["class"].append(clss)
      data["score"].append(score)
      data["x1"].append(x1)
      data["y1"].append(y1)
      data["x2"].append(x2)
      data["y2"].append(y2)
  return pd.DataFrame.from_dict(data)
    
def setup(args):
    env_name = args.Detector
    src_url = "https://github.com/open-mmlab/mmdetection.git"
    rep_path = "./Detectors/OpenMM/mmdetection"
    checkpoint_name="faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

    print(env_name)
    if not "mmdetection" in os.listdir("./Detectors/OpenMM/"):
      os.system(f"git clone {src_url} {rep_path}")
      if not "checkpoints" in os.listdir("./Detectors/OpenMM/mmdetection"):
        os.system(f"mkdir ./Detectors/OpenMM/mmdetection/checkpoints")
        os.system(f"wget -c https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/{checkpoint_name}\
        -O ./Detectors/OpenMM/mmdetection/checkpoints/{checkpoint_name}")
    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.7")
        # install library on conda env
        print("here I am 1")
        os.system(f"conda install -n {env_name} pytorch==1.6.0 cudatoolkit=10.1 torchvision -c pytorch -y")
        # os.system(f"conda clean --packages --tarballs")
        print("here I am 2")
        os.system(f"conda run -n {args.Detector} pip3 install openmim")
        print("here I am 3")
        os.system(f"conda run -n {args.Detector} mim install mmcv-full")
        print("here I am 4")
        # os.system(f"conda run -n {args.Detector} pip3 install -r ./Detectors/OpenMM/mmdetection/requirements/build.txt")
        # os.system(f"cd ./Detectors/OpenMM/mmdetection")
        # os.system(f"conda run -n {args.Detector} pip3 install  -v -e .  ")
        # os.system(f"conda run -n {args.Detector} pip3 install -e ./Detectors/OpenMM/mmdetection/")
        # print("HERE I AM 01010")
        # os.system(f"conda run -n {args.Detector} python3 Detectors/OpenMM/mmdetection/setup.py develop")
        os.system(f"conda run -n {args.Detector} pip3 install mmdet")
        # print("YOOOOO")
        



  

  # video_path = "./../Dataset/GX010069.avi"
