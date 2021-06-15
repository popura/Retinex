import argparse
import sys
import os
from pathlib import Path

import cv2
import json

import retinex


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    default="./data",
    help="Directory path for searching input images"
)
parser.add_argument(
    "--result_dir",
    type=str,
    default="./result",
    help="Directory path for saving resulting images"
)
parser.add_argument(
    "--config",
    type=str,
    default="./config.json",
    help="Path to config file"
)

args = parser.parse_args()
data_dir = Path.cwd() /args.data_dir
result_dir = Path.cwd() /args.result_dir

with open(args.config, 'r') as f:
    config = json.load(f)

for q in data_dir.glob("*.png"):
    img = cv2.imread(str(q))

    img_msrcr = retinex.MSRCR(
        img,
        config['sigma_list'],
        config['G'],
        config['b'],
        config['alpha'],
        config['beta'],
        config['low_clip'],
        config['high_clip']
    )
   
    img_amsrcr = retinex.automatedMSRCR(
        img,
        config['sigma_list']
    )

    img_msrcp = retinex.MSRCP(
        img,
        config['sigma_list'],
        config['low_clip'],
        config['high_clip']        
    )    

    shape = img.shape
    cv2.imwrite(str(result_dir / 'Image' / f"{q.stem}.png"), img)
    cv2.imwrite(str(result_dir / 'retinex' / f"{q.stem}.png"), img_msrcr)
    cv2.imwrite(str(result_dir / 'Automated retinex' / f"{q.stem}.png"), img_amsrcr)
    cv2.imwrite(str(result_dir / 'MSRCP' / f"{q.stem}.png"), img_msrcp)
