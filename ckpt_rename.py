import glob
import cv2
import os
import shutil
import sys
import numpy as np

target_dir = sys.argv[1]
output_dir = sys.argv[2]

#shutil.copytree(target_dir, output_dir, dirs_exist_ok=True)
os.system(f"cp -r {target_dir} {output_dir}")

for d in glob.glob(os.path.join(output_dir, '*/*/csv_log')):
    os.system(f'rm -r {d}')
for d in glob.glob(os.path.join(output_dir, '*/*/tb_log')):
    os.system(f'rm -r {d}')
for d in glob.glob(os.path.join(output_dir, '*/*/ckpt/last.ckpt')):
    os.system(f'rm -r {d}')


for d in glob.glob(os.path.join(output_dir, '*/*/ckpt/*.ckpt')):
    os.rename(d, d.replace('=', ''))
