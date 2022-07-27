import os 
from trees_core import preprocess, old_preprocess
from time import time 

top_dir = '/mnt/c/Users/Research/Documents/GitHub/nasa-africa-trees/data/first_mosaic/rebuilt_approach/input/'

ndvi = []
pan = []
backgrounds = []
annotations = []

for i in os.listdir(top_dir): 
    path = top_dir+i
    if 'ndvi' in i: 
        ndvi.append(path)

    elif 'pan' in i: 
        pan.append(path)

    elif 'vector' in i:
        backgrounds.append(path)

    elif 'annotations' in i: 
        annotations.append(path) 

out_dir = '/mnt/c/Users/Research/Documents/GitHub/nasa-africa-trees/data/first_mosaic/rebuilt_approach/output/'

for i in os.listdir(out_dir):
    os.remove(out_dir+i)

s = time()
preprocess(area_files=backgrounds, annotation_files=annotations, raw_ndvi_images=ndvi, raw_pan_images=pan, output_path=out_dir)
print('New Time:', time()-s)

for i in os.listdir(out_dir):
    os.remove(out_dir+i)

s = time()
old_preprocess(area_files=backgrounds, annotation_files=annotations, raw_ndvi_images=ndvi, raw_pan_images=pan, output_path=out_dir)
print('Old Time:', time()-s)

# python -m cProfile preprocess.py > profile.txt