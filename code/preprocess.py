import os 

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

from trees_core import preprocess

out_dir = '/mnt/c/Users/Research/Documents/GitHub/nasa-africa-trees/data/first_mosaic/rebuilt_approach/output'
preprocess(area_files=backgrounds, annotation_files=annotations, raw_ndvi_images=ndvi, raw_pan_images=pan, output_path=out_dir, verbose=True)