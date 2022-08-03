import os
from trees_core import train

ndvi_images = []
pan_images = [] 
annotations = []
boundaries = []

output_dir = '/mnt/c/Users/Research/Documents/GitHub/nasa-africa-trees/data/first_mosaic/rebuilt_approach/output/'
for file in os.listdir(output_dir):
    full_path = output_dir+file
    if '.png' in file: 
        if 'annotation' in file: 
            annotations.append(full_path)

        elif 'boundary' in file: 
            boundaries.append(full_path)

        elif 'ndvi' in file: 
            ndvi_images.append(full_path)

        elif 'extracted_pan' in file: 
            pan_images.append(full_path) 

for i in [ndvi_images, pan_images, annotations, boundaries]:
    print(len(i))

train(ndvi_images, pan_images, annotations, boundaries)