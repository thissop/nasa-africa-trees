import os 

times = []

dir = '/mnt/c/Users/Research/Documents/GitHub/africa-trees/data/first_mosaic/annotations/ready/output/'

for f in os.listdir(dir):
    times.append(os.path.getmtime(dir+f))

print(max(times)-min(times))