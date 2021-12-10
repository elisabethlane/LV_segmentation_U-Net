import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import cv2
import numpy as np
from scipy.interpolate import interp1d
import numpy as np
from scipy.interpolate import splprep, splev
import pandas as pd
import os

  
# CREATE AND SAVE MASKS AND IMAGES

img_dir = 'data/images/'
mask_dir = 'data/masks/'

with open('annotations.json') as json_file:
    data = json.load(json_file)
    
for file in data.items():
    
    fn = file[0]
    
    X = file[1][0][1]
    Y = file[1][0][2]
    
    X = np.array(X)
    Y = np.array(Y)    
    
    if len(X) > 3 and len(Y) > 3:
    
        pts = np.vstack((X, Y))
        tck, u = splprep(pts, s=0.0)
        u_new = np.linspace(u.min(), u.max(), 1000)
    
        X, Y = splev(u_new, tck)
        
        image = cv2.imread(f"PATH/{fn}", 0)
        
        coords = []
    
        for x, y in zip(X, Y):
            coords.append([int(x), int(y)])
            
        coords = np.array(coords)
        
        mask = np.zeros((image.shape[0], image.shape[1]))
    
        cv2.fillConvexPoly(mask, coords, 1)
        mask = mask.astype(bool)
        mask = mask*1
        
        cv2.imwrite(f'{mask_dir}{fn}', mask)
        cv2.imwrite(f'{img_dir}{fn}', image)
    
    
# # VISUALISE AND CHECK MASKS AND IMAGES
# # TEST SCRIPT FOR ONE RANDOM DATA SAMPLE

# with open('annotations.json') as json_file:
#     data = json.load(json_file)
    
# file = random.choice(list(data.items()))
    
# fn = file[0]

# X = file[1][0][1]
# Y = file[1][0][2]

# img_dir = 'data/images/'
# mask_dir = 'data/masks/'

# #file = random.choice(list(data.items()))
    
# # fn = file[0]
# # X = file[1][0][1]
# # Y = file[1][0][2]

# image = cv2.imread(f"{img_dir}{fn}", 0)
    
# mask = cv2.imread(f"{mask_dir}{fn}", 0)

# fig, axs = plt.subplots(1, 1, figsize=(100, 50), sharey=False)
# axs.imshow(image)
# axs.imshow(mask, alpha=0.5)
# for x, y in zip(X, Y):
#     circ = patches.Circle((x, y),5,color='red')
#     axs.add_patch(circ)

        
    

# # VISUALISE MASK FOR SPECIFIC IMAGE

# with open('all_annotations.json') as json_file:
#     data = json.load(json_file)
    
# #files = pd.read_csv('check_masks.csv').values.tolist()
    
# #file = list(data['01-0b11ea2b2726adf79a825e76c44e0b3a68b6f878d963de0b976c38ccbbdaeb91-0065.png'])
# t = "val"

# fn = '' # filename

# # X = file[0][1]
# # Y = file[0][2]
    
# # fn = x[0]

# X = data[fn][0][1]
# Y = data[fn][0][2]

# if os.path.isfile(f"data_use/images/test/{fn}"):

#     image = cv2.imread(f"data_use/images/test/{fn}", 0)
#     mask = cv2.imread(f"data_use/masks/test/{fn}", 0)
#     print("test")
    
# else:
    
#     image = cv2.imread(f"data_use/images/train/{fn}", 0)
#     mask = cv2.imread(f"data_use/masks/train/{fn}", 0)
#     print("train")



# fig, axs = plt.subplots(1, 1, figsize=(100, 50), sharey=False)
# axs.set_title(f"{fn}")
# axs.imshow(image)
# axs.imshow(mask, alpha=0.5)
# for x, y in zip(X, Y):
#     circ = patches.Circle((x, y),5,color='red')
#     axs.add_patch(circ)


    
    




