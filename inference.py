# Useful to read about segmentation metrics:
# https://www.kaggle.com/yassinealouini/all-the-segmentation-metrics
# another one:
# https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2


from tensorflow import keras
import tensorflow as tf
import glob

from PIL import Image
import numpy as np

import random

from functions import normalise, parse_image, load_image_test, dice_coeff, bce_dice_loss, create_mask

from evaluation_metrics import __surface_distances, Hausdorff_Distance, Dice_Coefficient

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from sklearn.metrics import jaccard_similarity_score
#from sklearn.metrics import jaccard_score
from scipy.spatial.distance import directed_hausdorff


def create_mask(pred_mask):
  pred_mask = tf.greater(pred_mask, 0.5)
  pred_mask = tf.dtypes.cast(pred_mask, tf.float32)
  pred_mask = pred_mask[0]
  return pred_mask

def Dice_Coefficient(reference, result):
    """
    Computes the Dice coefficient (also known as Sorensen index) between the binary objects in two images.

    result : Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    """
    result = np.array(result, dtype=np.bool)
    result = np.atleast_1d(result)
    result = tf.reshape(result, [-1])

    reference = np.array(reference, dtype=np.bool)
    reference = np.atleast_1d(reference)
    reference = tf.reshape(reference, [-1])
    
    intersection = np.count_nonzero(result & reference)
    
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    
    return dc


model_path = "checkpoints/model.027.hdf5"

VAL_IMG_DIR = "data/images/validation/"
VAL_MASK_DIR = "data/annotations_binary/validation/"
 
TEST_IMG_DIR = "data/images/testing/"
TEST_MASK_DIR = "data/annotations_binary/testing/"

TRAIN_IMG_DIR = "data/images/training/" 
TRAIN_MASK_DIR = "data/annotations_binary/training/"

val_filenames = glob.glob(f"{VAL_IMG_DIR}*.png")
test_filenames = glob.glob(f"{TEST_IMG_DIR}*.png")
train_filenames = glob.glob(f"{TRAIN_IMG_DIR}*.png")

val_filenames = [f.split('/')[3].split('.')[0] for f in val_filenames]
test_filenames = [f.split('/')[3].split('.')[0] for f in test_filenames]
train_filenames = [f.split('/')[3].split('.')[0] for f in train_filenames]

all_filenames = test_filenames + train_filenames

# load trained model
model = keras.models.load_model(model_path, custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coeff': dice_coeff})

# sample = random.choice(test_filenames)

# image_path = Image.open(f"{TEST_IMG_DIR}{sample}.png").convert('L')
    
# mask_path = f"{TEST_MASK_DIR}{sample}.png"

# image = np.asarray(image_path)
# image = (image / 255.)            
# image = np.array(image, dtype = "float") 
# image = image.reshape(-1, 512, 512, 1)

# plt.imshow(image[0])

# mask = tf.io.read_file(mask_path)
# mask = tf.image.decode_png(mask, channels=1)
# mask = tf.image.resize(mask, (512, 512))

# plt.imshow(mask)

# pred_mask = model.predict(image)
# pred_mask = create_mask(pred_mask)
# #plt.imshow(pred_mask) 

# dice = Dice_Coefficient(mask, pred_mask)
    
results = []

for fn in all_filenames:
    
    if fn in train_filenames:
        
        print(f"TRAIN - {fn}")

        image_path = Image.open(f"{TRAIN_IMG_DIR}{fn}.png").convert('L')
        mask_path = f"{TRAIN_MASK_DIR}{fn}.png"
        
    if fn in test_filenames:
        
        print(f"TEST - {fn}")

        image_path = Image.open(f"{TEST_IMG_DIR}{fn}.png").convert('L')
        mask_path = f"{TEST_MASK_DIR}{fn}.png"
    
    image = np.asarray(image_path)
    image = (image / 255.)            
    image = np.array(image, dtype = "float") 
    image = image.reshape(-1, 512, 512, 1)

    #plt.imshow(image[0])

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (512, 512))
    #plt.imshow(mask)
    
    pred_mask = model.predict(image)
    pred_mask = create_mask(pred_mask)
    #plt.imshow(pred_mask)
    
    mask = tf.keras.preprocessing.image.array_to_img(mask)
    mask = np.array(mask)
    pred_mask = tf.keras.preprocessing.image.array_to_img(pred_mask)
    pred_mask = np.array(pred_mask)
    
    dc = Dice_Coefficient(mask, pred_mask)
    #ji = jaccard_similarity_score(mask, pred_mask)
    hd = Hausdorff_Distance(mask, pred_mask)
    
    # IoU calculation
    intersection = np.logical_and(mask, pred_mask)
    union = np.logical_or(mask, pred_mask)
    iou = np.sum(intersection) / np.sum(union)
    #print("IoU is %s" % iou)
    
    results.append([fn, dc, hd, iou])
    

df = pd.DataFrame(results)
df.to_csv('train_test_predictions.csv', index=False, header=None) 

results = []

for fn in val_filenames:
    
    print(f"VAL - {fn}")

    image_path = Image.open(f"{VAL_IMG_DIR}{fn}.png").convert('L')
    mask_path = f"{VAL_MASK_DIR}{fn}.png"
    
    image = np.asarray(image_path)
    image = (image / 255.)            
    image = np.array(image, dtype = "float") 
    image = image.reshape(-1, 512, 512, 1)

    #plt.imshow(image[0])

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (512, 512))
    #plt.imshow(mask)
    
    pred_mask = model.predict(image)
    pred_mask = create_mask(pred_mask)
    #plt.imshow(pred_mask)
    
    mask = tf.keras.preprocessing.image.array_to_img(mask)
    mask = np.array(mask)
    pred_mask = tf.keras.preprocessing.image.array_to_img(pred_mask)
    pred_mask = np.array(pred_mask)
    
    dc = Dice_Coefficient(mask, pred_mask)
    #ji = jaccard_similarity_score(mask, pred_mask)
    hd = Hausdorff_Distance(mask, pred_mask)
    
    intersection = np.logical_and(mask, pred_mask)
    union = np.logical_or(mask, pred_mask)
    iou = np.sum(intersection) / np.sum(union)
    
    results.append([fn, dc, hd, iou])
    

df = pd.DataFrame(results)
df.to_csv('val_predictions.csv', index=False, header=None) 