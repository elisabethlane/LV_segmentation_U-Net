import numpy as np
import tensorflow as tf
from tensorflow.python.keras import losses
from IPython.display import clear_output
import matplotlib.pyplot as plt
import os
import sys


def parse_image(img_path: str) -> dict:
    """Load an image and its annotation (mask) and returning a dictionary.

    img_path : str : Image (not the mask) location.
    dict: Dictionary mapping an image and its annotation.
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    # For one Image path:
    # .../dataset/images/training/im_00000001.jpg
    # Its corresponding annotation path is:
    # .../dataset/annotations/training/im_00000001.png
    mask_path = tf.strings.regex_replace(img_path, "images", "annotations_binary")
    #mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    mask = tf.io.read_file(mask_path)
    # The masks contain a class index for each pixels
    mask = tf.image.decode_png(mask, channels=1)
    # In scene parsing, "not labeled" = 255
    # But it will mess up with our N_CLASS = 150
    # Since 255 means the 255th class, which doesn't exist
    #mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)
    # Note that we have to convert the new value (0)
    # With the same dtype than the tensor itself

    return {'image': image, 'segmentation_mask': mask}


#The following code performs a simple augmentation of flipping an image. In addition, image is normalized to [0,1]. 
#Finally, as mentioned above the pixels in the segmentation mask are labeled either {1, 2, 3}. 
#For the sake of convenience, let's subtract 1 from the segmentation mask, resulting in labels that are : {0, 1, 2}.

def normalise(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  #input_mask -= 1
  return input_image, input_mask


def load_image_train(datapoint):
  SIZE = 512
  input_image = tf.image.resize(datapoint['image'], (SIZE, SIZE))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (SIZE, SIZE))

  # if tf.random.uniform(()) > 0.5:
  #   input_image = tf.image.flip_left_right(input_image)
  #   input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalise(input_image, input_mask)

  return input_image, input_mask



def load_image_test(datapoint):
  SIZE = 512
  input_image = tf.image.resize(datapoint['image'], (SIZE, SIZE))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (SIZE, SIZE))

  input_image, input_mask = normalise(input_image, input_mask)

  return input_image, input_mask

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()
  
# Defining custom metrics and loss functions

# Defining loss and metric functions are simple with Keras. Simply define a function that takes both the True labels for a 
#given example and the Predicted labels for the same given example.

# Dice loss is a metric that measures overlap. 
#More info on optimizing for Dice coefficient (our dice loss) can be found in the paper, where it was introduced

# We use dice loss here because it performs better at class imbalanced problems by design. 
# In addition, maximizing the dice coefficient and IoU metrics are the actual objectives and goals of our segmentation task. 
#Using cross entropy is more of a proxy which is easier to maximize. Instead, we maximize our objective directly.

def dice_coeff(y_true, y_pred, loss_type='sorensen', smooth=1.):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.
    Parameters
    -----------
    y_true : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    y_pred : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background),
            dice = ```smooth/(small_value + smooth)``,
            then if smooth is very small, dice close to 0 (even the image values lower than the threshold),
            so in this case, higher smooth can have a higher dice.
    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__
    """
    
    # https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
    #numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    #denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    #score = (numerator + 1) / (denominator + 1)
   
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    #y_true_f = tf.layers.flatten(y_true)
    #y_pred_f = tf.layers.flatten(y_pred)
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    
    if loss_type == 'jaccard':
        union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))

    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)

    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)

    score = (2. * intersection + smooth) / (union + smooth)
    return score


def dice_loss(y_true, y_pred):
    #y_pred = tf.dtypes.cast(y_pred, tf.int64)
    #y_pred = tf.argmax(y_pred, axis=-1)
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    #y_pred = tf.argmax(y_pred, axis=-1)
    #y_pred = tf.dtypes.cast(y_pred, tf.int64)
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def create_mask(pred_mask):
  #print(pred_mask.shape)
  #pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = tf.greater(pred_mask, 0.5)
  pred_mask = tf.dtypes.cast(pred_mask, tf.float32)
  #pred_mask = pred_mask[..., tf.newaxis]
  pred_mask = pred_mask[0]
  #print(pred_mask.shape)
  return pred_mask
 