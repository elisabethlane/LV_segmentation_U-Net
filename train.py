import tensorflow as tf

import matplotlib.pyplot as plt

from model import UNet

from functions import normalise, parse_image, load_image_test, load_image_train, dice_coeff, bce_dice_loss, create_mask


EPOCHS = 1000
OUTPUT_CHANNELS = 1
input_shape_image = [512, 512, 1]
BATCH_SIZE = 16
BUFFER_SIZE = 1000

train_imgs = tf.data.Dataset.list_files("data/images/training/*.png")
test_imgs = tf.data.Dataset.list_files("data/images/testing/*.png")

train_set = train_imgs.map(parse_image)
test_set = test_imgs.map(parse_image)

dataset = {"train": train_set , "test": test_set}

num_training_examples = 0
num_test_examples = 0

for example in train_imgs:
    num_training_examples += 1

for example in test_imgs:
    num_test_examples += 1

print('num_training_examples = ', num_training_examples)
print('num_test_examples = ', num_test_examples)

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset  = test.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

model = UNet(input_shape_image, OUTPUT_CHANNELS)

filepath = 'checkpoints/unetModel.{epoch:03d}.hdf5'
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = filepath,
                               monitor='val_loss', 
                               verbose=2,
                               save_best_only=True,
                               save_weights_only=False, 
                               mode='auto')
   
early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=30, 
                          restore_best_weights=True)

TRAIN_LENGTH = num_training_examples
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VAL_SUBSPLITS = 5
VALIDATION_STEPS = num_test_examples //BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          verbose=1,
                          validation_data=test_dataset,
                          validation_steps=VALIDATION_STEPS,
                          callbacks=[checkpointer, early_stopper])



  


