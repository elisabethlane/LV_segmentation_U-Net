import tensorflow as tf
from tensorflow.keras.models import Model
from tf.keras.losses import BinaryCrossentropy
from tf.keras.layers import (Conv2D, BatchNormalization, Activation, 
                             MaxPooling2D, Conv2DTranspose, concatenate,
                             Input)

def conv_block(input_tensor, num_filters):
  encoder = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
  encoder = BatchNormalization()(encoder)
  encoder = Activation('relu')(encoder)
  encoder = Conv2D(num_filters, (3, 3), padding='same')(encoder)
  encoder = BatchNormalization()(encoder)
  encoder = Activation('relu')(encoder)
  return encoder

def encoder_block(input_tensor, num_filters):
  encoder = conv_block(input_tensor, num_filters)
  encoder_pool = MaxPooling2D((2, 2), strides=(2, 2))(encoder)
  
  return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
  decoder = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
  decoder = concatenate([concat_tensor, decoder], axis=-1)
  decoder = BatchNormalization()(decoder)
  decoder = Activation('relu')(decoder)
  decoder = Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = BatchNormalization()(decoder)
  decoder = Activation('relu')(decoder)
  decoder = Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = BatchNormalization()(decoder)
  decoder = Activation('relu')(decoder)
  return decoder

def UNet(input_shape_image, output_channels):
    
    inputs = Input(shape=input_shape_image)
    
    encoder0_pool, encoder0 = encoder_block(inputs, 32)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)
    
    center = conv_block(encoder4_pool, 1024)
    
    decoder4 = decoder_block(center, encoder4, 512)
    decoder3 = decoder_block(decoder4, encoder3, 256)
    decoder2 = decoder_block(decoder3, encoder2, 128)
    decoder1 = decoder_block(decoder2, encoder1, 64)
    decoder0 = decoder_block(decoder1, encoder0, 32)
    
    outputs = Conv2D(output_channels, (1, 1), activation='sigmoid')(decoder0)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    model.compile(optimizer='adam', loss=BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    
    return model