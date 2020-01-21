# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 16:09:33 2019

@author: Lenovo

Deeplabv3+ model for Keras.
This model is based on TF repo:
https://github.com/tensorflow/models/tree/master/research/deeplab
On Pascal VOC, original model gets to 84.56% mIOU
Now this model is only available for the TensorFlow backend,
due to its reliance on `SeparableConvolution` layers, but Theano will add
this layer soon.
MobileNetv2 backbone is based on this repo:
https://github.com/JonathanCMitchell/mobilenet_v2_keras
# Reference
- [Encoder-Decoder with Atrous Separable Convolution
    for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
- [Xception: Deep Learning with Depthwise Separable Convolutions]
    (https://arxiv.org/abs/1610.02357)
- [Inverted Residuals and Linear Bottlenecks: Mobile Networks for
    Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)

Modified: 
    22.09.19: creation of myv1, added the 2 skip blocks, this model has 3 skips
    05.10.19: Version 3, modifying the generator function as per the standard template
                
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras.models import Model#, load_model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Add  #, Reshape
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras import layers

from keras.layers import Conv2D  #, Conv2DTranspose

from keras.activations import relu

from keras.layers import DepthwiseConv2D#, UpSampling2D
from keras.layers import ZeroPadding2D, Lambda
from keras.layers import AveragePooling2D
# from keras.engine import Layer, InputSpec
from keras.engine.topology import get_source_inputs
from keras import backend as K

# from keras.applications import imagenet_utils
# from keras.utils import conv_utils, print_summary, plot_model
from keras.utils.data_utils import get_file
# import tensorflow as tf

#Ak imports
from keras.optimizers import adam
import datetime
from matplotlib import pyplot as plt
# from scratchModelv02UNET0915EncoderDecoder import preprocess_data
import tifffile as tff
import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "20"

WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"



def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False, link=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand
        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        #x = Lambda(lambda x: relu(x, max_value=6.))(x)
        x = Lambda(lambda x: relu(x, max_value=6.), name=prefix + 'expand_relu')(x)
        #x = Activation(relu(x, max_value=6.), name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)
    #x = Activation(relu(x, max_value=6.), name=prefix + 'depthwise_relu')(x)
    x = Lambda(lambda x: relu(x, max_value=6.), name=prefix + 'depthwise_relu')(x)

    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


def Deeplabv3(weights='pascal_voc', input_tensor=None, infer = False,
              input_shape=(183, 183, 3), classes=1, backbone='mobilenetv2',
              OS=16, alpha=1.):
    
    """ Instantiates the Deeplabv3+ architecture
    Optionally loads weights pre-trained
    on PASCAL VOC. This model is available for TensorFlow only,
    and can only be used with inputs following the TensorFlow
    data format `(width, height, channels)`.
    # Arguments
        weights: one of 'pascal_voc' (pre-trained on pascal voc)
            or None (random initialization)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images
        classes: number of desired classes. If classes != 21,
            last layer is initialized randomly
        backbone: backbone to use. one of {'xception','mobilenetv2'}
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
            Used only for xception backbone.
        alpha: controls the width of the MobileNetV2 network. This is known as the
            width multiplier in the MobileNetV2 paper.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                    are used at each layer.
            Used only for mobilenetv2 backbone
    # Returns
        A Keras model instance.
    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`
    """
        
    if not (weights in {'pascal_voc', None}):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `pascal_voc` '
                         '(pre-trained on PASCAL VOC)')

    if K.backend() != 'tensorflow':
        raise RuntimeError('The Deeplabv3+ model is only available with '
                           'the TensorFlow backend.')

    if not (backbone in {'xception', 'mobilenetv2'}):
        raise ValueError('The `backbone` argument should be either '
                         '`xception`  or `mobilenetv2` ')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    
    
    batches_input = Lambda(lambda x: x/127.5 - 1)(img_input)

    if backbone == 'xception':
        if OS == 8:
            entry_block3_stride = 1
            middle_block_rate = 2  # ! Not mentioned in paper, but required
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        else:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)
        x1 = Conv2D(32, (3, 3), strides=(2, 2),
                   name='entry_flow_conv1_1', use_bias=False, padding='same')(batches_input)
            
        x = BatchNormalization(name='entry_flow_conv1_1_BN')(x1)
        x = Activation('relu')(x)

        x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
        x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
        x = Activation('relu')(x)

        x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                            skip_connection_type='conv', stride=2,
                            depth_activation=False)
        x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                                   skip_connection_type='conv', stride=2,
                                   depth_activation=False, return_skip=True, link=True)

        x, skip0 = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                            skip_connection_type='conv', stride=entry_block3_stride,
                            depth_activation=False, return_skip=True)
        for i in range(16):
            x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                                skip_connection_type='sum', stride=1, rate=middle_block_rate,
                                depth_activation=False)

        x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                            skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                            depth_activation=False)
        x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                            skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                            depth_activation=True)

    else:
        OS = 8
        first_block_filters = _make_divisible(32 * alpha, 8)
        x = Conv2D(first_block_filters,
                   kernel_size=3,
                   strides=(2, 2), padding='same',
                   use_bias=False, name='Conv')(batches_input)
        x = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
        
        x = Lambda(lambda x: relu(x, max_value=6.))(x)

        x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                                expansion=1, block_id=0, skip_connection=False)

        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                                expansion=6, block_id=1, skip_connection=False)
        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                                expansion=6, block_id=2, skip_connection=True)

        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                                expansion=6, block_id=3, skip_connection=False)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=4, skip_connection=True)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=5, skip_connection=True)

        # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                                expansion=6, block_id=6, skip_connection=False)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=7, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=8, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=9, skip_connection=True)

        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=10, skip_connection=False)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=11, skip_connection=True)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=12, skip_connection=True)

        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                                expansion=6, block_id=13, skip_connection=False)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=14, skip_connection=True)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=15, skip_connection=True)

        x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=16, skip_connection=False)

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    #out_shape = int(np.ceil(input_shape[0] / OS))
    b4 = AveragePooling2D(pool_size=(int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(x)
        
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    
    b4 = Lambda(lambda x: K.tf.image.resize_bilinear(x,size=(int(np.ceil(input_shape[0]/OS)), int(np.ceil(input_shape[1]/OS)))))(b4)

    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2. not sure why
    if backbone == 'xception':
        # rate = 6 (12)
        b1 = SepConv_BN(x, 256, 'aspp1',
                        rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
        # rate = 12 (24)
        b2 = SepConv_BN(x, 256, 'aspp2',
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # rate = 18 (36)
        b3 = SepConv_BN(x, 256, 'aspp3',
                        rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # concatenate ASPP branches & project
        x = Concatenate()([b4, b0, b1, b2, b3])
    else:
        x = Concatenate()([b4, b0])
        
    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
###############################################################################
    # DeepLab v.3+ decoder

    if backbone == 'xception':
        # Feature projection
        # x4 (x2) block

#        #Ak this is a whole new block of linking at dim level (23,23) 22.09
#        x = Lambda(lambda x: K.tf.image.resize_bilinear(x,size=(int(np.ceil(input_shape[0]/8)), int(np.ceil(input_shape[1]/8)))))(x)
#        dec_skip0 = Conv2D(48, (1, 1), padding='same',
#                           use_bias=False, name='feature_projection0_ak')(skip0) #scope of experiment 128/48
#        dec_skip0 = BatchNormalization(
#            name='feature_projection0_BN_ak', epsilon=1e-5)(dec_skip0)
#        dec_skip0 = Activation('relu')(dec_skip0)
#        x = Concatenate(name='concatenation_23_ak')([x, dec_skip0])
#        x = Conv2D(256, (1, 1), padding='same',
#                   use_bias=False, name='concat_projection_ak')(x)
#        x = BatchNormalization(name='concat_projection_BN_ak1', epsilon=1e-5)(x)
#        x = Activation('relu')(x)
                
        x = Lambda(lambda x: K.tf.image.resize_bilinear(x,size=(int(np.ceil(input_shape[0]/4)), int(np.ceil(input_shape[1]/4)))))(x)
        
        dec_skip1 = Conv2D(48, (1, 1), padding='same',
                           use_bias=False, name='feature_projection0')(skip1)
        dec_skip1 = BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = Activation('relu')(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, 'decoder_conv0',
                       depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 256, 'decoder_conv1',
                       depth_activation=True, epsilon=1e-5)

# you can use it with arbitary number of classes
    if classes == 21:
        last_layer_name = 'logits_semantic'
    else:
        last_layer_name = 'custom_logits_semantic1'
    
   # Ak new upscaling lambda layer with shape = input_shape/2 to make dimension 92,92,32
#    x = Lambda(lambda x: K.tf.image.resize_bilinear(x,size=(int(np.ceil(input_shape[0]/2)),
    # int(np.ceil(input_shape[1]/2)))), name='ak_additional_lambda_3a')(x)

#    x = Concatenate(axis=3, name='ak_concatenate_92_32')([x, x1])    
#    # Ak new addition 01.10
#    x = Conv2D(48, (1, 1), padding='same', name='ak'+last_layer_name)(x)
#    x = BatchNormalization(name='concat_projection_BNak2', epsilon=1e-5)(x)
#
    x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
#    x = BatchNormalization(name='concat_projection_BNak3', epsilon=1e-5)(x)
    x = Lambda(lambda x: K.tf.image.resize_bilinear(x,size=(input_shape[0],input_shape[1])))(x)
    
    if infer:
        x = Activation('sigmoid', name='the_last_activation1')(x)
    else:
#        x = Reshape((input_shape[0]*input_shape[1], classes)) (x)
        x = Activation('sigmoid', name='the_last_activation2')(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input  

    model = Model(inputs, x, name='deeplabv3p')
 
    # load weights

    if weights == 'pascal_voc':
        if backbone == 'xception':
            weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_X,
                                    cache_subdir='models')
        else:
            weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_MOBILE,
                                    cache_subdir='models')
        model.load_weights(weights_path, by_name=True)
    print(model.summary)
    return model

def generate_prediction_data(path):
    """
    same as generate_arrays_from_file() but this function is for the prediction data
    and will process only one pair of X-y image files
    created on: 24.09.19
    """
    prefix = 'G:\\Merged-Sat-Imgs-Tiles\\final-merged-183by183\\'
    with open(path) as f:
        for line in f:
            try:
                # create numpy arrays of input data
                # and labels, from each line in the file
                flist = line.split()
                Xfilename, yfilename = flist[0],flist[1]
                Ximg = np.array([tff.imread(prefix+Xfilename)])
                yimg = np.array([tff.imread(prefix+yfilename)])
                # scaling the data
                elevMin = Ximg[:,:,:,1].min()
                elevMax = Ximg[:,:,:,1].max()
                Ximg[:,:,:,1] = (Ximg[:,:,:,1] - elevMin) / (elevMax - elevMin)

                tMin = Ximg[:,:,:,0].min()
                tMax = Ximg[:,:,:,0].max()
                Ximg[:,:,:,0] = (Ximg[:,:,:,0] - tMin) / (tMax - tMin)

                return (Ximg, yimg[:,:,:,2:3])
            except Exception as e :
                print("Error in generate_prediction_data():",str(e))

        
def generate_arrays_from_file(path):
    """
    """
    prefix = 'G:\\Merged-Sat-Imgs-Tiles\\final-merged-183by183\\'
    while True:
        with open(path) as f:
            for line in f:
                try:
                    # create numpy arrays of input data
                    # and labels, from each line in the file
                    flist = line.split()
                    Xfilename, yfilename = flist[0],flist[1]
                    Ximg = np.array([tff.imread(prefix+Xfilename)])
                    yimg = np.array([tff.imread(prefix+yfilename)])
                    # check for NaN; if NaN found in either X or y then 
                    if np.isnan(Ximg).any() or np.isnan(yimg).any():
                        pass
                    else:
                        # scaling the data
                        elevMin = Ximg[:,:,:,1].min()
                        elevMax = Ximg[:,:,:,1].max()
                        Ximg[:,:,:,1] = (Ximg[:,:,:,1] - elevMin) / (elevMax - elevMin)
        
                        tMin = Ximg[:,:,:,0].min()
                        tMax = Ximg[:,:,:,0].max()
                        Ximg[:,:,:,0] = (Ximg[:,:,:,0] - tMin) / (tMax - tMin)
        
                        yield ({'input_1': Ximg}, {'the_last_activation2': yimg[:,:,:,2:3]})
                except Exception as e :
                    print("Error in generate_arrays_from_file():",str(e))
# a,b=generate_arrays_from_file('G:\\Merged-Sat-Imgs-Tiles\\img-xy-list-Copy.txt')

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, prefix, batch_size, dim=(183,183), n_channels=3, n_classes=10, shuffle=False):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels # a dictionary
        self.list_IDs = list_IDs # a dictionary
        self.prefix = prefix
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        # print("In len",int(np.floor(len(self.list_IDs) / self.batch_size)))
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        try:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#            print("Index:",index,type(index))
#            print("Indexes:",indexes,type(indexes))
            list_IDs_temp = [self.list_IDs[k] for k in indexes]
            # print(list_IDs_temp)
            
            X, y = self.__data_generation(list_IDs_temp)
#            print("shape of y", type(y))
#            print(X[0][0][:5])
#            print(y[0][0][:5])
            return X, y
        except Exception as e:
            print("error in getitem",str(e))
    
    def on_epoch_end(self):
        try:
            self.indexes = np.arange(len(self.list_IDs))
            if self.shuffle==True:
                np.random.shuffle(self.indexes)
        except Exception as e:
            print("Error in on_epoch_end", str(e))
            
    def __data_generation(self, list_IDs_temp):
        try:
            X=np.empty((self.batch_size, *self.dim, self.n_channels))
            y=np.empty((self.batch_size, *self.dim,1), dtype=float)
#            X=np.zeros((1,183,183,3))
#            y=np.zeros((1,183,183,1))
            #generating data
            cnt=0
            for i, ID in enumerate(list_IDs_temp):
                # here ID will be the actuall tile file name
                Xtemp=np.array(tff.imread(self.prefix+ID))
                elevMin = Xtemp[:,:,1].min()
                elevMax = Xtemp[:,:,1].max()
                Xtemp[:,:,1] = (Xtemp[:,:,1] - elevMin) / (elevMax - elevMin)

                tMin = Xtemp[:,:,0].min()
                tMax = Xtemp[:,:,0].max()
                Xtemp[:,:,0] = (Xtemp[:,:,0] - tMin) / (tMax - tMin)

                X[i]=Xtemp
                y[i]=np.array(tff.imread(self.prefix+labels[ID])[:,:,2:3]) #this 2:3 is done only to match the dimensions of y here and in model final layer
                cnt+=1
            # print("returning tensor of lenth",cnt)
            return X, y
        except Exception as e:
            print("Error in data generation", str(e))

try:
    print("calling main block")
    # to change tiles# inputed to training needs to be adjusted in script scratchModelv02UNET0915EncoderDecoder.py
#    X,y,tiles,dates=preprocess_data()
    mod = Deeplabv3(OS=16, backbone='xception')
#    plot_model(mod, to_file='E:\\4_Prakalpa-Nirmiti\\PythonCode\\DeepLabv3p-models\\model.png')
#    print_summary(mod,line_length=200) #this works well
    # writing the model summary to file
    stringlist = []
    mod.summary(print_fn=lambda x: stringlist.append(x),line_length=200)
    short_model_summary = "\n".join(stringlist)
    fmodelsum = open('E:\\4_Prakalpa-Nirmiti\\PythonCode\\DeepLabv3p-models\\DeepLabv3p-2skip-summary.txt','w+')
    fmodelsum.write(str(short_model_summary))
    fmodelsum.close()
    mod.compile(loss='mean_squared_error', optimizer=adam(0.001, decay=1e-6), metrics=['mean_absolute_error'])
    print("compiled...")

    modelPath = 'DeepLabv3p-models\\'
    modelName = 'MyDeepLabv3pp-newDG.h5'
    path = 'G:\\Merged-Sat-Imgs-Tiles\\'

    if not os.path.exists(modelPath+modelName):
#################################TRAINING######################################
        myEpoch = 2
        myValSplit = 0.1
        myBatchSize = 16
        myValBatchSize = 8
        myStepPerEpoch = 32
        myValStepPerEpoch = 1
        st = datetime.datetime.now()
        
        # create dictionaries partition and labels (the values will be img tile names)
        partition = {}
        labels = {}
        trainFile = 'img-xy-list-Copy.txt'
        partition['train'] = []
    
        with open(path+trainFile) as f:
            for line in f:
                fList = line.split()
                partition['train'].append(fList[0])
                labels[fList[0]] = fList[1]
                    
        valFile = 'img-xy-list-validation-Copy.txt'
        partition['validation'] = []
        with open(path+valFile) as f:
            for line in f:
                fList = line.split()
                partition['validation'].append(fList[0])
                labels[fList[0]] = fList[1]
                                        
        prefix = path+'final-merged-183by183\\'
        print("Batch size:", myBatchSize)
        print("Val Batch size:", myValBatchSize)
        training_generator = DataGenerator(partition['train'], labels, prefix,  myBatchSize)
        validation_generator = DataGenerator(partition['validation'], labels, prefix, myValBatchSize)
        
        try:        
            hist = mod.fit_generator(generator=training_generator,
                                     validation_data=validation_generator,
                                     # validation_steps=round(len(partition['validation'])/myValStepPerEpoch),
                                     #  steps_per_epoch=myStepPerEpoch,
                                     max_queue_size=2,
                                     epochs=myEpoch,
                                     verbose=2)
        except Exception as e:
            print("Catching fit_generator specific exception", str(e))
        et = datetime.datetime.now()
        print("Training time:", str(et-st))
        print("model trained.")
        
###############################SAVE-MODEL######################################
        mod.save(modelPath+modelName)
        print("model saved...")
        
#############################PLOTTING LOSS/METRICS############################
        # fig, ((ax_loss, ax_mae, blank), (ip_img, exp_img, pred_img)) = plt.subplots(2, 3, figsize=(15,15))
        fig, (ax_loss, ax_mae) = plt.subplots(2, 1, figsize=(10,10))
        ax_loss.plot(hist.epoch, hist.history["loss"], label="Train loss")
        ax_loss.plot(hist.epoch, hist.history["val_loss"], label="val Train loss")
        ax_loss.set_title("Loss per epoch")
        ax_mae.plot(hist.epoch, hist.history["mean_absolute_error"], label="mean_absolute_error")
        ax_mae.plot(hist.epoch, hist.history["val_mean_absolute_error"], label="val mean_absolute_error")
        ax_mae.set_title("MAE per epoch")
    
###############################LOAD-MODEL######################################
    else:
        # mod = load_model(modelPath+modelName)
        print("model loaded")
    
##################################predict######################################
    # Using func generate_prediction_data() to fetch prediction X-y instead of deriving from generator
    try:
        predXy = generate_prediction_data(path+'img-xy-list-prediction.txt')
        predX, predy = predXy[0],predXy[1]
        st=datetime.datetime.now()
        predImg=mod.predict(predX, batch_size=None, steps=1)
        et=datetime.datetime.now()
        print("prediction time:",str(et-st))
        print("expected",str(predy[0][0][:10]))
        print("prediction",str(predImg[0][0][:10]),predImg.shape)
        yimg=predy[0,:,:,0]
        print(yimg.shape,type(yimg))
        print(predImg[0,:,:,0].shape,type(predImg))

    except Exception as e:
        print("Error in prediction block:",str(e))

    f, (ei, pi) = plt.subplots(1,2, figsize=(15,5))
    ei.imshow(yimg)
    ei.set_title("expected img")
    #    exp_img.savefig('\\scratch-model-results\\expimg.png', bbox_inches="tight")
    pi.imshow(predImg[0,:,:,0])
    pi.set_title("predicted img")
    print("done")

except Exception as e:
    print("Error in main block:",str(e))