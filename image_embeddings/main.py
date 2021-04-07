import re
import os
import numpy as np
import pandas as pd
import random
import math
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
import efficientnet.tfkeras as efn

from config import GlobalConfig
from data import *
from common import *
from loss import ArcMarginProduct

def get_model():
    with strategy.scope():
        margin = ArcMarginProduct(
            n_classes = N_CLASSES,
            s = 30,
            m = 0.5,
            name='head/arc_margin',
            dtype='float32')

        inp = tf.keras.layers.Input(shape = (*IMAGE_SIZE, 3), name = 'inp1')
        label = tf.keras.layers.Input(shape = (), name = 'inp2')
        x = efn.EfficientNetB3(weights = 'imagenet', include_top = False)(inp)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = margin([x, label])

        output = tf.keras.layers.Softmax(dtype='float32')(x)
        model = tf.keras.models.Model(inputs = [inp, label], outputs = [output])
        opt = tf.keras.optimizers.Adam(learning_rate = LR)

        model.compile(
            optimizer = opt,
            loss = [tf.keras.losses.SparseCategoricalCrossentropy()],
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])

        return model



if __name__ == '__main__':
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()

    print("REPLICAS: ", strategy.num_replicas_in_sync)


    config = GlobalConfig
    seed_everything(config.SEED)
    config.BATCH_SIZE = 8 * strategy.num_replicas_in_sync

    NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
    print(f'Dataset: {NUM_TRAINING_IMAGES} training images')

    print('\n')
    print('-'*50)
    train, valid = train_test_split(config.TRAINING_FILENAMES, shuffle = True, random_state = config.SEED)
    train_dataset = get_training_dataset(train, ordered = False)
    train_dataset = train_dataset.map(lambda posting_id, image, label_group, matches: (image, label_group))
    val_dataset = get_validation_dataset(valid, ordered = True)
    val_dataset = val_dataset.map(lambda posting_id, image, label_group, matches: (image, label_group))

    STEPS_PER_EPOCH = count_data_items(train) // config.BATCH_SIZE
    K.clear_session()
    model = get_model()
    
    # Model checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'EfficientNetB3_{config.IMAGE_SIZE[0]}_{config.SEED}.h5',
                                                    monitor = 'val_loss',
                                                    verbose =config.VERBOSE,
                                                    save_best_only = True,
                                                    save_weights_only = True,
                                                    mode = 'min')

    history = model.fit(train_dataset,
                        steps_per_epoch = STEPS_PER_EPOCH,
                        epochs = config.EPOCHS,
                        callbacks = [checkpoint, get_lr_callback()],
                        validation_data = val_dataset,
                        verbose = config.VERBOSE)
