import re
import os
import argparse
import numpy as np
import pandas as pd
import random
import math
from datetime import datetime
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

%load_ext tensorboard

def get_model(config):
    with strategy.scope():
        margin = ArcMarginProduct(
            n_classes = config.N_CLASSES,
            s = config.scale,
            m = config.margin,
            name='head/arc_margin',
            dtype='float32')

        inp = tf.keras.layers.Input(shape = (*config.IMAGE_SIZE, 3), name = 'inp1')
        label = tf.keras.layers.Input(shape = (), name = 'inp2')
        if config.model == 'effnetb0':
            backbone = efn.EfficientNetB1(weights = 'imagenet', include_top = False)(inp)

        x = tf.keras.layers.GlobalAveragePooling2D()(backbone)
        x = margin([x, label])
        output = tf.keras.layers.Softmax(dtype='float32')(x)

        model = tf.keras.models.Model(inputs = [inp, label], outputs = [output])
        opt = tf.keras.optimizers.Adam(learning_rate = config.LR)

        model.compile(
            optimizer = opt,
            loss = [tf.keras.losses.SparseCategoricalCrossentropy()],
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])

        return model


# Initialize TPUs
def init_tpu():
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
    return strategy


# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Shopee')
    parser.add_argument('--fold-num', type=int, required=True,
            help='validation fold number')
    parser.add_argument('--model', type=str, required=True,
            help='model name')
    parser.add_argument('num_epochs', type=int, default=20,
            help='number of training epochs')
    args = parser.parse_args()

    # Overwrite config
    config = GlobalConfig
    config.EPOCHS = args.num_epochs
    config.model = args.model

    seed_everything(config.SEED)
    strategy = init_tpu()
    config.BATCH_SIZE = 8 * strategy.num_replicas_in_sync

    logger = tf.get_logger().setLevel('INFO')
    logger.info(config.__dict__)

    ##Train Folds
    for fold in range(config.FOLDS):
        logger.info(f'Training with Validation Fold {fold}')
        GCS_PATH = config.GCS_PATH['fold' + fold)] #base GCS path

        # Retrive tfrecords
        TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/*.tfrec')
        NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
        logger.info(f'Dataset: {NUM_TRAINING_IMAGES} training images')

        valid = TRAINING_FILENAMES[fold]
        train = [file for file in TRAINING_FILENAMES if file != valid]
        logger.info(f'Valid Set: {valid}')
        logger.info(f'Train Set: {train}')
        config.N_CLASSES = config.OOF_CLASSES[fold]
        logger.info("Number of Training Classes: {}".format(config.N_CLASSES))

        logger.info('\n')
        logger.info('-'*50)
        train_dataset = get_training_dataset(train, ordered = False)
        train_dataset = train_dataset.map(lambda posting_id, image, label_group, matches: (image, label_group))
        # val_dataset = get_validation_dataset(valid, ordered = True)
        # val_dataset = val_dataset.map(lambda posting_id, image, label_group, matches: (image, label_group))

        STEPS_PER_EPOCH = count_data_items(train) // config.BATCH_SIZE
        K.clear_session()
        model = get_model(config)

        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_cb = keras.callbacks.TensorBoard(log_dir=logdir)

        # Model checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(f'EfficientNetB1_{config.IMAGE_SIZE[0]}_fold{fold}.h5',
                                                        monitor = 'loss',
                                                        verbose = config.VERBOSE,
                                                        save_best_only = True,
                                                        save_weights_only = True,
                                                        mode = 'min')

        history = model.fit(train_dataset,
                            steps_per_epoch = STEPS_PER_EPOCH,
                            epochs = config.EPOCHS,
                            callbacks = [checkpoint, get_lr_callback(config), tensorboard_cb],
                            verbose = config.VERBOSE)

        %tensorboard --logdir logs/scalars
