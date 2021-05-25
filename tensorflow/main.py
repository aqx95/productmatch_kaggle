import re
import os
import argparse
import numpy as np
import pandas as pd
import random
import math
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
import efficientnet.tfkeras as efn
from classification_models.keras import Classifiers

from config import GlobalConfig
from data import *
from common import *
from loss import ArcMarginProduct


class GeMPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, p=1., train_p=False):
        super().__init__()
        if train_p:
            self.p = tf.Variable(p, dtype=tf.float32)
        else:
            self.p = p
        self.eps = 1e-6

    def call(self, inputs: tf.Tensor, **kwargs):
        inputs = tf.clip_by_value(inputs, clip_value_min=1e-6, clip_value_max=tf.reduce_max(inputs))
        inputs = tf.pow(inputs, self.p)
        inputs = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        inputs = tf.pow(inputs, 1./self.p)
        return inputs


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

        if config.model == 'effnetb1':
            backbone = efn.EfficientNetB1(weights = 'imagenet', include_top = False)(inp)
        if config.model == 'resnet50':
            backbone = tf.keras.applications.ResNet50(include_top=False, weights='imagenet',
                                                      input_shape=(512,512,3))(inp)
        if config.model == 'xception':
            backbone = tf.keras.applications.Xception(include_top=False, weights='imagenet',
                                                      input_shape=(512,512,3))(inp)
        if config.gem_pool:
            x = GeMPoolingLayer(config.p_, train_p=config.train_p)(backbone)
        else:
            x = tf.keras.layers.GlobalAveragePooling2D()(backbone)

        x = tf.keras.layers.Dense(512)(x)
        x = tf.keras.layers.BatchNormalization()(x)
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


def plot_history(history, fold, config):
  if not os.path.exists(config.LOG_PATH):
    os.makedirs(config.LOG_PATH)

  loss = history.history['loss']
  acc = history.history['sparse_categorical_accuracy']
  plt.figure(figsize=(8, 8))

  plt.subplot(2, 1, 1)
  plt.plot(loss, label='Training Loss')
  plt.ylabel('Loss')
  plt.title('Training Loss')

  plt.subplot(2, 1, 2)
  plt.plot(acc, label='Training Accuracy')
  plt.ylabel('Accuracy')
  plt.title('Training Accuracy')

  plt.savefig(os.path.join(config.LOG_PATH, f'fold_{fold}.png'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shopee')
    parser.add_argument('--model', type=str, required=True,
            help='model name')
    parser.add_argument('--num-epochs', type=int, default=20,
            help='number of training epochs')
    args = parser.parse_args()

    # Overwrite config
    config = GlobalConfig
    config.EPOCHS = args.num_epochs
    config.model = args.model

    seed_everything(config.SEED)
    strategy = init_tpu()
    config.BATCH_SIZE = 8 * strategy.num_replicas_in_sync
    print(config.__dict__)

    # Create output dir
    if not os.path.exists(config.SAVE_PATH):
        os.makedirs(config.SAVE_PATH)


    ##Train Folds
    for fold in range(config.FOLDS):
        print('\n')
        print('-'*50)
        print(f'Training with Validation Fold {fold}')
        GCS_PATH = config.GCS_PATH['fold' + str(fold)] #base GCS path

        # Retrive tfrecords
        TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/*.tfrec')
        NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
        print(f'Dataset: {NUM_TRAINING_IMAGES} training images \n')

        valid = TRAINING_FILENAMES[fold]
        #config.VALID_PATH = valid
        train = [file for file in TRAINING_FILENAMES if file != valid]
        print(f'Valid Set: {valid}')
        print(f'Train Set: {train} \n')
        config.N_CLASSES = config.OOF_CLASSES[fold]

        #assert n_class
        csv = tf.io.gfile.glob(GCS_PATH + '/*.csv')[0]
        #config.CSV_PATH = csv
        csv = pd.read_csv(csv)
        true_n_class = csv[csv['fold'] != fold]['label_group'].nunique()
        assert true_n_class == config.N_CLASSES, "Mismatch training class"
        print("Number of Training Classes: {}\n".format(config.N_CLASSES))

        train_dataset = get_training_dataset(train, ordered = False)
        train_dataset = train_dataset.map(lambda posting_id, image, label_group, matches: (image, label_group))
        # val_dataset = get_validation_dataset(valid, ordered = True)
        # val_dataset = val_dataset.map(lambda posting_id, image, label_group, matches: (image, label_group))

        STEPS_PER_EPOCH = count_data_items(train) // config.BATCH_SIZE
        K.clear_session()
        model = get_model(config)

        # Model checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(config.SAVE_PATH, \
                                    f'{config.model}_{config.IMAGE_SIZE[0]}_fold{fold}.h5'),
                                    monitor = 'loss',
                                    verbose = config.VERBOSE,
                                    save_best_only = True,
                                    save_weights_only = True,
                                    mode = 'min')

        history = model.fit(train_dataset,
                            steps_per_epoch = STEPS_PER_EPOCH,
                            epochs = config.EPOCHS,
                            callbacks = [checkpoint, get_lr_callback(config)],
                            verbose = config.VERBOSE)

        plot_history(history, fold, config)
