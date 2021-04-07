import tensorflow as tf

class GlobalConfig:
    # For tf.dataset
    AUTO = tf.data.experimental.AUTOTUNE
    # Data access
    #GCS_PATH = KaggleDatasets().get_gcs_path('shopee-tf-records-512-stratified')
    GCS_PATH = 'gs://kds-4455b1924919edf886d7524c5507375700d9bb4fee9d03ebb699fea0'

    # Configuration
    EPOCHS = 20
    #BATCH_SIZE = 8 * strategy.num_replicas_in_sync
    IMAGE_SIZE = [512, 512]
    # Seed
    SEED = 42
    # Learning rate
    LR = 0.001
    # Verbosity
    VERBOSE = 1
    # Number of classes
    N_CLASSES = 11014
    # Number of folds
    FOLDS = 5

    # Training filenames directory
    TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/*.tfrec')
