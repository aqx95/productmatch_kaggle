import tensorflow as tf

class GlobalConfig:
    # For tf.dataset
    AUTO = tf.data.experimental.AUTOTUNE
    # Data access
    #GCS_PATH = KaggleDatasets().get_gcs_path('shopee-tf-records-512-stratified')
    GCS_PATH = {'fold4': 'gs://kds-838b39d0b57295158cb378a4843d6eaf8f15cf439ab04b764095b419', #record-1
                'fold3': 'gs://kds-2563599881b082e80f807556b0ec28115392e8490d6e53ccd7744f15',
                'fold2': 'gs://kds-8fa83558d02d079e2cfb1505d9c0c1b073c9cdcfa284b08eb96c297c',
                }

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
    OOF_CLASSES = [8811, 8812, 8811, 8811, 8811]
    N_CLASSES = None
    # Number of folds
    FOLDS = 5
