import tensorflow as tf

class GlobalConfig:
    # For tf.dataset
    AUTO = tf.data.experimental.AUTOTUNE
    # Data access
    ## 512
    GCS_PATH = {'fold4': 'gs://kds-9ba2f2cb51c9e0518f7b325c49d244898c091332094101eeb05d5dc6', #record-1
                'fold3': 'gs://kds-01c5abe8d9c96391d70580634bb0744d6d9bda70b650d3ed83342146',
                'fold2': 'gs://kds-e0634726706afef1b28d34064cca8b0fb57a64ec2762334798cd6a6c',
                'fold1': 'gs://kds-b286d2c01c3227c4cba912757f12bf8019835b4785fa47aa5ff6126c',
                'fold0': 'gs://kds-a4851a353f29361cb5c5326c9c16e4a1ed05ac260fe4231237f82b37'
                }

    ## 680
    # GCS_PATH = {'fold4': 'gs://kds-43dd486cb7ffb45145165d99d299efcd30de323600580814aef601e6', #record-1
    #             'fold3': 'gs://kds-addf59ea07607b02811f4213cc050bbb4a09f22b6527859d5b3409e7',
    #             'fold2': 'gs://kds-85212c7e5994b9e5cad590ea8eeb4fa78c573655b7d016c2f48cdb6f',
    #             'fold1': 'gs://kds-57c19e6c4fce34330230a18f286cc444591957d6938b0ab253c60e8d',
    #             'fold0': 'gs://kds-0b3d7f76804467d42dd2d6863344ce26646251c54ee34eaa8f2b6175'
    #             }

    SAVE_PATH = 'checkpoint'
    LOG_PATH = 'history'

    # Configuration
    EPOCHS = 20
    IMAGE_SIZE = [512, 512]
    # Seed
    SEED = 42
    # Learning rate
    LR = 0.001
    # Verbosity
    VERBOSE = 1
    # Number of classes
    OOF_CLASSES = [8811, 8812, 8811, 8811, 8811] #value for oof != index
    N_CLASSES = None
    # Number of folds
    FOLDS = 5

    #Arcface
    scale = 30
    margin = 0.7

    #gem
    gem_pool = False
    p_ = 3.0
    train_p = False
