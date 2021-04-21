import tensorflow as tf

class GlobalConfig:
    # For tf.dataset
    AUTO = tf.data.experimental.AUTOTUNE
    # Data access
    ## 512
    # GCS_PATH = {'fold4': 'gs://kds-7209d3fbb77ddfff30e833d015cac7e2314371c5b7e879012dfaf14e', #record-1
    #             'fold3': 'gs://kds-d49a15b017cf2f17384722b2cd1b26208bb8753643b4e41391a3c743',
    #             'fold2': 'gs://kds-35f2577fc867feb30fef097c871194ad564ec81a9b59d8899f50d457',
    #             'fold1': 'gs://kds-b7d74b30808c58b508415d01cedb4af2ca89d114599f20487c7e6d81',
    #             'fold0': 'gs://kds-844eef7c7a46975154939ecb988d6f10a0615b24fccb6004c37382cc'
    #             }

    ## 680
    GCS_PATH = {'fold4': 'gs://kds-43dd486cb7ffb45145165d99d299efcd30de323600580814aef601e6', #record-1
                'fold3': 'gs://kds-addf59ea07607b02811f4213cc050bbb4a09f22b6527859d5b3409e7',
                'fold2': 'gs://kds-85212c7e5994b9e5cad590ea8eeb4fa78c573655b7d016c2f48cdb6f',
                'fold1': 'gs://kds-57c19e6c4fce34330230a18f286cc444591957d6938b0ab253c60e8d',
                'fold0': 'gs://kds-0b3d7f76804467d42dd2d6863344ce26646251c54ee34eaa8f2b6175'
                }

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
