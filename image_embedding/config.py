class GlobalConfig:
    img_size = 512
    batch_size = 16
    num_epochs = 10
    seed = 2020
    num_folds = 5

    paths = {"csv_path": "../../train.csv",
             "train_path": "../../train_images"}

    #model
    model = 'effnet'
    model_name = 'efficientnet_b3'
    model_params = {
        'n_classes':11014,
        'model_name':'efficientnet_b3',
        'use_fc':False,
        'fc_dim':512,
        'dropout':0.0,
        'loss_module':'arcface',
        's':30.0,
        'margin':0.50,
        'ls_eps':0.0,
        'theta_zero':0.785,
        'pretrained':True
    }

    #loss
    criterion = 'arcface'
    s = 30.0
    m = 0.5
    ls_eps = 0.0
    easy_margin = False

    train_step_scheduler = True
    scheduler_params = {
            "lr_start": 1e-5,
            "lr_max": 1e-5 * 16,
            "lr_min": 1e-6,
            "lr_ramp_ep": 5,
            "lr_sus_ep": 0,
            "lr_decay": 0.8,
        }
