

MODEL_CFG = {
    "net2d_cfg": {
        "num_input_channels": 1,
        "num_output_channels": 1,
        "kc": 128,
        "ks": 7,
        "ista_iters": 3
    },
    "net1d_cfg": {
        "num_input_channels": 1,
        "num_output_channels": 1,
        "kc": 32,
        "ks": 11,
        "ista_iters": 3
    }
}

TRAIN_CAVE_CFG = {
    "randga25_a0": {
        "epoch": 200,
        "batch_size": 8,
        "learning_rate": 0.0004,
        "test_data_path": "/home/zhuhl/clab/denoising/dncnn-2/cave/randga25/test.npz",
        "log_dir": "/home/zhuhl/clab/denoising/Yeager/saved_models/cave/<DIR>/randga25_a0",
        "scene_id": 0,
        "crop_size": 64,
        "alpha": 0.0,
    },
    "randga75_a0": {
        "epoch": 200,
        "batch_size": 8,
        "learning_rate": 0.0004,
        "test_data_path": "/home/zhuhl/clab/denoising/dncnn-2/cave/randga75/test.npz",
        "log_dir": "/home/zhuhl/clab/denoising/Yeager/saved_models/cave/<DIR>/randga75_a0",
        "scene_id": 0,
        "crop_size": 64,
        "alpha": 0.0,
    },
    "randga75_im_a8": {
        "epoch": 200,
        "batch_size": 8,
        "learning_rate": 0.0004,
        "test_data_path": "/home/zhuhl/clab/denoising/dncnn-2/cave/randga75_im/test.npz",
        "log_dir": "/home/zhuhl/clab/denoising/Yeager/saved_models/cave/<DIR>/randga75_im_a8",
        "scene_id": 0,
        "crop_size": 64,
        "alpha": 0.8,
    }
}




