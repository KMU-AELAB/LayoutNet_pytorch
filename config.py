class Config(object):
    epoch = 5000
    batch_size = 16
    learning_rate = 0.002
    pretraining_step_size = 220

    sigma = 1.0

    cuda = True
    gpu_cnt = 4

    async_loading = True
    pin_memory = True

    root_path = '/home/D2019063/LayoutNet_pytorch'
    data_path = 'data'

    summary_dir = 'board'
    checkpoint_dir = 'trained'
    checkpoint_file = 'checkpoint.pth.tar'
