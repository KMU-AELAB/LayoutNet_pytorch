import logging
from torch import nn


def free(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_logger(file_name):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(filename=file_name)

    file_handler.setLevel(logging.WARNING)

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def count_model_prameters(model):
    count = sum([p.data.nelement() for p in model.parameters()])
    return count


def record_image(writer, tag, epoch, image):
    writer.add_image(tag, image, epoch)


def record_images(writer, default_tag, epoch, images):
    for idx, image in enumerate(images):
        writer.add_image('{} {}'.format(default_tag, idx + 1), image, epoch)