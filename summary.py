from torchsummary import summary

from graph.model.model import Model


if __name__ == '__main__':
    model = Model().cuda()
    summary(model, (6, 512, 1024), batch_size=10)