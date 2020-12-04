from config import Config

from agent.multi_gpu_sample import Sample


def main():
    config = Config()

    agent = Sample(config)
    agent.run()


if __name__ == '__main__':
    main()
