from config import Config

from agent.edge import Edge
from agent.corner import Corner
from agent.box import Box
from agent.total import Total


def main():
    config = Config()

    print('#### pre-train edge model ####')
    agent = Edge(config)
    agent.run()

    print('#### pre-train edge/corner model ####')
    agent = Corner(config)
    agent.run()

    print('#### pre-train regress model ####')
    agent = Box(config)
    agent.run()

    print('#### train all model ####')
    agent = Total(config)
    agent.run()


if __name__ == '__main__':
    main()
