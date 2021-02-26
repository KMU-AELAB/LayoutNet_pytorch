from config import Config

from agent.edge import Edge
from agent.corner import Corner
from agent.box import Box
from agent.total import Total


def main():
    config = Config()

    agent = Edge(config)
    agent.run()

    agent = Corner(config)
    agent.run()

    agent = Box(config)
    agent.run()

    agent = Total(config)
    agent.run()


if __name__ == '__main__':
    main()
