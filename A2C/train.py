from agents import AgentA2C
from networks import networkV2
from workers import Worker
import gym


def train():
    actions = 5
    workers_len = 50
    iterations = 1000001
    stack = 4
    steps = 10

    agent = AgentA2C("final0", networkV2(actions),
                     0.99, [0.001, 0.0005, 0.0001], 0.01, id=0)

    workers = []
    for id in range(workers_len):
        env = gym.make("CartPole-v0")
        env.seed(id)
        w = Worker(id, env, agent, print_score=True)
        workers.append(w)

    episode = 0
    # path = 'models/final/final_3_10000_a2c.pt'
    # agent.load_model(path)
    agent.learn(workers, iterations, steps, start_episode=episode)


if __name__ == '__main__':
    train()
