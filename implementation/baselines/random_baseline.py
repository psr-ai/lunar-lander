import gym
env = gym.make('LunarLander-v2')

rewards = []
for _ in range(100):
    env.reset()
    isEnd = False
    reward = 0
    while not isEnd:
        step = env.step(env.action_space.sample())
        isEnd = step[2]
        reward += step[1]
        env.render()
    rewards.append(reward)

print(rewards)
