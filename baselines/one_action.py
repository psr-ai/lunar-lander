import gym
env = gym.make('LunarLander-v2')

for action in range(3):
    rewards = []
    for _ in range(5):
        env.reset()
        is_end = False
        reward = 0
        while not is_end:
            step = env.step(action)
            is_end = step[2]
            reward += step[1]
            env.render()
        rewards.append(reward)

    print(rewards)
