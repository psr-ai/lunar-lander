import gym

env = gym.make('LunarLander-v2')
actions = {
    0: 'Idle',
    1: 'Fire Right Engine / Rotate Anticlockwise',
    2: 'Thrust',
    3: 'Fire Left Engine / Rotate Clockwise'
}

for action in actions:
    rewards = []
    for _ in range(100):
        env.reset()
        is_end = False
        reward = 0
        while not is_end:
            step = env.step(action)
            is_end = step[2]
            reward += step[1]
            env.render()
        rewards.append(reward)

    print("Rewards for %s: %s" % (actions[action], rewards))
    print("Avg Reward for %s: %s" % (actions[action], (float(sum(rewards))/len(rewards))))
