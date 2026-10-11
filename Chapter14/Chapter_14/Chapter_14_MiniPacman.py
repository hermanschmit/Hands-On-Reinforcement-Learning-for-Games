import numpy as np
from minipacman import MiniPacman

import matplotlib.pyplot as plt

def displayImage(image, step, reward):
    s = "step" + str(step) + " reward " + str(reward)
    plt.clf()
    plt.title(s)
    plt.imshow(image)
    # The original called the blocking plt.show(), so every move needed the window closed first. Draw without blocking.
    plt.pause(0.001)

keys = {
    'w': 2,
    'd': 1,
    'a': 3,
    's': 4,
    ' ': 0
}

MODES = ('regular', 'avoid', 'hunt', 'ambush', 'rush')
frame_cap = 1000

mode = 'rush'

env = MiniPacman(mode, 1000)

state = env.reset()
done = False

total_reward = 0
step = 1

displayImage(state.transpose(1, 2, 0), step, total_reward)

while not done:
    # NOTE: The original had the input line commented out (it used Python 2's raw_input), so x was never defined
    # and the loop printed the error message forever.
    x = input("Move (w/a/s/d, space to stay): ")
    try:
        keys[x]
    except:
        print("Only 'w' 'a' 'd' 's'")
        continue
    action = keys[x]
    
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    step += 1
    displayImage(next_state.transpose(1, 2, 0), step, total_reward)