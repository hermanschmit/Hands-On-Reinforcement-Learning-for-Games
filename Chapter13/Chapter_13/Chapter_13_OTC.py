# NOTE: This chapter needs the Obstacle Tower game itself, which can no longer be downloaded: the official builds
# (https://storage.googleapis.com/obstacle-tower-build/v4.1/...) now return "Access denied". If you have a copy,
# extract it to ./ObstacleTower and install the environment in a separate virtual environment with Python 3.7 or
# 3.8, e.g. pip install git+https://github.com/Unity-Technologies/obstacle-tower-env. It needs gym (not gymnasium)
# and mlagents-envs 0.17/0.18, which conflict with the packages used by the other chapters (Chapter 11 needs
# mlagents-envs 0.28). The obs_tower2 code is written against that same gym-based API.
from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation


def run_episode(env):
    done = False
    episode_return = 0.0
    
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        episode_return += reward
    return episode_return


if __name__ == '__main__':
    # In this example we use the seeds used for evaluating submissions 
    # to the Obstacle Tower Challenge.
    eval_seeds = [1001, 1002, 1003, 1004, 1005]

    # Create the ObstacleTowerEnv gym and launch ObstacleTower
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower')

    # Wrap the environment with the ObstacleTowerEvaluation wrapper
    # and provide evaluation seeds.
    env = ObstacleTowerEvaluation(env, eval_seeds)

    # We can run episodes (in this case with a random policy) until 
    # the "evaluation_complete" flag is True.  Attempting to step or reset after
    # all of the evaluation seeds have completed will result in an exception.
    while not env.evaluation_complete:
        episode_rew = run_episode(env)

    # Finally the evaluation results can be fetched as a dictionary from the 
    # environment wrapper.
    print(env.results)

    env.close()
