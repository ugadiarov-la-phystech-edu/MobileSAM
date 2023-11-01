import argparse
import os
from pathlib import Path

import cv2
import gym

import envs


def collect_data(dataset_size):
    env = gym.make('Navigation5x5-v0')
    observations = []
    for i in range(dataset_size):
        observations.append(env.reset())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_size', type=int, default=100000)
    parser.add_argument('--dataset_path', type=str, default='shapes2d_dataset')

    args = parser.parse_args()

    env = gym.make('Navigation5x5-v0')
    observations = []
    for i in range(args.dataset_size):
        observations.append(env.reset())

    Path(args.dataset_path).mkdir(parents=True, exist_ok=True)
    for i, observation in enumerate(observations):
        path = os.path.join(args.dataset_path, f'{i:06d}.png')
        cv2.imwrite(path, cv2.cvtColor(observation, cv2.COLOR_RGB2BGR))
