import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--n-epochs', type=int, default=500, help='the number of epochs to train the agent')
    parser.add_argument('--n-batch', type=int, default=200, help='the number of batches in one epoch')
    parser.add_argument('--network-learn-freq', type=int, default=40, help='the number to learn in one epoch')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')  # 256
    parser.add_argument('--model-name', type=str, default='agent_4_sat', help='the name of saved or loaded models')
    parser.add_argument('--buffer-size', type=int, default=int(2e7), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor of bellman')
    parser.add_argument('--lr-actor', type=float, default=9e-4, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=9e-4, help='the learning rate of the critic')
    parser.add_argument('--change-goal-cycle', type=int, default=10, help='the number of cycles to change the goal')
    parser.add_argument('--max-cycle-steps', type=int, default=16, help='the max number of steps within one cycle')
    parser.add_argument('--no-layer-norm', action='store_false', dest='layer_norm', help='disable layer normalization')
    parser.add_argument('--tau', type=float, default=0.005, help='soft update rate of target networks')
    parser.add_argument('--target-update-freq', type=int, default=3, help='update frequency of target networks')
    parser.add_argument('--test-episodes', type=int, default=5, help='number of episodes to test')
    parser.add_argument('--no-plot', action='store_false', dest='plot', help='whether to plot')
    parser.add_argument('--log-alpha', type=float, default=-2.0, help='initial value of alpha')
    parser.add_argument('--alpha-lr', type=float, default=3e-4, help='the learning rate of alpha')
    parser.add_argument('--log-dir', type=str, default='logs_4sat/train_1', help='the logging direction')

    args = parser.parse_args()

    return args

# 4sat
# epoch:500  batch:200  learn freq:40  batch-size:256  buffer-size:1e6  replay k:4  change Goal cycle:10  max cycle steps:16  target update freq:3  log alpha:-2  lr:3e-4,1e-4
# epoch:500  batch:200  learn freq:40  batch-size:256  buffer-size:1e6  replay k:4  change Goal cycle:10  max cycle steps:16  target update freq:3  log alpha:-2  lr:9e-4,3e-4
# 6sat
# epoch:100  batch:6000  learn freq:60  batch-size:1024  buffer-size:2e7  replay k:4  change Goal cycle:30  max cycle steps:35  target update freq:6  log alpha:-0.5  lr:3e-4,1e-4
# epoch:1000  batch:500  learn freq:40  batch-size:512  buffer-size:2e7  replay k:4  change Goal cycle:15  max cycle steps:25  target update freq:5  log alpha:-1  lr:6e-4,2e-4
# epoch:1200  batch:500  learn freq:40  batch-size:512  buffer-size:2e7  replay k:4  change Goal cycle:15  max cycle steps:25  target update freq:5  log alpha:-1  lr:9e-4,3e-4

# train1:normal
# train2:without HER
# train3:without mask
# train4:without both