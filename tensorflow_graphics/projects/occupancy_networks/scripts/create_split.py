import argparse
import os
import random

parser = argparse.ArgumentParser(
    description='Split data into train, test and validation sets.')
parser.add_argument('in_folder', type=str,
                    help='Input folder where data is stored.')

parser_nval = parser.add_mutually_exclusive_group(required=True)
parser_nval.add_argument('--n_val', type=int,
                         help='Size of validation set.')
parser_nval.add_argument('--r_val', type=float,
                         help='Relative size of validation set.')

parser_ntest = parser.add_mutually_exclusive_group(required=True)
parser_ntest.add_argument('--n_test', type=int,
                          help='Size of test set.')
parser_ntest.add_argument('--r_test', type=float,
                          help='Relative size of test set.')

parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--seed', type=int, default=4)

args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)

all_samples = [name for name in os.listdir(args.in_folder)
               if os.path.isdir(os.path.join(args.in_folder, name))]

if args.shuffle:
    random.shuffle(all_samples)

# Number of examples
n_total = len(all_samples)

if args.n_val is not None:
    n_val = args.n_val
else:
    n_val = int(args.r_val * n_total)

if args.n_test is not None:
    n_test = args.n_test
else:
    n_test = int(args.r_test * n_total)

if n_total < n_val + n_test:
    print('Error: too few training samples.')
    exit()

n_train = n_total - n_val - n_test

assert(n_train >= 0)

# Select elements
train_set = all_samples[:n_train]
val_set = all_samples[n_train:n_train+n_val]
test_set = all_samples[n_train+n_val:]

# Write to file
with open(os.path.join(args.in_folder, 'train.lst'), 'w') as f:
    f.write('\n'.join(train_set))

with open(os.path.join(args.in_folder, 'val.lst'), 'w') as f:
    f.write('\n'.join(val_set))

with open(os.path.join(args.in_folder, 'test.lst'), 'w') as f:
    f.write('\n'.join(test_set))
