import os
import argparse

parser = argparse.ArgumentParser(description='Animation Control')
parser.add_argument('--scenario', default='predator', type=str, help='animation scenario [planning, predator]')
args = parser.parse_args()

print(args)

if __name__ == "__main__":
    if args.scenario=='planning':
        os.system('python planning_animate.py')
    elif args.scenario=='predator':
        os.system('python predator_animate.py')
    else:
        print("Please enter a valid animation scenario")