from report_generation import generate_report
import sys
import os
import warnings
import wandb
import argparse

wandb_key =  os.environ.get("wandbKey")

if not wandb_key:
    # API key is missing or empty
    warnings.warn("Warning: W&B API key is not set.")
else:
    # API key is present, so log in
    wandb.login(key=wandb_key)


parser = argparse.ArgumentParser(description='Take in inputs')
parser.add_argument('my_param', metavar='param', type=str, nargs=1,
                    help='The parameter you wish to log')
parser.add_argument('project_name', metavar='project name', type=str, nargs=1,
                    help='The name of the project')
parser.add_argument('entity_name', metavar='entity name', type=str, nargs=1,
                    help='The name of the entity who owns the project')

args = parser.parse_args()

code = generate_report(args.my_param[0], args.project_name[0], args.entity_name[0])

if code == 0:
    print("Report generated!")
elif code == -1:
    print("Enter a valid parameter.")
    sys.exit(1)

