from report_generation import generate_report
import sys
import os
import wandb

wandb.login(key = os.getenv("wandbKey"))

if len(sys.argv) != 4:
    print("Usage: python myscript.py parameter_name project_name entity_name")
    sys.exit(1)

# Get the parameter passed as a command-line argument
my_param = sys.argv[1]
project_name = sys.argv[2]
entity_name = sys.argv[3]

code = generate_report(my_param, project_name, entity_name)

if code == 0:
    print("Report generated!")
elif code == -1:
    print("Enter a valid parameter.")
    sys.exit(1)
