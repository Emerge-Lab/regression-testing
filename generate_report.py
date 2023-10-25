import sys
import os
import wandb
import csv
from log_parameter import log
from generate_test_data import generate_data
import wandb.apis.reports as wr
from datetime import date, datetime
import matplotlib.pyplot as plt

wandb.login(key = os.environ["wandbKey"])
api = wandb.Api()

project_name = "test3"
entity_name = "2sabo"

generate_data()
project_data = {}
with open('fake-data.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    headers = next(csv_reader)

    for header in headers:
        project_data[header] = []

    for row in csv_reader:
        for i, value in enumerate(row):
            project_data[headers[i]].append(float(value))

if len(sys.argv) != 2:
    print("Usage: python myscript.py parameter_name")
    sys.exit(1)

# Get the parameter passed as a command-line argument
my_param = sys.argv[1]

curr_date = date.today()

if my_param not in project_data:
    print("Enter a valid parameter.")
    sys.exit(1)

log(project_data, my_param, project_name, curr_date, 1000)

my_runs = api.runs(path=f"{entity_name}/{project_name}")

# filtered_runs = [
#     run for run in my_runs if my_param in run.config
# ]
# filtered_runs.sort(key=lambda run: run.summary.get("_step", 0), reverse=True)

# filtered_runs.sort(key=lambda run: datetime.fromisoformat(run.start_time))

# x_values = []
# y_values = []

# # Iterate through the filtered runs to retrieve the last value of the metric and date
# for run in filtered_runs:
#     # Replace "metric_name" with the name of your metric
#     metric_name = "metric_name"
#     if metric_name in run.summary:
#         metric_value = run.summary[metric_name]
#         x_values.append(run.start_time)
#         y_values.append(metric_value)

# plt.plot(x_values, y_values, marker='o', linestyle='-')
# plt.xlabel("X Values")
# plt.ylabel("Y Values")
# plt.title("Line Plot of (X, Y) Coordinates")

# wandb.log({"line_plot": wandb.Image(plt)})

print(my_runs)
last_vals = []
for run in my_runs:
    history = run.scan_history(keys=[f"{my_param}"])
    run_vals = [row[f"{my_param}"] for row in history]
    if len(run_vals) != 0:
        last_vals.append(run_vals[-1])

last_vals.reverse()
print(last_vals)

wandb.init(
    # Set the project where this run will be logged
    project=project_name, 
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"Last Value Log for {curr_date}", 
    # Track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    "steps" : 1000,
    })

for val in last_vals:
    wandb.log({"last-values": val})

wandb.finish()

report = wr.Report(
    project=project_name,
    title=f'{curr_date}' + "\'s report",
    description="Here are the runs up to this date.",
    blocks=[
        wr.H1(f"All runs for project {project_name} with parameter: {my_param}"),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(
                    title="Recent Graphs for:" + f"{my_param}",
                    y=f"{my_param}",
                    title_x="steps",
                    title_y=f"{my_param}",
                    ignore_outliers=True,
                    smoothing_factor=0.5,
                    smoothing_type="gaussian",
                    smoothing_show_original=True,
                    max_runs_to_show=10,
                    font_size="large",
                    legend_position="west",
                ),
                wr.RunComparer(),
                
            ],

            # runsets=[wr.Runset(project=project_name, entity=entity_name)]
            runsets=[
                wr.Runset(project=project_name, entity=entity_name, query=f"{my_param}", name=f"Runs with {my_param}"),
            ],
        ),


        wr.H1(f"Last values for project {project_name} with parameter: {my_param}"),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(
                    title="Recent Graphs for:",
                    y="last-values",
                    title_x="steps",
                    title_y="Last Values",
                    ignore_outliers=True,
                    smoothing_factor=0.5,
                    smoothing_type="gaussian",
                    smoothing_show_original=True,
                    max_runs_to_show=10,
                    font_size="large",
                    legend_position="west",
                ),
                
            ],

            # runsets=[wr.Runset(project=project_name, entity=entity_name)]
            runsets=[
                wr.Runset(project=project_name, entity=entity_name, name=f"Last Values of Runs"),
            ],
        ),



        wr.H1(f"All runs for project: {project_name}"),
        wr.PanelGrid(
            panels=[
                # wr.LinePlot(
                #     title="Recent Graphs for:" + f"{my_param}",
                #     y=f"{my_param}",
                #     title_x="steps",
                #     title_y="{my_param}",
                #     ignore_outliers=True,
                #     smoothing_factor=0.5,
                #     smoothing_type="gaussian",
                #     smoothing_show_original=True,
                #     max_runs_to_show=10,
                #     plot_type="stacked-area",
                #     font_size="large",
                #     legend_position="west",
                # )
            ],

            # runsets=[wr.Runset(project=project_name, entity=entity_name)]
            runsets=[
                wr.Runset(project=project_name, entity=entity_name, name="Complete Run Set")
            ],
        )
    ]   
).save()                       
                 
wr.Report.from_url(report.url) 

