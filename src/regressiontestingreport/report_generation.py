import wandb
from log_parameter import log
import wandb.sdk.wandb_metric
import wandb.apis.reports as wr
from datetime import date, datetime
import matplotlib.pyplot as plt
import time

def log_last_values(param, project_name, entity_name):
    api = wandb.Api()
    my_runs = api.runs(path=f"{entity_name}/{project_name}")
    
    last_vals = []
    last_vals_date = []
    last_vals_commit = []
    for run in my_runs:
        history = run.scan_history(keys=[f"{param}"])
        run_vals = [row[f"{param}"] for row in history]
        if len(run_vals) != 0:
            last_vals.append(run_vals[-1])
            last_vals_date.append(run.config["date"])
            # last_vals_commit.append(run.config["commit"])

    last_vals.reverse()
    last_vals_date.reverse()
    last_vals_commit.reverse()

    curr_date = date.today()

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
        "date" : str(curr_date),
        })

    for val in last_vals:
        wandb.log({"Last Values": val})

    wandb.finish()
    return last_vals

def get_parameter_runs_block(project_name, entity_name, param):
    param_run_block = {
        "header" : wr.H1(f"All runs for project {project_name} with parameter: {param}"),
        "panel_grid" : wr.PanelGrid(
                panels=[
                    wr.LinePlot(
                        title="Recent Graphs for:" + f"{param}",
                        y=f"{param}",
                        title_x="steps",
                        title_y=f"{param}",
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
                    wr.Runset(project=project_name, entity=entity_name, query=f"{param}", name=f"Runs with {param}"),
                ],
            ),
    }
    return param_run_block

def get_last_values_block(project_name, entity_name, param):
    last_value_block = {
        "header" :  wr.H1(f"Last values for project {project_name} with parameter: {param}"),
        "panel_grid" : wr.PanelGrid(
            panels=[
                wr.LinePlot(
                    title="Last Value Graph:",
                    y="Last Values",
                    title_x="Date",
                    title_y="{param}",
                    range_y=[0, 2],
                    ignore_outliers=True,
                    max_runs_to_show=1,
                    font_size="large",
                    legend_position="west",
                ),
                
            ],
            runsets=([[
                wr.Runset(project=project_name, entity=entity_name, query="Last Value Log", name=f"Last Values of Runs"),
            ][-1]]),
        ),
    }
    return last_value_block

def get_all_runs_block(project_name, entity_name):
    all_runs_block = {
        "header" : wr.H1(f"All runs for project: {project_name}"),
        "panel_grid" : wr.PanelGrid(
            panels=[
            ],
            runsets=[
                wr.Runset(project=project_name, entity=entity_name, name="Complete Run Set")
            ],
        ),
    }
    return all_runs_block

def generate_report(param, project_name, entity_name):
    curr_date = date.today()

    log_last_values(param, project_name, entity_name)

    param_runs_block = get_parameter_runs_block(project_name, entity_name,  param)
    last_value_block = get_last_values_block(project_name, entity_name, param)
    all_runs_block = get_all_runs_block(project_name, entity_name)

    report_blocks=[
        param_runs_block["header"],
        param_runs_block["panel_grid"],

        last_value_block["header"],
        last_value_block["panel_grid"],

        all_runs_block["header"],
        all_runs_block["panel_grid"],
    ]

    report = wr.Report(
        project=project_name,
        title=f'{curr_date}' + "\'s report",
        description="Here are the runs up to this date.",
        blocks = report_blocks,  
    ).save()                     
                
    return report

