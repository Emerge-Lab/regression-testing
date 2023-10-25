import wandb
from datetime import datetime
import random
wandb.login(key = "c1eb42a0671e3abfae933e6c884e6ae4bacec208")

# # Initialize WandB
# wandb.init(project="your_project_name", config={"model": "your_model_name"})

# # Log success rates with timestamps
# now = datetime.now()
# current_time = now.strftime("%Y-%m-%d %H:%M:%S")
# success_rate = random.random()


run = wandb.init(id="etg3za77", resume="allow")
run.log({"loss": 0.2}, commit=False)
# Somewhere else when I'm ready to report this step:
run.log({"accuracy": 0.8})
run.save()
print(run.name)