import wandb
from pathlib import Path

wandb.init(project="Manual_FiLM_VLA_Testing", name="model_upload")

artifact = wandb.Artifact(
    name="can_model_v2", 
    type="model"
)

model_path = Path("~/VLA-via-FiLM/checkpoints/can_model_v2.pt").expanduser()
if not model_path.is_file():
    raise FileNotFoundError(f"Model file not found: {model_path}")

artifact.add_file(str(model_path))

wandb.log_artifact(artifact)
wandb.finish()