import wandb

wandb.init(project="Manual_FiLM_VLA_Testing", name="model_upload")

artifact = wandb.Artifact(
    name="can_model_v2", 
    type="model"
)

artifact.add_file("./checkpoints/can_model_v2.pt")

wandb.log_artifact(artifact)
wandb.finish()