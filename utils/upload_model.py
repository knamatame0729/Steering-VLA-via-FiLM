import wandb

wandb.init(project="Manual_FiLM_VLA_Testing", name="model_upload")

artifact = wandb.Artifact(
    name="fm_bottleneck_model", 
    type="model"
)

artifact.add_file("/home/knamatam/mini-vla/checkpoints/fm_bottleneck_model.pt")

wandb.log_artifact(artifact)
wandb.finish()