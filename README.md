<div align="center">

# How to run

</div>

### Getting started
Create a conda environment
```
conda create --name mini-vla python=3.10
conda activate mini-vla
```

Clone repo
```

```

Install dependencies
```
pip install -r requirements.text
```

### Donwload the model (3.7MB)
Download the [VLA Model](https://wandb.ai/kaitos_projects/Manual_FiLM_VLA_Testing/artifacts/model/fm_bottleneck_model/v0/files) in /checkpoints/ directory
### FiLM is applied into Bottleneck in fusion.py
- 16 dims of output that we can apply FiLM paramters (gamma, beta)
```
python -m scripts.manual_film_layer --device cuda --episodes 100 --robot sawyer  --save-video --checkpoint checkpoints/fm_bottleneck_model.pt
```

### CMA-ES
```
python -m scripts.manual_film_layer --device cuda --checkpoint checkpoints/fm_bottleneck_model.pt --episodes 100 --save-video --env-name pick-place-v3
```
