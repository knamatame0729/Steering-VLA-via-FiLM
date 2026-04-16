<div align="center">

# How to run

</div>

### Getting started
Create a conda environment
```
conda create -n robomimic_venv python=3.8.0
conda activate robomimic_venv
```
Install PyTorch
```
conda install pytorch==2.0.0 torchvision==0.15.1 -c pytorch
```

Install robomimic
```
pip install robomimic
```

Clone repo
```
git clone https://github.com/knamatame0729/Steering-VLA-via-FiLM.git
```

Install robosuite
```
cd Steering-VLA-via-FiLM
git submodule add https://github.com/ARISE-Initiative/robosuite.git robosuite
cd robosuite
pip install -r requirements.txt
```
Install cmaes
```
pip install cmaes
```

Create a directory for checkpoints
```
cd Steering-VLA-via-FiLM
mkdir checkpoints
```

### Donwload the model (3.7MB)
Download the [VLA Model](https://wandb.ai/kaitos_projects/Manual_FiLM_VLA_Testing/artifacts/model/fm_bottleneck_model/v0/files) in /checkpoints/
### FiLM is applied into Bottleneck in fusion.py
- 16 dims of output that we can apply FiLM paramters (gamma, beta)
```
python -m scripts.manual_film_layer --device cuda --episodes 100 --save-video --checkpoint checkpoints/model.pt
```

### CMA-ES
```
python -m scripts.optimize_film_params_cmaes --checkpoint checkpoints/model.pt --device cuda --eval-episodes 10 --cmaes-popsize 300 --cmaes-generations 20 --cmaes-sigma0 0.05
```
