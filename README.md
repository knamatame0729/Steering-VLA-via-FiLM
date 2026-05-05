# LLM as an optimizer in robosuite

## **☆Reward function still needs to be tweaked for better context for LLM**  

## Overview
Pre-trained VLA policies are powerful, but they are not perfect. Retraining is one solution, but it is costly and time consuming.  
Can FiLM layer be enough to fix a failing policy and enable cross-task, cross-embodiment adaptation?

## VLA Architecture
- FiLM layer is applied at the 16 dims bottleneck
- Only the FiLM params are updated, all pretrained VLA weights are kept frozen  

![VLA_Architecture](assets/vla_arch.png)

## Setup

Create a Python environment first.
```bash
conda create -n vla_film python=3.10
conda activate vla_film
```

Clone this repository
```
git clone https://github.com/knamatame0729/Steering-VLA-via-FiLM.git
cd Steering-VLA-via-FiLM
```
Install dependencies
```
pip install -r requirements.txt
```  

## Checkpoints
Make a directory for checkpoints
```
mkdir ~/Steering-VLA-via-FiLM/checkpoints
```

### Checkpoint for metaworld Pick and Place Task in Metaworld
Download checkpoints from [here](https://wandb.ai/kaitos_projects/Manual_FiLM_VLA_Testing/artifacts/model/fm_bottleneck_model/v0/files)

## Props

LLM Model : google/gemma-3-4b-it

```bash
export MODEL_SOURCE=/path/to/model
python -m scripts.llm_film_optimize --device cuda --episodes 400 --save-video --prompt-id 4 --eval-episodes 10
```

## Manual FiLM runs

Apply fixed optimal FiLM params overrides and evaluates them.

```bash
python -m scripts.manual_film_layer --checkpoint checkpoints/can_model_v2.pt --device cuda --episodes 100 --save-video
```