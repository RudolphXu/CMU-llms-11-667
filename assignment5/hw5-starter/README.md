# HW5 - Efficient Training and Inference


## Setting up

### AWS
If you do not already have access to GPUs, you may need an AWS virtual machine for model training. Note that for this homework different types of instances are needed. Follow the recommendations on the handout.  
[Here are the instructions for setting that up.](https://docs.google.com/presentation/d/1Tw_klO84R9G7CZ3cINAKgy4BfdNm-8dlnRXSBIVD_3A/edit?usp=sharing) 
Note that for

### Python environment
1. Install conda: `bash setup-conda.sh && source ~/.bashrc`

2. Create conda environment:
   If you run into error like `UnavailableInvalidChannel: HTTP 403 FORBIDDEN for channel <some channel>` on your EC2 instance, you can solve it by running `conda config --remove channels <some channel>`, and make sure you have the default channel by running `conda config --add channels defaults`.
```bash
conda create -n cmu-llms-hw5 python=3.11
conda activate cmu-llms-hw5
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 -c pytorch
pip install -r requirements.txt
pip install ninja
pip install flash-attn==2.6.3 --no-build-isolation
pip install wandb
pip install -U "huggingface_hub[cli]"
```
3. Run `wandb login` to finish setting up weights & biases for experiment tracking (you will need to have a [weights & biases account](https://wandb.ai/login)).  

4. Run `huggingface-cli login` to allow downloading and uploading to Huggingface hub.  

5. Run `./get_initial_model.sh` to download the model starting point.  

## Contents
This repo contains a simple huggingface-based pre-training script, supporting two datasets:

- two splits of wikitext, with 50M tokens each. Both splits are pre-tokenized for your convinience, one set with sequences of 512 tokens, and the other 2048.
- minipile, with 1.4B tokens, pre-tokenized with sequences of 2048 tokens.


## Pre-training

The folder ```scripts``` contains access points to the pre-training code. All scripts under there can be called as follows:

```./scripts/launch_<name>.sh <path_to_config>```, where ```<path_to_config>``` points to a model configuration under ```configs```.

## Pushing your model to HuggingFace Hub

One question asks you to push your model to the huggingface hub. Steps:

1- Create an account. In your account, create a **public** repo for your model. It's handle will be your_username/model_name.  

2- cd to the model directory you want to push   

3- run ```huggingface-cli upload your_username/model_name .```  

