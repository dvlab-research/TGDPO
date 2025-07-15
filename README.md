# TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization

This repository contains the code for our paper [TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization](https://arxiv.org/abs/2506.14574).

## Overview
This work introduces a framework for incorporating token-level reward guidance into preference optimization. Experiment results demonstrate that TGDPO achieves substantial performance improvements over DPO and SimPO, with win rate gains of up to 7.5 points on MT-Bench, 6.2 points on AlpacaEval 2, and 4.3 points on Arena-Hard.
## Installation

Environment preparation:

```bash
conda env create -f environment.yml
conda activate tgdpo
pip install -e ".[torch,metrics]"
```

## Dataset Preparation
We provide the training data in the following links:

- [Llama3-8B-Instruct PairRM](https://huggingface.co/datasets/mk1111/llama3-8b-instruct-ultrafeedback/tree/main)
- [Llama3-8B-Instruct ArmoRM](https://huggingface.co/datasets/mk1111/llama3-8b-instruct-ultrafeedback-armorm/tree/main)
- [Llama3.2-3B-Instruct ArmoRM](https://huggingface.co/datasets/mk1111/llama3.2-3b-instruct-ultrafeedback-armorm/tree/main)
- [Gemma2-2B-it ArmoRM](https://huggingface.co/datasets/mk1111/gemma2-2b-it-ultrafeedback-armorm/tree/main)

After downloading the training data, please adjust their corresponding path to dataset in `data/dataset_info.json`.

## Token-level Reward Model Preparation

You can use models trained with DPO, SimPO, or other RLHF algorithms on the datasets above as the token-level reward models. You can also leverage any off-the-shelf open-source token-level reward models as guidance.


## Training Scripts
The example training script is in `examples/llama3_8b_instruct_tgdpo.yaml`. The training config is set for 8x80GB GPUs. You will need to adjust `model_name_or_path` and `ref_model` to specify the base model (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`), and set the path of the token-level reward model in `tgdpo_reward_model`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch  --config_file ./examples/accelerate/fsdp_config.yaml  ./src/train.py  ./examples/llama3_8b_instruct_tgdpo.yaml
```

## Acknowledgements

We would like to thank the authors of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for their excellent code base.

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{
zhu2025tgdpo,
title={{TGDPO}: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization},
author={Mingkang Zhu and Xi Chen and Zhongdao Wang and Bei Yu and Hengshuang Zhao and Jiaya Jia},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=TKHWvyzR1t}
}
```