# **STT**: Soft Template Tuning for Few-Shot Adaptation

This is the implementation of the paper [STT: Soft Template Tuning for Few-Shot Adaptation](https://arxiv.org/abs/2207.08408).

## Quick links

* [Overview](#overview)
* [Requirements](#requirements)
* [Prepare the data](#prepare-the-data)
* [Run the model](#run-lm-bff)
* [Citation](#citation)


## Overview

In this work, we present a new prompt-tuning framework, called Soft Template Tuning (STT). STT combines manual and auto prompts, and treats downstream classification tasks as a masked language modeling task. 


## Requirements

To run our code, please install all the dependency packages by using the following command:

```
pip install -r requirements.txt
```

**NOTE**: Different versions of packages (like `pytorch`, `transformers`, etc.) may lead to different results from the paper. However, the trend should still hold no matter what versions of packages you use.

## Prepare the data

We pack the original datasets (SST-2, SST-5, MR, CR, MPQA, Subj, TREC, CoLA, MNLI, SNLI, QNLI, RTE, MRPC, QQP, STS-B) [here](https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar). Please download it and extract the files to `./data/original`, or run the following commands:

```bash
cd data
bash download_dataset.sh
```

Then use the following command (in the root directory) to generate the few-shot data we need:

```bash
python tools/generate_k_shot_data.py
```

See `tools/generate_k_shot_data.py` for more options. For results in the paper, we use the default options: we take `K=16` and take 5 different seeds of 13, 21, 42, 87, 100. The few-shot data will be generated to `data/k-shot`. In the directory of each dataset, there will be folders named as `$K-$SEED` indicating different dataset samples. You can use the following command to check whether the generated data are exactly the same as ours:

```bash
cd data/k-shot
md5sum -c checksum
```

**NOTE**: During training, the model will generate/load cache files in the data folder. If your data have changed, make sure to clean all the cache files (starting with "cache").

## Run STT

Our code is built on [transformers](https://github.com/huggingface/transformers) and we use its `3.4.0` version. Other versions of `transformers` might cause unexpected errors.

Before running any experiments, create the result folder by `mkdir result` to save checkpoints. Then you can run our code with the following bash file.

```bash experiments.sh```

The arguments in the bash file contains:
* `task`: list all the experiment tasks;
* `model_type`: list different models for comparision
  * `finetune`: Standard fine-tuning
  * `prompt-tuning`: our STT method (right model in Fig.1)
  * `prompt`: Prompt Tuning (middle model in Fig.1)
  * `prefix-tuning`: Prefix Tuning (left model in Fig.1)
  

## Bugs or questions?

If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Acknowledge

Our work is benefiting a lot from the following project:
https://github.com/princeton-nlp/LM-BFF

## Citation

Please cite our paper if you use our STT in your work:

```bibtex
@article{yu2022stt,
  title={STT: Soft Template Tuning for Few-Shot Adaptation},
  author={Yu, Ping and Wang, Wei and Li, Chunyuan and Zhang, Ruiyi and Jin, Zhanpeng and Chen, Changyou},
  journal={arXiv preprint arXiv:2207.08408},
  year={2022}
}
```
