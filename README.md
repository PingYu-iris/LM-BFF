# LM-BFF (**B**etter **F**ew-shot **F**ine-tuning of **L**anguage **M**odels)

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

## Run LM-BFF

### Finetune
Our code is built on [transformers](https://github.com/huggingface/transformers) and we use its `3.4.0` version. Other versions of `transformers` might cause unexpected errors.

Before running any experiments, create the result folder by `mkdir result` to save checkpoints. Then you can run our code with the following example:

```bash
python run.py \
    --task_name SST-2 \
    --data_dir data/k-shot/SST-2/16-42 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluate_during_training \
    --model_name_or_path roberta-large \
    --few_shot_type prompt-demo \
    --num_k 16 \
    --max_steps 1000 \
    --eval_steps 100 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 0 \
    --output_dir result/tmp \
    --seed 42 \
    --template "*cls**sent_0*_It_was*mask*.*sep+*" \
    --mapping "{'0':'terrible','1':'great'}" \
    --num_sample 16 \
```

* `few_shot_type`: There are three modes
  * `finetune`: Standard fine-tuning
  * `prompt`: Prompt-based fine-tuning.
  * `prompt-demo`: Prompt-based fine-tuning with demonstrations.
* `num_k`: Number of training instances for each class. We take `num_k`=16 in our paper. This argument is mainly used for indexing logs afterwards (because the training example numbers are actually decided by the data split you use).
* `template`: Template for prompt-based fine-tuning. We will introduce the template format later.
* `mapping`: Label word mapping for prompt-based fine-tuning. It is a string of dictionary indicating the mapping from label names to label words. **NOTE**: For RoBERTa, the model will automatically add space before the word. See the paper appendix for details.
* `num_sample`: When using demonstrations during inference, the number of samples for each input query. Say `num_sample`=16, then we sample 16 different sets of demonstrations for one input, do the forward seperately, and average the logits for all 16 samples as the final prediction.

Also, this codebase supports BERT-series and RoBERTa-series pre-trained models in Huggingface's `transformers`. You can check [Huggingface's website](https://huggingface.co/models) for available models and pass models with a "bert" or "roberta" in their names to `--model_name_or_path`. Some examples would be `bert-base-uncased`, `bert-large-uncased`, `roberta-base`, `roberta-large`, etc.

### Prompt finetuning + prompt tuning

```bash
python run.py \
    --task_name SST-2 \
    --data_dir data/k-shot/SST-2/16-42 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluate_during_training \
    --model_name_or_path roberta-large \
    --few_shot_type prompt-tuning \
    --num_k 16 \
    --max_steps 1000 \
    --eval_steps 100 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 0 \
    --output_dir result/tmp \
    --seed 42 \
    --template "*cls**sent_0*_It_was*mask*.*sep+*" \
    --mapping "{'0':'terrible','1':'great'}" \
    --num_sample 16 \
    --soft_prompt_tokens 20
    
```
