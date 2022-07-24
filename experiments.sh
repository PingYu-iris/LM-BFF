# seed in 13 21 42 87 100
# bs in 2 4 8
# lr in 1e-5 2e-5 5e-5
# task in SST-2 sst-5 mr cr mpqa trec SNLI QNLI QQP 
# model_type in prompt-tuning prompt prefix-tuning finetune
for task in mpqa trec SNLI QNLI QQP
do 
    for seed in 13 21 42 87 100
    do
        for bs in 32
        do
            for lr in 2e-5
            do
                for k in 1
                do
                    for soft_prompt_tokens in 25
                    do
                        for model_type in prompt-tuning
                        do
                            for max_step in 500
                            do
                                TAG=exp \
                                TYPE=$model_type \
                                TASK=$task \
                                BS=$bs \
                                LR=$lr \
                                SEED=$seed \
                                MODEL=roberta-large \
                                SOFT_PROMPT_NUM=$soft_prompt_tokens\
                                K=$k \
                                MAX_STEP=$max_step \
                                bash run_experiment.sh
                            done
                        done
                    done
                done
            done
        done
    done
done

