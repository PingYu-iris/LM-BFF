# seed in 13 21 42 87 100
# bs in 2 4 8
# lr in 1e-5 2e-5 5e-5
# task in SST-2 sst-5 mr cr mpqa trec SNLI QNLI QQP

for seed in 13 21 42 87 100
do
    for bs in 2
    do
        for lr in 2e-5
        do
            for k in 1 2
            do
                for soft_prompt_tokens in 25
                do
                    for model_type in prefix-tuning-stanford
                    do
                        for max_step in 500
                        do
                            for task in SST-2 sst-5 mr cr mpqa trec SNLI QNLI QQP
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

