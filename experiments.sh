# seed in 13 21 42 87 100
# bs in 2 4 8
# lr in 1e-5 2e-5 5e-5


for seed in 42
do
    for bs in 2
    do
        for lr in 2e-5
        do
            for k in 3 8 16 64 256
            do
                for soft_prompt_tokens in 30
                do
                    for model_type in prefix-tuning-stanford
                    do
                        for max_step in 500
                        do
                            for task in SST-2
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

