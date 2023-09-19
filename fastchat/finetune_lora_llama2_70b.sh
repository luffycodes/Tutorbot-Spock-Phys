# https://github.com/lm-sys/FastChat/pull/322
# https://www.reddit.com/r/LocalLLaMA/comments/15sgg4m/what_modules_should_i_target_when_training_using/
# https://www.reddit.com/r/LocalLLaMA/comments/1578ahb/target_modules_for_llama2_for_better_finetuning/
# https://huggingface.co/uwnlp/llama-2-70b-qlora-openorca
# https://gist.github.com/jondurbin/87fc040b92a3073125ed516b04bc6e19
deepspeed fastchat/train/train_higgs_lora.py \
    --model_name_or_path meta-llama/Llama-2-70b-chat-hf  \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_target_modules "q_proj", "k_proj", "o_proj", "v_proj", "down_proj", "gate_proj", "up_proj" \
    --lora_dropout 0.05 \
    --data_path fastchat_conversations_format.json \
    --cache_dir /data/ss164/cache \
    --output_dir /data/ss164/spock_physics/lora-higgs-llama-vicuna-ep25-70b \
    --num_train_epochs 25 \
    --bf16 True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 1000000  \
    --save_strategy "steps" \
    --save_steps 2000000 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --q_lora False \
    --deepspeed default_offload_opt_param.json \
    --gradient_checkpointing True \
    --lazy_preprocess True
    #--flash_attn False
