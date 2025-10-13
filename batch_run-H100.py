import subprocess
import itertools
import os


# re-align

lrs=[1e-5]
beta=0.1
effective_batch=12
base_model="llava-v1.5-7b"
model_name=f"liuhaotian/{base_model}"
# run_name=f"dpo-llava-v1.5-7b-lr-{lr}-acc_batch-$effective_batch-beta-{beta}"


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

# yilin's loss

use_text_similarity=True
ls_factor_text_weights=[0.3,0.5,0.7,0.9]
use_img_similarity=True
ls_factor_img_weight=0.5
# run_name=f"direct-last_hidden_state-lr-{lr}-acc_batch-$effective_batch-text_weight-{ls_factor_text_weight}-img_weight-{ls_factor_img_weight}"

# beta_dpo setting

beta_dpo=False
ls_factor_weight=0.1
# run_name="beta_dpo-lr-$lr-acc_batch-$effective_batch-beta-$beta-ls_factor_weight-$ls_factor_weight"

# anchor
use_anchor=False
# run_name="anchor-lr-$lr-acc_batch-$effective_batch-beta-$beta"

master_port=60002

# data_path="./preference_data/yilin_pref_data_last_hidden_state.json"
data_path="./preference_data/yilin_pref_data_pooler_output.json"

image_folder="/data/yilin/train2014/"


# for ls_factor_text_weight in ls_factor_text_weights:
for ls_factor_text_weight, lr in itertools.product(ls_factor_text_weights, lrs):
    # os.makedirs(output_dir, exist_ok=True)

    ls_factor_img_weight = ls_factor_text_weight
    run_name=f"reverse-pooler_output-lr-{lr}-acc_batch-{effective_batch}-text_weight-{ls_factor_text_weight}-img_weight-{ls_factor_img_weight}"
    pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"
    cmd = [
        f"""deepspeed --include=localhost:0,1,2,3,4,5  --master_port {master_port} train_rdpo.py \
        --model_name_or_path {model_name} \
        --data_path {data_path} \
        --deepspeed "./deepspeed/zero2.json" \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps {effective_batch} \
        --evaluation_strategy "no" \
        --save_strategy "no" \
        --learning_rate {lr} \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --bf16 True \
        --lora_enable True \
        --beta {beta} \
        --output_dir {pretrained} \
        --image_folder {image_folder} \
        --mm_projector_lr 2e-5 \
        --mm_projector_type mlp2x_gelu \
        --run_name {run_name} \
        --project_name "yilin-align" \
        --use_text_similarity {use_text_similarity}  \
        --ls_factor_text_weight {ls_factor_text_weight} \
        --use_img_similarity {use_img_similarity} \
        --ls_factor_img_weight {ls_factor_img_weight} \
        --beta_dpo {beta_dpo} \
        --ls_factor_weight {ls_factor_weight} """
    ]

    print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
    ret = subprocess.run(cmd, shell=True)

    if ret.returncode != 0:
        print(f"❌ Failed: {base_model} lr={lr} bs={effective_batch}")
    else:
        print(f"✅ Finished: {base_model} lr={lr} bs={effective_batch}")

