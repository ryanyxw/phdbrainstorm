CLUSTER="ai2/jupiter"
PRIORITY="urgent"
WORKSPACE=ai2/flex2

command='''
export HF_HOME="/root/ryanwang/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
cd FlexOlmo && bash src/scripts/eval/setup_eval_env.sh && cd ..
bash eval.sh
'''

gantry run \
    --name "eval_moe" \
    --description "eval_moe on minerva intermediate algebra" \
    --workspace $WORKSPACE \
    --beaker-image 'ai2/cuda12.8-dev-ubuntu22.04-torch2.6.0' \
    --venv 'base' \
    --timeout -1 \
    --show-logs \
    --gpus 4 \
    --host-networking \
    --priority "${PRIORITY}" \
    --leader-selection \
    --replicas 1 \
    --cluster "${CLUSTER}" \
    --budget ai2/oe-base \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
    --env-secret HF_TOKEN=RYAN_HF_TOKEN \
    --env-secret WANDB_API_KEY=RYAN_WANDB_API_KEY \
    --shared-memory 10GiB \
    --weka=oe-training-default:/root \
    --yes \
    -- bash -c "$command"



#    --install "pip install -r requirements.txt" \
#    --mount "src=weka,ref=oe-training-default,subpath=ryanwang,dst=/root" \
#    --conda-env "onboarding" \
#