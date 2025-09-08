CLUSTER="ai2/jupiter"
PRIORITY="urgent"
WORKSPACE=ai2/flex2

command="echo hello && echo hello > /data/input/ryanwang/testing3.txt && echo goodbye"

gantry run \
    --task-name "test_gantry" \
    --description "Run first test of gantry" \
    --workspace $WORKSPACE \
    --beaker-image 'ai2/cuda12.8-dev-ubuntu22.04-torch2.6.0' \
    --venv 'base' \
    --timeout -1 \
    --show-logs \
    --host-networking \
    --priority "${PRIORITY}" \
    --leader-selection \
    --gpus 1 \
    --replicas 1 \
    --cluster "${CLUSTER}" \
    --budget ai2/oe-base \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
    --env-secret HF_TOKEN=RYAN_HF_TOKEN \
    --env-secret WANDB_API_KEY=RYAN_WANDB_API_KEY \
    --shared-memory 10GiB \
    --weka=oe-training-default:/data/input/ \
    --yes \
    -- $command


#    --install "pip install -r requirements.txt" \
#    --mount "src=weka,ref=oe-training-default,subpath=ryanwang,dst=/root" \
#    --conda-env "onboarding" \
#



