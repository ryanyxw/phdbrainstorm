# list of directories of deepspeed runs
RUN_DIRS=(
    "pubmed_100k_BS-16"
    "pubmed_100k_BS-32"
    "pubmed_100k_BS-64"
    "pubmed_100k_BS-128"
)

# loop through the directories
for RUN_DIR in "${RUN_DIRS[@]}"; do
    echo "Processing directory: $RUN_DIR"

    for ckpt in "$RUN_DIR"/global_step*; do
        if compgen -G "$ckpt/*.bin" > /dev/null; then
            echo "Skipping $ckpt (already has merged bin files)"
        else
            echo "Converting shards in $ckpt ..."
#            python ${RUN_DIR}/zero_to_fp32.py $ckpt $ckpt
        fi
    done
done