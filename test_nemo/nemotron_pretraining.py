import nemo_run as run
import nemo.lightning as nl
from collections.llm import PreTrainingDataModule
from nemo.collections import llm


def configure_recipe(nodes: int = 1, gpus_per_node: int = 8):
    recipe = llm.llama3_8b.pretrain_recipe(
        dir="checkpoints/llama3", # Path to store checkpoints
        name="llama3_pretraining",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
    )

    recipe.data = PreTrainingDataModule(
        paths=["path-to-data"],
        seq_length=8192,
        micro_batch_size=1,
        global_batch_size=512,
        dataset_kwargs={},
        split="95,3,2",
    )

    # Configure validation interval
    recipe.trainer.val_check_interval = 100
    
    # # Create ModelCheckpoint callback to save every 2 steps
    # checkpoint_callback = nl.ModelCheckpoint(
    #     every_n_train_steps=2,  # Save a checkpoint every 2 training steps
    #     save_top_k=-1,          # Save all checkpoints (no limit)
    #     dirpath="checkpoints/llama3",  # Directory to save checkpoints
    #     filename="checkpoint-{step:06d}",  # Checkpoint filename format with step number
    #     save_last=True,         # Always save the last checkpoint
    #     monitor=None            # No monitoring of validation metrics
    # )
    #
    # # Add the checkpoint callback to the trainer
    # if not hasattr(recipe.trainer, 'callbacks') or recipe.trainer.callbacks is None:
    #     recipe.trainer.callbacks = []
    # recipe.trainer.callbacks.append(checkpoint_callback)
    #

    recipe.trainer.strategy.pipeline_model_parallel_size = 2

    return recipe

def local_executor_torchrun(nodes: int = 1, devices: int = 8) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor

def run_pretraining():
    recipe = configure_recipe()
    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)

    run.run(recipe, executor=executor, name="llama_3_8b_pretraining")

# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    run_pretraining()