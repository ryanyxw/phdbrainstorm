import nemo_run as run

from nemo.collections import llm
from nemo import lightning as nl

from megatron.core.distributed import DistributedDataParallelConfig


def configure_recipe(nodes: int = 1, gpus_per_node: int = 8):
    # recipe = llm.nemotron3_4b.pretrain_recipe(
    #     dir="checkpoints/nemotron", # Path to store checkpoints
    #     name="nemotron_pretraining",
    #     tensor_parallelism=2,
    #     num_nodes=nodes,
    #     num_gpus_per_node=gpus_per_node,
    #     max_steps=100, # Setting a small value for the quickstart
    # )
    recipe = llm.llama3_8b.pretrain_recipe(
        dir="checkpoints/llama3", # Path to store checkpoints
        name="llama3_pretraining",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
    )

    recipe.trainer.val_check_interval = 100

    # strategy = nl.MegatronStrategy(
    #     tensor_model_parallel_size=4,
    #     pipeline_model_parallel_size=4,
    #     virtual_pipeline_model_parallel_size=None,
    #     context_parallel_size=2,
    #     sequence_parallel=True,
    #     gradient_as_bucket_view=True,
    #     ckpt_async_save=False,
    #     ckpt_parallel_load=True,
    #     ddp=run.Config(
    #         DistributedDataParallelConfig,
    #         check_for_nan_in_grad=True,
    #         grad_reduce_in_fp32=True,
    #         overlap_grad_reduce=True,
    #         overlap_param_gather=True,
    #         average_in_collective=True,  # Not supported for custom FSDP for now, need to be set to False if using FSDP
    #         data_parallel_sharding_strategy="optim_grads_params",  # For custom FSDP only
    #     ),
    #     fsdp=None,
    # )

    # recipe.trainer.strategy.tensor_model_parallel_size = 4
    # recipe.trainer.strategy.pipeline_model_parallel_size = 1
    recipe.trainer.strategy.ckpt_async_save = False
    return recipe

def local_executor_torchrun(nodes: int = 1, devices: int = 2) -> run.LocalExecutor:
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

    run.run(recipe, executor=executor, name="nemotron3_4b_pretraining")

# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    run_pretraining()