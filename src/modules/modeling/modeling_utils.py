import os

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch


def setup_model(path_to_model, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(path_to_model, **kwargs)
    print(f"imported model from {path_to_model}")
    return model.to("cuda")

def setup_model_torch(path_to_model, **kwargs):

    if (os.path.exists(path_to_model)):
        model = torch.load(path_to_model).to("cuda")
        print(f"imported model from {path_to_model}")
        return model
    else:
        raise FileNotFoundError(f"unknown model name at {path_to_model}")


#will call the garbage collector on the indefinite list of pointers given
def free_gpus(*args):
    import gc
    for arg in args:
        del arg
    torch.cuda.empty_cache()
    gc.collect()
