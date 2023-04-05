import torch as th


def get_model(name='bert-base-chinese', is_first_time=True):
    if is_first_time:
        return th.hub.load('huggingface/pytorch-transformers', 'model', name)
    else:
        return th.hub.load(th.hub.get_dir() + "/huggingface_pytorch-transformers_main", "model", name, source="local")
    

def get_tokenizer(name='bert-base-chinese', is_first_time=True):
    if is_first_time:
        return th.hub.load('huggingface/pytorch-transformers', 'tokenizer', name)
    else:
        return th.hub.load(th.hub.get_dir() + "/huggingface_pytorch-transformers_main", "tokenizer", name, source="local")