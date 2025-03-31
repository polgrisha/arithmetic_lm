from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import yaml
from omegaconf import DictConfig, OmegaConf
from interventions import three_operands
from tqdm.notebook import tqdm
import numpy as np
from omegaconf import DictConfig, OmegaConf
from interventions import three_operands

from huggingface_hub import login

seed = 0
random.seed(seed)
torch.manual_seed(seed)

torch.set_grad_enabled(False)


# model_name = 'Qwen/Qwen2-1.5B'
# # model_name = 'meta-llama/Llama-2-7b-hf'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )
tokenizer_qwen_2 = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B')
tokenizer_llama_2 = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
tokenizer_pythia = AutoTokenizer.from_pretrained('EleutherAI/pythia-6.9b-deduped-v0')
tokenizer_mistral = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.3')
tokenizer_opt = AutoTokenizer.from_pretrained('facebook/opt-6.7b')

conf = OmegaConf.load('conf/config.yaml')
conf.model = 'meta-llama/Llama-2-7b-hf'
conf.max_n = 9
intervention_list = three_operands.get_arithmetic_data_three_operands(tokenizer_llama_2, conf)

# conf = OmegaConf.load('conf/config.yaml')
# conf.model = 'facebook/opt-6.7b'
# print(conf.model)
# intervention_list = three_operands.get_arithmetic_data_three_operands(tokenizer_opt, conf)
