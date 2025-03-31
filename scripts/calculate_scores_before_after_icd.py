from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import yaml
from omegaconf import DictConfig, OmegaConf
from interventions import three_operands
from tqdm.notebook import tqdm
import numpy as np

from huggingface_hub import login

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.set_grad_enabled(False)
device = 'cuda'



def tokenize_whitespace(item, tokenizer):
    item_base = [tokenizer.bos_token_id] + tokenizer.encode(' ' + item.few_shots.lstrip() + item.base_string, 
                                                            add_special_tokens=False)
    item_alt = [tokenizer.bos_token_id] + tokenizer.encode(' ' + item.base_string.lstrip(),
                                                           add_special_tokens=False)
    whitespace = tokenizer.encode(' ', add_special_tokens=False)
    item_alt = whitespace * (len(item_base) - len(item_alt)) + item_alt
    return torch.tensor(item_base).unsqueeze(0).to(device), torch.tensor(item_alt).unsqueeze(0).to(device)


def tokenize_simple(item, tokenizer):
    item_base = [tokenizer.bos_token_id] + tokenizer.encode(' ' + item.few_shots.lstrip() + item.base_string, 
                                                            add_special_tokens=False)
    item_alt = [tokenizer.bos_token_id] + tokenizer.encode(' ' + item.base_string.lstrip(),
                                                           add_special_tokens=False)
    whitespace = tokenizer.encode(' ', add_special_tokens=False)
    return torch.tensor(item_base).unsqueeze(0).to(device), torch.tensor(item_alt).unsqueeze(0).to(device)


# The code for models for which we are not able to create the dataset
# model_name = 'Qwen/Qwen2-1.5B'
# # model_name = 'meta-llama/Llama-2-7b-hf'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# tokenizer_for_interventions = AutoTokenizer.from_pretrained('EleutherAI/pythia-6.9b-deduped-v0')
# conf = OmegaConf.load('conf/config.yaml')
# intervention_list = three_operands.get_arithmetic_data_three_operands(tokenizer_for_interventions, conf)

# accuracy_base = []
# accuracy_alt = []
# for item in tqdm(intervention_list):
#     base_string = item.few_shots + item.base_string + ' '
#     alt_string = item.alt_string + ' '

#     base_toks = tokenizer.encode(base_string)
#     alt_toks = tokenizer.encode(alt_string)
#     # padding_length = len(base_toks) - len(alt_toks)
#     # alt_toks = tokenizer.encode(' ') * (padding_length - 1) + [tokenizer.eos_token_id] + alt_toks
#     # assert(len(base_toks) == len(alt_toks))

#     input_id_base = torch.tensor(base_toks).unsqueeze(0).to(model.device)
#     input_id_alt = torch.tensor(alt_toks).unsqueeze(0).to(model.device)

#     output_base = model.generate(input_id_base, max_new_tokens=3, do_sample=False)[0, -3:].cpu().numpy()
#     output_alt = model.generate(input_id_alt, max_new_tokens=3, do_sample=False)[0, -3:].cpu().numpy()
#     output_base_str = tokenizer.decode(output_base)
#     output_alt_str = tokenizer.decode(output_alt)
#     correct_output_str = item.res_base_string

#     print(base_string, output_base_str, '#' * 5, alt_string, output_alt_str)
    
# #     try:
# #         accuracy_base.append(int(output_base_str) == int(correct_output_str))
# #     except:
# #         accuracy_base.append(0)
    
# #     try:
# #         accuracy_alt.append(int(output_alt_str) == int(correct_output_str))
# #     except:
# #         accuracy_alt.append(0)

# # print('Accuracy with ICD: ', np.mean(accuracy_base))
# # print('Accuracy without ICD: ', np.mean(accuracy_alt))

# The code for models for which we are able to create the dataset
# model_name = 'EleutherAI/pythia-6.9b-deduped-v0'
# model_name = 'EleutherAI/pythia-12b-deduped-v0'
# model_name = 'facebook/opt-2.7b'
# model_name = 'facebook/opt-6.7b'
# model_name = 'facebook/opt-13b'
# model_name = 'meta-llama/Llama-2-7b-hf'
# model_name = 'stabilityai/stablelm-base-alpha-7b'
# model_name = 'stabilityai/stablelm-base-alpha-3b'
model_name = 'microsoft/phi-2'

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

conf = OmegaConf.load('conf/config.yaml')
conf.model = model_name
conf.n_operands = 3
# conf.max_n = 9
# conf.n_operands = 2
intervention_list = three_operands.get_arithmetic_data_three_operands(tokenizer, conf)

def calc_accuracy(intervention_list, model):
    accuracy_base = []
    accuracy_alt = []

    accuracy_base_tok = []
    accuracy_alt_tok = []

    for item in tqdm(intervention_list):
        # input_id_base = item.base_string_tok.to(model.device)
        # input_id_alt = item.alt_string_tok.to(model.device)
        # input_id_base, input_id_alt = tokenize_whitespace(item, tokenizer)
        input_id_base, input_id_alt = tokenize_simple(item, tokenizer)

        output_base = model.generate(input_id_base, max_new_tokens=1, do_sample=False)[0, -1].cpu().numpy()
        output_alt = model.generate(input_id_alt, max_new_tokens=1, do_sample=False)[0, -1].cpu().numpy()
        output_base_str = tokenizer.decode(output_base)
        output_alt_str = tokenizer.decode(output_alt)
        correct_output_tok = item.res_base_tok[0]
        correct_output_str = item.res_base_string

        # print(output_base, output_alt, correct_output_tok)
        # correct_output_str = int(item.res_base_string)
        print('With icd: ', output_base_str, 'Without icd: ', output_alt_str, 'Correct: ', correct_output_str)
        print(tokenizer.decode(input_id_base[0]))
        print(tokenizer.decode(input_id_alt[0]))
        print('#' * 10)

        try:
            accuracy_base.append(int(output_base_str) == int(correct_output_str))
        except:
            accuracy_base.append(0)
        
        try:
            accuracy_alt.append(int(output_alt_str) == int(correct_output_str))
        except:
            accuracy_alt.append(0)
        
        accuracy_base_tok.append(output_base == correct_output_tok)
        accuracy_alt_tok.append(output_alt == correct_output_tok)

    return accuracy_base, accuracy_alt, accuracy_base_tok, accuracy_alt_tok

accuracy_base, accuracy_alt, accuracy_base_tok, accuracy_alt_tok = calc_accuracy(intervention_list, model)

print('Accuracy with icd, numbers equality', np.mean(accuracy_base))
print('Accuracy without icd, numbers equality', np.mean(accuracy_alt))

print('Accuracy with icd, tokens equality', np.mean(accuracy_base_tok))
print('Accuracy without icd, tokens equality', np.mean(accuracy_alt_tok))