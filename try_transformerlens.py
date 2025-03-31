import os
import hydra
import json
import random
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import GPT2Tokenizer, BertTokenizer, AutoTokenizer, BloomTokenizerFast, GPTNeoXTokenizerFast, LlamaTokenizer
from intervention_models.intervention_model import load_model
from interventions.intervention import get_data
import pickle
import sys
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
    HookedRootModule,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix, HookedTransformerConfig, ActivationCache
import transformer_lens.patching as patching

# Import stuff
import torch
import torch.nn as nn
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm

from jaxtyping import Float
from functools import partial

def logits_to_ave_logit_diff(
    logits: Float[torch.Tensor, "batch seq d_vocab"],
    answer_tokens,
    per_prompt: bool = False
):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    # Only the final logits are relevant for the answer
    final_logits: Float[torch.Tensor, "batch d_vocab"] = logits[:, -1, :]
    # Get the logits corresponding to the indirect object / subject tokens respectively
    print(f"final_logits.shape: {final_logits.shape}")
    print(f"answer_tokens.shape: {answer_tokens.shape}")
    answer_logits: Float[torch.Tensor, "batch 2"] = final_logits.gather(dim=-1, index=answer_tokens)
    # Find logit difference
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()


@hydra.main(config_path='conf', config_name='config')
def run_experiment(args: DictConfig):
    print(OmegaConf.to_yaml(args))
    print("Model:", args.model)

    print('args.intervention_type', args.intervention_type)

    if 'llama_models_hf/7B' in args.model:
        model_str = 'llama7B'
    elif 'llama_models_hf/13B' in args.model:
        model_str = 'llama13B'
    elif 'llama_models_hf/30B' in args.model:
        model_str = 'llama30B'
    elif 'alpaca' in args.model:
        model_str = 'alpaca'
    else:
        model_str = args.model

    # initialize logging
    log_directory = args.output_dir
    log_directory += f'/{model_str}'
    if args.model_ckpt:
        ckpt_name = '_'.join(args.model_ckpt.split('/')[5:9])
        log_directory += f'_from_ckpt_{ckpt_name}'
    log_directory += f'/n_operands{args.n_operands}'
    log_directory += f'/template_type{args.template_type}'
    log_directory += f'/max_n{args.max_n}'
    log_directory += f'/n_shots{args.n_shots}'
    log_directory += f'/examples_n{args.examples_per_template}'
    log_directory += f'/seed{args.seed}'
    print(f'log_directory: {log_directory}')
    os.makedirs(log_directory, exist_ok=True)
    wandb_name = ('random-' if args.random_weights else '')
    wandb_name += f'{model_str}'
    wandb_name += f' -p {args.template_type}'
    wandb.init(project='mathCMA', name=wandb_name, notes='', dir=log_directory,
               settings=wandb.Settings(start_method='fork'), mode=args.wandb_mode)
    args_to_log = dict(args)
    args_to_log['out_dir'] = log_directory
    print("\n" + json.dumps(str(args_to_log), indent=4) + "\n")
    wandb.config.update(args_to_log)
    del args_to_log

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize Model and Tokenizer
    #model = load_model(args)
    torch.set_grad_enabled(False)

    device = 'cuda'

    model = transformer_lens.HookedTransformer.from_pretrained('pythia-12b', device=device, 
                                                               n_devices=7, move_to_device=True)
    model.set_use_attn_result(True)
    model.set_use_split_qkv_input(True)

    path_to_data = os.path.join(args.data_dir, 
                                'intervention_' + str(args.n_shots) + '_shots_max_' + str(args.max_n) + '_' + args.representation + '_further_templates' + '.pkl')
    with open(path_to_data, 'rb') as f:
        intervention_list = pickle.load(f)
    print("Loaded data from", path_to_data)

    if args.debug_run:
        intervention_list = intervention_list[:2]

    for intervention in intervention_list:
        with torch.no_grad():
            clean_tokens = intervention.base_string_tok.to(device)
            flipped_tokens = intervention.alt_string_tok.to(device)
            clean_logits, clean_cache = model.run_with_cache(clean_tokens)
            flipped_logits, flipped_cache = model.run_with_cache(flipped_tokens)
            answer_tokens = torch.tensor([intervention.res_base_tok[0], intervention.res_alt_tok[0]])
            answer_tokens = answer_tokens.unsqueeze(0)
            answer_tokens = answer_tokens.to('cuda:7')

            clean_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens)
            flipped_logit_diff = logits_to_ave_logit_diff(flipped_logits, answer_tokens)

            print(
                "Clean string 0:    ", model.to_string(clean_tokens[0]), "\n"
                "Flipped string 0:", model.to_string(flipped_tokens[0]))
            print(f"Clean logit diff: {clean_logit_diff:.4f}")
            print(f"Flipped logit diff: {flipped_logit_diff:.4f}")
            del clean_tokens, flipped_tokens, clean_logits, flipped_logits, clean_cache, flipped_cache, answer_tokens, clean_logit_diff, flipped_logit_diff
    

    

if __name__ == '__main__':
    run_experiment()



