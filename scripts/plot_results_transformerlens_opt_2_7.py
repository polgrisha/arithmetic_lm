import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import yaml
from omegaconf import DictConfig, OmegaConf
from interventions import three_operands
from tqdm.notebook import tqdm
import numpy as np
from functools import partial
import pickle

import transformer_lens.utils as utils
from transformer_lens import ActivationCache, HookedTransformer
import transformer_lens.patching as patching
import seaborn as sns
import matplotlib.pyplot as plt

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.set_grad_enabled(False)
device = 'cuda'

LABELS = ['operand_1', 'operator_1', 'operand_2', 'operator_2', 'operand_3', 'eq', 'res', 'operand_1', 'operator_1', 'operand_2', 'operator_2', 'operand_3', 'eq']

# save_directory = '/mnt/qb/work/eickhoff/esx208/arithmetic-lm/data/patching_results_opt_2_7b/'
# save_directory = '/mnt/qb/work/eickhoff/esx208/arithmetic-lm/data/patching_results_opt_2_7b_whitespace'
save_directory = '/mnt/qb/work/eickhoff/esx208/arithmetic-lm/data/patching_results_opt_2_7b_whitespace_no_bos_base'


def get_logit_diff(logits, answer_token_indices):
    if len(logits.shape)==3:
        # Get final logits only
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()


def get_patched_result(item, model, activation_name='resid_pre'):
    clean_logits, clean_cache = model.run_with_cache(item.base_string_tok)
    corrupted_logits = model(item.alt_string_tok)

    answer_token_indices = torch.tensor([[item.res_base_tok[0], item.pred_res_alt_tok]]).to(model.cfg.device)

    clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).cpu()
    corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).cpu()

    # def metric(logits, answer_token_indices=answer_token_indices):
    #     return (get_logit_diff(logits, answer_token_indices) - corrupted_logit_diff) / torch.abs(clean_logit_diff - corrupted_logit_diff)
    metric = partial(get_logit_diff, answer_token_indices=answer_token_indices)
    
    if activation_name == 'resid_pre':
        patched_logit_diff = patching.get_act_patch_resid_pre(model, item.alt_string_tok.to(model.cfg.device), clean_cache, metric).cpu()
    elif activation_name == 'attn_layer':
        patched_logit_diff = patching.get_act_patch_attn_out(model, item.alt_string_tok.to(model.cfg.device), clean_cache, metric).cpu()
    elif activation_name == 'mlp':
        patched_logit_diff = patching.get_act_patch_mlp_out(model, item.alt_string_tok.to(model.cfg.device), clean_cache, metric).cpu()
    elif activation_name == 'head':
        patched_logit_diff = patching.get_act_patch_attn_head_out_all_pos(model, item.alt_string_tok.to(model.cfg.device), clean_cache, metric).cpu()

    return patched_logit_diff, clean_logit_diff, corrupted_logit_diff


def take_operands_and_operators_results(intervention_list, patched_results, tokenizer):
    all_operators = {'+', '-', '*', 'times', 'minus', 'plus'}
    all_equal_signs = {'=', 'is'}
    
    patched_results_operands_and_operators = []
    for idx, item in enumerate(intervention_list[:len(patched_results)]):
        curr_tokens = [tokenizer.decode(token) for token in item.base_string_tok[0]]
        considered_positions = []
        for position, token in enumerate(curr_tokens):
            if token.strip() in all_operators or token.strip() in all_equal_signs or token.strip().isnumeric():
                considered_positions.append(position)
        patched_results_operands_and_operators.append(patched_results[idx][:, considered_positions])
    return patched_results_operands_and_operators


def compute_macro_mean(patched_results, clean_logit_diffs, corrupted_logit_diffs):
    return (torch.mean(torch.stack(patched_results), dim=0) - torch.tensor(corrupted_logit_diffs).mean()) / (torch.tensor(clean_logit_diffs).mean() - torch.tensor(corrupted_logit_diffs).mean())


def compute_micro_mean(patched_results, clean_logit_diffs, corrupted_logit_diffs):
    # return (torch.mean(torch.stack(patched_results), dim=0) - torch.mean(corrupted_logit_diffs)) / torch.abs(torch.mean(clean_logit_diffs) - torch.mean(corrupted_logit_diffs))
    return torch.mean((torch.stack(patched_results) - torch.tensor(corrupted_logit_diffs)[:, None, None]) / (torch.tensor(clean_logit_diffs)[:, None, None] - torch.tensor(corrupted_logit_diffs)[:, None, None]), dim=0)


def compute_pe(patched_results, clean_logit_diffs, corrupted_logit_diffs):
    return torch.mean(torch.stack(patched_results) - torch.tensor(corrupted_logit_diffs)[:, None, None], dim=0)


def tokenize_w_bos_and_whitespace(item, tokenizer):
    return torch.tensor([tokenizer.bos_token_id] + tokenizer.encode(' ' + item.few_shots.lstrip() + item.base_string)).unsqueeze(0).to(device)


def tokenize(item, tokenizer):
    item_base = tokenizer.encode(' ' + item.few_shots.lstrip() + item.base_string)
    item_alt = tokenizer.encode(' ' + item.base_string.lstrip())
    item_alt = [tokenizer.pad_token_id] * (len(item_base) - len(item_alt)) + item_alt
    return torch.tensor(item_base).unsqueeze(0).to(device), torch.tensor(item_alt).unsqueeze(0).to(device)


def tokenize_whitespace(item, tokenizer):
    item_base = tokenizer.encode(' ' + item.few_shots.lstrip() + item.base_string)
    item_alt = tokenizer.encode(' ' + item.base_string.lstrip())
    whitespace = tokenizer.encode(' ', add_special_tokens=False)
    item_alt = whitespace * (len(item_base) - len(item_alt)) + item_alt
    return torch.tensor(item_base).unsqueeze(0).to(device), torch.tensor(item_alt).unsqueeze(0).to(device)


def tokenize_whitespace_no_bos_base(item, tokenizer):
    item_base = tokenizer.encode(' ' + item.few_shots.lstrip() + item.base_string, add_special_tokens=False)
    item_alt = tokenizer.encode(' ' + item.base_string.lstrip())
    whitespace = tokenizer.encode(' ', add_special_tokens=False)
    item_alt = whitespace * (len(item_base) - len(item_alt)) + item_alt
    return torch.tensor(item_base).unsqueeze(0).to(device), torch.tensor(item_alt).unsqueeze(0).to(device)


def main():
    # model_name = 'EleutherAI/pythia-12b-deduped-v0'
    # model_name_lens = 'pythia-12b-deduped-v0'
    # model_name = 'EleutherAI/pythia-6.9b-deduped-v0'
    # model_name_lens = 'pythia-6.9b-deduped-v0'
    # model_name_lens = 'facebook/opt-125m'
    model_name = 'facebook/opt-2.7b'
    model_name_lens = 'opt-2.7b'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = HookedTransformer.from_pretrained_no_processing(
        model_name_lens,
        dtype=torch.float16
    )
    model.eval()
    conf = OmegaConf.load('conf/config.yaml')

    intervention_list = torch.load('/mnt/qb/work/eickhoff/esx208/arithmetic-lm/data/information_flow_routes_results_opt_2_7b/base_data_arabic_opt_2_7.pkl')

    corr_intervention_list = []
    for item in intervention_list:
        # input_id_base, input_id_alt = tokenize(item, tokenizer)
        # input_id_base, input_id_alt = tokenize_whitespace(item, tokenizer)
        input_id_base, input_id_alt = tokenize_whitespace_no_bos_base(item, tokenizer)

        output_base = model.generate(input_id_base, max_new_tokens=1, do_sample=False)[0, -1].cpu().numpy()
        output_alt = model.generate(input_id_alt, max_new_tokens=1, do_sample=False)[0, -1].cpu().numpy()
        corr_tok = item.res_base_tok[0]

        item.set_predicted_alt_result(pred_alt_string=tokenizer.decode([output_alt]), pred_res_alt_tok=output_alt)
        item.base_string_tok = input_id_base
        item.alt_string_tok = input_id_alt
        if output_base == corr_tok and output_alt != corr_tok:
            corr_intervention_list.append(item)
        
    # # torch.save(corr_intervention_list, '/mnt/qb/work/eickhoff/esx208/arithmetic-lm/data/patching_results_original_dataset_pythia_12b/base_data_arabic_pythia_12b.pkl')
    # corr_intervention_list = torch.load('/mnt/qb/work/eickhoff/esx208/arithmetic-lm/data/patching_results_original_dataset_pythia_12b/base_data_arabic_pythia_12b.pkl')
    print(f'Out of {len(intervention_list)} examples from provided data {len(corr_intervention_list)} left')
    
    # # Patching Residual Stream
    # all_resid_pre_act_patch_results = []
    # all_resid_pre_clean_logit_diffs = []
    # all_resid_pre_corrupted_logit_diffs = []
    # for item in tqdm(corr_intervention_list):
    #     patched_logit_diff, clean_logit_diff, corrupted_logit_diff = get_patched_result(item, model)
    #     all_resid_pre_act_patch_results.append(patched_logit_diff)
    #     all_resid_pre_clean_logit_diffs.append(clean_logit_diff)
    #     all_resid_pre_corrupted_logit_diffs.append(corrupted_logit_diff)
        
    # torch.save(all_resid_pre_act_patch_results, os.path.join(save_directory, 'resid_pre_logit_diff.pkl'))
    # torch.save(all_resid_pre_clean_logit_diffs, os.path.join(save_directory, 'resid_pre_clean_logit_diff.pkl'))
    # torch.save(all_resid_pre_corrupted_logit_diffs, os.path.join(save_directory, 'resid_pre_corrupted_logit_diff.pkl'))
    # all_resid_pre_act_patch_results = torch.load(os.path.join(save_directory, 'resid_pre_logit_diff.pkl'))
    # all_resid_pre_clean_logit_diffs = torch.load(os.path.join(save_directory, 'resid_pre_clean_logit_diff.pkl'))
    # all_resid_pre_corrupted_logit_diffs = torch.load(os.path.join(save_directory, 'resid_pre_corrupted_logit_diff.pkl'))
    
    # patched_results_operands_operators = take_operands_and_operators_results(corr_intervention_list, all_resid_pre_act_patch_results, tokenizer)
    
    # patching_effect = compute_pe(patched_results_operands_operators, all_resid_pre_clean_logit_diffs, all_resid_pre_corrupted_logit_diffs)
    # plt.figure()
    # sns.heatmap(
    #     patching_effect.T,
    #     cmap='RdBu',
    #     vmin=torch.min(patching_effect),
    #     vmax=torch.max(patching_effect),
    #     yticklabels=LABELS,
    #     annot=False,
    #     xticklabels=True,
    #     center=0
    # )
    
    # plt.title('Patching Effect of the Residual Stream')
    # plt.ylabel('Layer')
    # plt.savefig(os.path.join(save_directory, 'patching_residual_stream.png'))
    
    # patching_effect_normalized = compute_macro_mean(patched_results_operands_operators, all_resid_pre_clean_logit_diffs, all_resid_pre_corrupted_logit_diffs)
    # plt.figure()
    # sns.heatmap(
    #     patching_effect_normalized.T,
    #     cmap='RdBu',
    #     vmin=-1,
    #     vmax=1,
    #     yticklabels=LABELS,
    #     annot=False,
    #     xticklabels=True,
    #     center=0
    # )
    
    # plt.title('Patching Effect of the Residual Stream')
    # plt.ylabel('Layer')
    # plt.savefig(os.path.join(save_directory, 'patching_residual_stream_macro_mean.png'))
    
    # patching_effect_normalized = compute_micro_mean(patched_results_operands_operators, all_resid_pre_clean_logit_diffs, all_resid_pre_corrupted_logit_diffs)
    # plt.figure()
    # sns.heatmap(
    #     patching_effect_normalized.T,
    #     cmap='RdBu',
    #     vmin=-1,
    #     vmax=1,
    #     yticklabels=LABELS,
    #     annot=False,
    #     xticklabels=True,
    #     center=0
    # )
    
    # plt.title('Patching Effect of the Residual Stream')
    # plt.ylabel('Layer')
    # plt.savefig(os.path.join(save_directory, 'patching_residual_stream_micro_mean.png'))
    
    # Patching Attention Layers
    # attn_layer_act_patch_results = []
    # attn_layer_clean_logit_diffs = []
    # attn_layer_corrupted_logit_diffs = []
    # for item in tqdm(corr_intervention_list):
    #     patched_logit_diff, clean_logit_diff, corrupted_logit_diff = get_patched_result(item, model, activation_name='attn_layer')
    #     attn_layer_act_patch_results.append(patched_logit_diff)
    #     attn_layer_clean_logit_diffs.append(clean_logit_diff)
    #     attn_layer_corrupted_logit_diffs.append(corrupted_logit_diff)
        
    # torch.save(attn_layer_act_patch_results, os.path.join(save_directory, 'attn_layer_logit_diff.pkl'))
    # torch.save(attn_layer_clean_logit_diffs, os.path.join(save_directory, 'attn_layer_clean_logit_diff.pkl'))
    # torch.save(attn_layer_corrupted_logit_diffs, os.path.join(save_directory, 'attn_layer_corrupted_logit_diff.pkl'))
    attn_layer_act_patch_results = torch.load(os.path.join(save_directory, 'attn_layer_logit_diff.pkl'))
    attn_layer_clean_logit_diffs = torch.load(os.path.join(save_directory, 'attn_layer_clean_logit_diff.pkl'))
    attn_layer_corrupted_logit_diffs = torch.load(os.path.join(save_directory, 'attn_layer_corrupted_logit_diff.pkl'))

    attn_layer_patched_results_operands_operators = take_operands_and_operators_results(corr_intervention_list, attn_layer_act_patch_results, tokenizer)
    
    print(len(attn_layer_patched_results_operands_operators))
    print(attn_layer_patched_results_operands_operators[0].shape)
    
    patching_effect = compute_pe(attn_layer_patched_results_operands_operators, attn_layer_clean_logit_diffs, attn_layer_corrupted_logit_diffs)
    # plt.figure()
    # sns.heatmap(
    #     patching_effect.T,
    #     cmap='RdBu',
    #     vmin=torch.min(patching_effect),
    #     vmax=torch.max(patching_effect),
    #     yticklabels=LABELS,
    #     annot=False,
    #     xticklabels=True,
    #     center=0
    # )
    # plt.title('Patching Effect of Attention Layer')
    # plt.ylabel('Layer')
    # plt.savefig(os.path.join(save_directory, 'patching_attention_layer.png'))
    
    print(patching_effect.shape)
    
    plt.figure()
    sns.heatmap(
        patching_effect.T[:7, :],
        cmap='Blues',
        vmin=torch.min(patching_effect.T[:7, :]),
        vmax=torch.max(patching_effect.T[:7, :]),
        yticklabels=LABELS[:7],
        annot=False,
        xticklabels=True,
        center=0
    )
    plt.title('Patching Effect of Attention Layer')
    plt.ylabel('Layer')
    plt.savefig(os.path.join(save_directory, 'patching_attention_layer_only_icd.png'))
    
    # Patching MLPs
    # mlp_act_patch_results = []
    # mlp_clean_logit_diffs = []
    # mlp_corrupted_logit_diffs = []
    # for item in tqdm(corr_intervention_list):
    #     patched_logit_diff, clean_logit_diff, corrupted_logit_diff = get_patched_result(item, model, activation_name='mlp')
    #     mlp_act_patch_results.append(patched_logit_diff)
    #     mlp_clean_logit_diffs.append(clean_logit_diff)
    #     mlp_corrupted_logit_diffs.append(corrupted_logit_diff)
    
    # torch.save(mlp_act_patch_results, os.path.join(save_directory, 'mlp_logit_diff.pkl'))
    # torch.save(mlp_clean_logit_diffs, os.path.join(save_directory, 'mlp_clean_logit_diff.pkl'))
    # torch.save(mlp_corrupted_logit_diffs, os.path.join(save_directory, 'mlp_corrupted_logit_diff.pkl'))
    mlp_act_patch_results = torch.load(os.path.join(save_directory, 'mlp_logit_diff.pkl'))
    mlp_clean_logit_diffs = torch.load(os.path.join(save_directory, 'mlp_clean_logit_diff.pkl'))
    mlp_corrupted_logit_diffs = torch.load(os.path.join(save_directory, 'mlp_corrupted_logit_diff.pkl'))
    
    mlp_patched_results_operands_operators = take_operands_and_operators_results(corr_intervention_list, mlp_act_patch_results, tokenizer)
    
    patching_effect = compute_pe(mlp_patched_results_operands_operators, mlp_clean_logit_diffs, mlp_corrupted_logit_diffs)
    # plt.figure()
    # sns.heatmap(
    #     patching_effect.T,
    #     cmap='RdBu',
    #     vmin=torch.min(patching_effect),
    #     vmax=torch.max(patching_effect),
    #     yticklabels=LABELS,
    #     annot=False,
    #     xticklabels=True,
    #     center=0
    # )
    # plt.title('Patching Effect of the MLP')
    # plt.ylabel('Layer')
    # plt.savefig(os.path.join(save_directory, 'patching_mlp_original_data_transposed.png'))
    
    print(patching_effect.shape)
    
    plt.figure()
    sns.heatmap(
        patching_effect.T[:7,:],
        cmap='Blues',
        vmin=torch.min(patching_effect.T[:7,:]),
        vmax=torch.max(patching_effect.T[:7,:]),
        yticklabels=LABELS[:7],
        annot=False,
        xticklabels=True,
        center=0
    )
    plt.title('Patching Effect of the MLP')
    plt.ylabel('Layer')
    plt.savefig(os.path.join(save_directory, 'patching_mlp_only_icd.png'))
    

if __name__ == "__main__":
    main()


