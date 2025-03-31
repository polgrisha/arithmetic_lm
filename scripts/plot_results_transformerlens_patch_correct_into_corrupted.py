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
import copy
import os

import transformer_lens.utils as utils
from transformer_lens import ActivationCache, HookedTransformer
import transformer_lens.patching as patching
import seaborn as sns
import matplotlib.pyplot as plt

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.set_grad_enabled(False)
device='cuda'


model_name = 'EleutherAI/pythia-12b-deduped-v0'
model_name_lens = 'pythia-12b-deduped-v0'
# model_name = 'EleutherAI/pythia-6.9b-deduped-v0'
# model_name_lens = 'pythia-6.9b-deduped-v0'
# model_name_lens = 'facebook/opt-125m'
if 'facebook/opt' in model_name:
    SYMBOLS = ["alpha", "beta", "lambda", "delta", "omega", "mu", "nu", "pi", "chi", "number", "res", "x", "y", "z", "a", "b", "c"]
    WORD_NUMBERS = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"
    ]
elif 'EleutherAI/pythia' in model_name:
    SYMBOLS = ["alpha", "beta", "gamma", "delta", "mu", "nu", "chi", "theta", "sigma", "number", "res", "x", "y", "z", "a", "b", "c"]
    WORD_NUMBERS = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"
    ]
LABELS = ['operand_1', 'operator_1', 'operand_2', 'operator_2', 'operand_3', 'eq', 'res', 'operand_1', 'operator_1', 'operand_2', 'operator_2', 'operand_3', 'eq']

save_directory = '/mnt/qb/work/eickhoff/esx208/arithmetic-lm/data/patching_results_original_dataset_from_correct_into_corrupted_pythia_12b'

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


def tokenize_w_bos_and_whitespace(item, tokenizer):
    return torch.tensor([tokenizer.bos_token_id] + tokenizer.encode(' ' + item.few_shots.lstrip() + item.base_string)).unsqueeze(0).to(device)


def get_patched_result_counterfactual(item, item_counterfactual, model, tokenizer, activation_name='resid_pre'):
    clean_tokens = tokenize_w_bos_and_whitespace(item, tokenizer)
    corrupted_tokens = tokenize_w_bos_and_whitespace(item_counterfactual, tokenizer)
    
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits = model(corrupted_tokens)
    
    prediction_clean = torch.argmax(clean_logits.squeeze()[-1, :]).cpu()
    prediction_corrupted = torch.argmax(corrupted_logits.squeeze()[-1, :]).cpu()
    
    assert prediction_clean == item.res_base_tok[0]
    assert prediction_corrupted != item.res_base_tok[0]

    answer_token_indices = torch.tensor([[prediction_clean, prediction_corrupted]]).to(model.cfg.device)

    clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).cpu()
    corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).cpu()

    # def metric(logits, answer_token_indices=answer_token_indices):
    #     return (get_logit_diff(logits, answer_token_indices) - corrupted_logit_diff) / torch.abs(clean_logit_diff - corrupted_logit_diff)
    metric = partial(get_logit_diff, answer_token_indices=answer_token_indices)
    
    if activation_name == 'resid_pre':
        patched_logit_diff = patching.get_act_patch_resid_pre(model, corrupted_tokens, clean_cache, metric).cpu()
    elif activation_name == 'attn_layer':
        patched_logit_diff = patching.get_act_patch_attn_out(model, corrupted_tokens, clean_cache, metric).cpu()
    elif activation_name == 'mlp':
        patched_logit_diff = patching.get_act_patch_mlp_out(model, corrupted_tokens, clean_cache, metric).cpu()
    elif activation_name == 'head':
        patched_logit_diff = patching.get_act_patch_attn_head_out_all_pos(model, corrupted_tokens, clean_cache, metric).cpu()

    return patched_logit_diff, clean_logit_diff, corrupted_logit_diff, clean_tokens, corrupted_tokens


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


def replace_numbers_with(intervention_list, conf, tokenizer, operands=False, result = False, replacement = 'symbols'):
    counterfactual_intervention_list = []
    for intervention in intervention_list:
        # Create copy of intervention
        new_intervention = copy.deepcopy(intervention)
        
        # Get few shot example and tokenize it
        few_shot = new_intervention.few_shots
        few_shot_tokenized = tokenizer.encode(few_shot, add_special_tokens=False)        
        
        # Get positions of numbers
        number_positions = []
        for pos, item in enumerate(few_shot_tokenized):
            if tokenizer.decode(item).strip().isnumeric():
                number_positions.append(pos)
        
        number_positions_selected = []
        if operands:
            number_positions_selected += number_positions[:-1]
        if result:
            number_positions_selected += number_positions[-1:]
        
        # For all position in number positions randomly choose symbol to replace and 
        for pos in number_positions_selected:
            if replacement == 'symbols':
                few_shot_result_str = ' ' + random.choice(SYMBOLS)
                new_result_enc = tokenizer.encode(few_shot_result_str, add_special_tokens=False)[0]
                few_shot_tokenized[pos] = new_result_enc
            elif replacement == 'words':
                number = int(tokenizer.decode(few_shot_tokenized[pos]).strip())
                num_word = ' ' + WORD_NUMBERS[number]
                num_word_enc = tokenizer.encode(num_word, add_special_tokens=False)[0]
                few_shot_tokenized[pos] = num_word_enc
            elif replacement == 'random':
                if result and not operands:
                    few_shot_result_int = int(tokenizer.decode(few_shot_tokenized[pos]).strip())
                    new_result = random.randint(1, conf.max_n)
                    while new_result == few_shot_result_int:
                        new_result = random.randint(1, conf.max_n)
                    new_result_enc = tokenizer.encode(' ' + str(new_result), add_special_tokens=False)[0]
                    few_shot_tokenized[pos] = new_result_enc
                else:
                    raise NotImplementedError('Only supported for altering the result')
            else:
                raise NotImplementedError(replacement + ' ' + 'not implemented!')
        few_shot = tokenizer.decode(few_shot_tokenized)
        
        new_intervention.few_shots = few_shot
        print(new_intervention.few_shots)
        counterfactual_intervention_list.append(new_intervention)
    return counterfactual_intervention_list


def calc_accuracy(intervention_list, model, tokenizer):
    accuracy_base = []
    accuracy_alt = []

    accuracy_base_tok = []
    accuracy_alt_tok = []

    for item in tqdm(intervention_list):
        # input_id_base = item.base_string_tok.to(model.device)
        # input_id_base = torch.tensor([tokenizer.bos_token_id] + item.base_string_tok_list).unsqueeze(0).to(model.device)
        
        # For other models
        # input_id_base = torch.tensor(tokenizer.encode(' ' + item.few_shots + item.base_string)).unsqueeze(0).to(model.device)
        # input_id_alt = item.alt_string_tok.to(model.device)
        
        # For other models and for testing the results without padding
        # input_id_base = torch.tensor(tokenizer.encode(' ' + item.few_shots + item.base_string)).unsqueeze(0).to(device)
        # input_id_alt = torch.tensor(tokenizer.encode(' ' + item.base_string)).unsqueeze(0).to(device)
        
        # For pythia and for testing the results without padding
        input_id_base = torch.tensor([tokenizer.bos_token_id] + tokenizer.encode(' ' + item.few_shots.lstrip() + item.base_string)).unsqueeze(0).to(device)
        input_id_alt = torch.tensor([tokenizer.bos_token_id] + tokenizer.encode(' ' + item.base_string.lstrip())).unsqueeze(0).to(device)

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


def main():
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = HookedTransformer.from_pretrained_no_processing(
        model_name_lens,
        dtype=torch.float16
    )
    model.eval()
    conf = OmegaConf.load('conf/config.yaml')
    
    # Get original data obtained by Christian
    intervention_list = pickle.load(open('/mnt/qb/work/eickhoff/esx208/arithmetic-lm/data/intervention_1_shots_max_20_arabic_further_templates.pkl', 'rb'))

    # # Ensure that my loaded model got same predictions
    # corr_intervention_list = []
    # for item in intervention_list:
    #     input_id_base = item.base_string_tok.to(model.cfg.device)
    #     input_id_alt = item.alt_string_tok.to(model.cfg.device)

    #     output_base = model.generate(input_id_base, max_new_tokens=1, do_sample=False)[0, -1].cpu().numpy()
    #     output_alt = model.generate(input_id_alt, max_new_tokens=1, do_sample=False)[0, -1].cpu().numpy()
    #     corr_tok = item.res_base_tok[0]

    #     item.set_predicted_alt_result(pred_alt_string=tokenizer.decode([output_alt]), pred_res_alt_tok=output_alt)
    #     if output_base == corr_tok and output_alt != corr_tok:
    #         corr_intervention_list.append(item)
    
    # torch.save(corr_intervention_list, '/mnt/qb/work/eickhoff/esx208/arithmetic-lm/data/patching_results_original_dataset_from_correct_into_corrupted_pythia_12b/base_data_arabic_pythia_12b.pkl')
    # corr_intervention_list = torch.load('/mnt/qb/work/eickhoff/esx208/arithmetic-lm/data/patching_results_original_dataset_from_correct_into_corrupted_pythia_12b/base_data_arabic_pythia_12b.pkl')
    # print(f'Out of {len(intervention_list)} examples from provided data {len(corr_intervention_list)} left')
    
    # Create counterfactual datasets
    counterfactual_res_alpha = replace_numbers_with(intervention_list, conf, tokenizer, result=True, replacement='symbols')
    counterfactual_res_words = replace_numbers_with(intervention_list, conf, tokenizer, result=True, replacement='words')
    counterfactual_res_random = replace_numbers_with(intervention_list, conf, tokenizer, result=True, replacement='random')
    counterfactual_operands_words = replace_numbers_with(intervention_list, conf, tokenizer, operands = True, result=False, replacement='words')
    counterfactual_operands_alpha = replace_numbers_with(intervention_list, conf, tokenizer, operands = True, result=False, replacement='symbols')
    
    torch.save(counterfactual_res_alpha, os.path.join(save_directory, 'counterfactual_res_alpha.pkl'))
    torch.save(counterfactual_res_words, os.path.join(save_directory, 'counterfactual_res_words.pkl'))
    torch.save(counterfactual_res_random, os.path.join(save_directory, 'counterfactual_res_random.pkl'))
    torch.save(counterfactual_res_random, os.path.join(save_directory, 'counterfactual_res_random.pkl'))
    torch.save(counterfactual_operands_words, os.path.join(save_directory, 'counterfactual_operands_words.pkl'))
    torch.save(counterfactual_operands_alpha, os.path.join(save_directory, 'counterfactual_operands_alpha.pkl'))
    
    accuracy_base, accuracy_alt, accuracy_base_tok, accuracy_alt_tok = calc_accuracy(intervention_list, model)
    accuracy_base_alpha, accuracy_alt_alpha, accuracy_base_tok_alpha, accuracy_alt_tok_alpha = calc_accuracy(counterfactual_res_alpha, model)
    accuracy_base_words, accuracy_alt_words, accuracy_base_tok_words, accuracy_alt_tok_words = calc_accuracy(counterfactual_res_words, model)
    accuracy_base_random, accuracy_alt_random, accuracy_base_tok_random, accuracy_alt_tok_random = calc_accuracy(counterfactual_res_random, model)
    accuracy_base_operands_words, accuracy_alt_operands_words, accuracy_base_tok_operands_words, accuracy_alt_tok_operands_words = calc_accuracy(counterfactual_operands_words, model)
    accuracy_base_operands_alpha, accuracy_alt_operands_alpha, accuracy_base_tok_operands_alpha, accuracy_alt_tok_operands_alpha = calc_accuracy(counterfactual_operands_alpha, model)
    
    # Counterfactual results, operands, words
    
    # Patching Residual Stream
    all_resid_pre_act_patch_results = []
    all_resid_pre_clean_logit_diffs = []
    all_resid_pre_corrupted_logit_diffs = []
    all_resid_pre_clean_tokens = []
    for idx, (item, item_counterfactual) in enumerate(tqdm(zip(intervention_list, counterfactual_operands_words), total=len(intervention_list))):
        if accuracy_base[idx] and not accuracy_base_operands_words[idx]:
            patched_logit_diff, clean_logit_diff, corrupted_logit_diff, clean_tokens, corrupted_tokens = get_patched_result_counterfactual(item, item_counterfactual)
            all_resid_pre_act_patch_results.append(patched_logit_diff)
            all_resid_pre_clean_logit_diffs.append(clean_logit_diff)
            all_resid_pre_corrupted_logit_diffs.append(corrupted_logit_diff)
            all_resid_pre_clean_tokens.append(clean_tokens)
        
    torch.save(all_resid_pre_act_patch_results, os.path.join(save_directory, 'operands_words_resid_pre_logit_diff.pkl'))
    torch.save(all_resid_pre_clean_logit_diffs, os.path.join(save_directory, 'operands_words_clean_diff.pkl'))
    torch.save(all_resid_pre_corrupted_logit_diffs, os.path.join(save_directory, 'operands_words_corrupted_logit_diff.pkl'))
    
    patched_results_operands_operators = take_operands_and_operators_results(all_resid_pre_clean_tokens, all_resid_pre_act_patch_results, tokenizer)
    
    patching_effect = compute_pe(patched_results_operands_operators, all_resid_pre_clean_logit_diffs, all_resid_pre_corrupted_logit_diffs)
    sns.heatmap(
        patching_effect.T,
        cmap='RdBu',
        vmin=torch.min(patching_effect),
        vmax=torch.max(patching_effect),
        yticklabels=LABELS,
        annot=False,
        xticklabels=True,
        center=0
    )
    plt.title('Patching Effect of the Residual Stream')
    plt.ylabel('Layer')
    plt.savefig(os.path.join(save_directory, 'operands_words_patching_residual_stream.png'))
    
    # Counterfactual results, operands, symbols
    
    # Patching Residual Stream
    all_resid_pre_act_patch_results = []
    all_resid_pre_clean_logit_diffs = []
    all_resid_pre_corrupted_logit_diffs = []
    all_resid_pre_clean_tokens = []
    for idx, (item, item_counterfactual) in enumerate(tqdm(zip(intervention_list, counterfactual_operands_alpha), total=len(intervention_list))):
        if accuracy_base[idx] and not accuracy_base_operands_alpha[idx]:
            patched_logit_diff, clean_logit_diff, corrupted_logit_diff, clean_tokens, corrupted_tokens = get_patched_result_counterfactual(item, item_counterfactual)
            all_resid_pre_act_patch_results.append(patched_logit_diff)
            all_resid_pre_clean_logit_diffs.append(clean_logit_diff)
            all_resid_pre_corrupted_logit_diffs.append(corrupted_logit_diff)
            all_resid_pre_clean_tokens.append(clean_tokens)
        
    torch.save(all_resid_pre_act_patch_results, os.path.join(save_directory, 'operands_alpha_resid_pre_logit_diff.pkl'))
    torch.save(all_resid_pre_clean_logit_diffs, os.path.join(save_directory, 'operands_alpha_clean_diff.pkl'))
    torch.save(all_resid_pre_corrupted_logit_diffs, os.path.join(save_directory, 'operands_alpha_corrupted_logit_diff.pkl'))
    
    patched_results_operands_operators = take_operands_and_operators_results(all_resid_pre_clean_tokens, all_resid_pre_act_patch_results, tokenizer)
    
    patching_effect = compute_pe(patched_results_operands_operators, all_resid_pre_clean_logit_diffs, all_resid_pre_corrupted_logit_diffs)
    sns.heatmap(
        patching_effect.T,
        cmap='RdBu',
        vmin=torch.min(patching_effect),
        vmax=torch.max(patching_effect),
        yticklabels=LABELS,
        annot=False,
        xticklabels=True,
        center=0
    )
    plt.title('Patching Effect of the Residual Stream')
    plt.ylabel('Layer')
    plt.savefig(os.path.join(save_directory, 'operands_alpha_patching_residual_stream.png'))

if __name__ == "__main__":
    main()
