import os
import hydra
import json
import random
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import GPT2Tokenizer, BertTokenizer, AutoTokenizer, BloomTokenizerFast, GPTNeoXTokenizerFast, LlamaTokenizer
from intervention_models.intervention_model import load_model
from interventions.intervention import get_data
from interventions import three_operands
from tqdm import tqdm
import pickle
import sys

@hydra.main(config_path='conf', config_name='config')
def create_data(args: DictConfig):
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

    log_directory = args.data_dir
    os.makedirs(log_directory, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize Model and Tokenizer
    model = load_model(args)
    tokenizer_class = (GPT2Tokenizer if model.is_gpt2 or model.is_gptneo else
                       BertTokenizer if model.is_bert else
                       AutoTokenizer if model.is_gptj or model.is_flan or model.is_pythia or model.is_opt
                         or model.is_mistral or model.is_persimmon or model.is_mpt else
                       BloomTokenizerFast if model.is_bloom else
                       GPTNeoXTokenizerFast if model.is_neox else
                       LlamaTokenizer if model.is_llama else
                       None)
    if not tokenizer_class:
        raise Exception(f'Tokenizer for model {args.model} not found')

    if 'goat' in args.model:
        tokenizer_id = 'decapoda-research/llama-7b-hf'
    else:
        tokenizer_id = args.model

    if model.is_opt:
        tokenizer = tokenizer_class.from_pretrained(tokenizer_id, cache_dir=args.transformers_cache_dir, use_fast=False)
    elif model.is_bloom:
        tokenizer = tokenizer_class.from_pretrained(tokenizer_id, use_fast=True)
        print(f"Using tokenizer {tokenizer}")
    elif model.is_persimmon:
        tokenizer = tokenizer_class.from_pretrained("/home/aoq559/.cache/huggingface/hub/persimmon/")
        print(f"Using tokenizer {tokenizer}")
    else:
        tokenizer = tokenizer_class.from_pretrained(tokenizer_id, cache_dir=args.transformers_cache_dir)

    intervention_list = three_operands.get_arithmetic_data_three_operands(tokenizer, args)

    # check if the result generated from the model is correct
    def check_result(intervention):
        res_base_tok = intervention.res_base_tok[0]
        if not model.is_opt:
            res_base_tok2 = tokenizer.encode(' ' + intervention.res_base_string)
        res_alt_tok = intervention.res_alt_tok[0]
        if not model.is_opt:
            res_alt_tok2 = tokenizer.encode(' ' + intervention.res_alt_string)
        with torch.no_grad():
            input_id_base = intervention.base_string_tok.to(model.device)
            input_id_alt = intervention.alt_string_tok.to(model.device)

            # print(input_id_base[0].cpu().numpy().tolist())
            # print(input_id_alt[0].cpu().numpy().tolist())

            # print('Base string: ', tokenizer.decode(input_id_base[0].cpu().numpy().tolist()))
            # print('Alt string: ', tokenizer.decode(input_id_alt[0].cpu().numpy().tolist()))

            output_base = model.model.generate(input_id_base, max_new_tokens=1, do_sample=False)
            output_alt = model.model.generate(input_id_alt, max_new_tokens=1, do_sample=False)
            
            pred_base_tok = output_base[0, -1].cpu().numpy()
            pred_alt_tok = output_alt[0, -1].cpu().numpy()
            print(f"res_base_tok: {res_base_tok}")
            print(f"res_alt_tok: {res_alt_tok}")
            base_text = tokenizer.decode(output_base[0])
            pred_alt_text = tokenizer.decode(output_alt[0])
            pred_alt_string = tokenizer.decode(output_alt[0, -1])
            print(tokenizer.decode(input_id_base[0]))
            print("base", base_text)
            print("alt", pred_alt_text)
            print("real result and token:", tokenizer.decode(res_base_tok), res_base_tok)
            print("predicted result base and token:", tokenizer.decode(pred_base_tok), pred_base_tok)
            print("predicted result alt and token:", tokenizer.decode(pred_alt_tok), pred_alt_tok)

            if not model.is_opt:
                return (res_base_tok == pred_base_tok and res_alt_tok != pred_alt_tok and res_alt_tok2 != pred_alt_tok,
                        (res_base_tok == pred_base_tok) or (res_base_tok2 == pred_base_tok), 
                        res_alt_tok == pred_alt_tok,
                        [pred_alt_tok.tolist()],
                        pred_alt_string)
            if model.is_opt:
                return (res_base_tok == pred_base_tok and res_alt_tok != pred_alt_tok,
                        res_base_tok == pred_base_tok, 
                        res_alt_tok == pred_alt_tok,
                        [pred_alt_tok.tolist()],
                        pred_alt_string)
        
    correct_intervention = []
    progress = tqdm(total=len(intervention_list), desc='collecting data')
    counter = 0
    counter2 = 0

    for intervention in intervention_list:
        check1, check2, check_3, pred_alt_tok, pred_alt_string = check_result(intervention)
        if check1:
            intervention.set_predicted_alt_result(pred_alt_string=pred_alt_string, pred_res_alt_tok=pred_alt_tok)
            correct_intervention.append(intervention)
        if check2 and args.n_shots == 0:
            intervention.set_predicted_alt_result(pred_alt_string=pred_alt_string, pred_res_alt_tok=pred_alt_tok)
            correct_intervention.append(intervention)
        if check2:
            counter += 1
        if check_3:
            counter2 += 1

        progress.update()

    n_shots = str(args.n_shots)
    max_n = str(args.max_n)
    representation = str(args.representation)
    file_name = args.data_dir
    file_name += '/' + str(args.model)
    os.makedirs(file_name, exist_ok=True)
    file_name += '/intervention_' + n_shots + '_shots_' + 'max_' + max_n + '_' + representation
    file_name += '_further_templates' if args.extended_templates else ''
    file_name += '_reversed_fewshot' if args.reversed_fewshot else ''
    file_name += '_mpt2' if args.mpt_data_version_2 else ''
    file_name += '.pkl'

    with open(file_name, 'wb') as f:
        pickle.dump(correct_intervention, f)

    print(f"Out of {args.examples_per_template} interventions {counter} were right on the base prompt solely.")

    print(f"Out of {args.examples_per_template} examples {len(correct_intervention)} were right on the base prompt and wrong on the alternative prompt.")
    print(f"Out of {args.examples_per_template} examples {counter2} were right on the alternative prompt solely.")

    print("Saved to ", file_name)

if __name__ == '__main__':
    create_data()

    


