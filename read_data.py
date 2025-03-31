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
from interventions import three_operands
from tqdm import tqdm
import pickle
import sys
import copy
from utils.number_utils import convert_to_words


@hydra.main(config_path='conf', config_name='config')
def test(args: DictConfig):
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

    n_shots = str(args.n_shots)
    max_n = str(args.max_n)
    representation = args.representation
    f_name = args.data_dir
    f_name += '/' + str(args.model)

    f_name += '/intervention_' + n_shots + '_shots_' + 'max_' + max_n + '_' + representation
    f_name += '_further_templates' if args.extended_templates else ''
    f_name += '_reversed_fewshot' if args.reversed_fewshot else ''
    f_name += '_mpt2' if args.mpt_data_version_2 else ''

    with open(f_name + '.pkl', 'rb') as f:
        data = pickle.load(f)
    if not args.counterfactual:

        # # Initialize Model and Tokenizer
        model = load_model(args)
        tokenizer_class = (GPT2Tokenizer if model.is_gpt2 or model.is_gptneo or model.is_opt else
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
    elif args.counterfactual:
        tokenizer_class = AutoTokenizer
        tokenizer_id = args.model

    tokenizer = tokenizer_class.from_pretrained(tokenizer_id, cache_dir=args.transformers_cache_dir)
    if not args.counterfactual:
        model.create_vocab_subset(tokenizer, args)
    counter = 0

    counterfactual_list = []
    words_to_n = {convert_to_words(str(i)): i for i in range(args.max_n + 1)}
    equation_position_operands={"({x}+{y}+{z})": [0, 2, 4],
    "({x}+{y} + {z})": [3, 5, 7], 
    "({x} -{y}-{z})": [3, 5, 7], 
    "({x}*{y} * {z})": [3, 5, 7], 
    "({x} * {y} * {z})": [3, 5, 7], 
    "({x}*{y}*{z})": [0, 2, 4], 
    "(({x}-{y})*{z})": [4, 6, 9]}

    for intervention in data:
        base_input_tok = intervention.base_string_tok
        alt_input_tok = intervention.alt_string_tok
        #print(f"Alt input length: {len(alt_input_tok[0])}")
        with torch.no_grad():
            if args.counterfactual and args.representation == "arabic" and not args.extended_templates:
                new_intervention = copy.deepcopy(intervention)
                if not args.counterfactual_symbol_result:
                    few_shot_result = tokenizer.decode(new_intervention.base_string_tok[0][7])
                    few_shot_result_int = int(few_shot_result)
                    few_shot_result_int += 5
                    few_shot_result_str = str(few_shot_result_int)
                    if not args.consistent_counterfactual:
                        few_shot_result_str = convert_to_words(str(few_shot_result_int))
                elif args.counterfactual_symbol_result:
                    symbols = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "number", "result", "x", "y", "z", "a", "b", "c"]
                    # draw a random element from symbols
                    few_shot_result_str = random.choice(symbols)
                    few_shot_result_str = ' ' + few_shot_result_str                  
                few_shot_result_enc = tokenizer.encode(few_shot_result_str)[0]
                new_intervention.base_string_tok[0][7] = few_shot_result_enc
                new_few_shot_string = tokenizer.decode(new_intervention.base_string_tok[0][:intervention.len_few_shots-1])
                new_intervention.few_shots = new_few_shot_string
                new_intervention.base_string_tok_list = new_intervention.base_string_tok[0].numpy().tolist()
                counterfactual_list.append(new_intervention)
            
            elif args.counterfactual and args.representation == "arabic" and args.extended_templates:
                new_intervention = copy.deepcopy(intervention)
                few_shot_result = tokenizer.decode(new_intervention.base_string_tok[0][new_intervention.len_few_shots - 3])[1:]
                if not args.counterfactual_symbol_result and not args.counterfactual_symbol_operands:
                    few_shot_result_int = int(few_shot_result)
                    new_result = random.randint(1, args.max_n)
                    while new_result == few_shot_result_int:
                        new_result = random.randint(1, args.max_n)
                    new_result_str = str(new_result)
                    if not args.consistent_counterfactual:
                        new_result_str = convert_to_words(new_result_str)
                    new_result_str = ' ' + new_result_str
                    new_result_enc = tokenizer.encode(new_result_str)[0]
                elif args.counterfactual_symbol_result and not args.counterfactual_symbol_operands:
                    symbols = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", 
                               "eta", "theta", "iota", "number", "result", 
                               "x", "y", "z", "a", "b", "c"]
                    # draw a random element from symbols
                    few_shot_result_str = random.choice(symbols)
                    few_shot_result_str = ' ' + few_shot_result_str
                    new_result_enc = tokenizer.encode(few_shot_result_str)[0]
                elif args.counterfactual_symbol_operands:
                    symbols = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", 
                               "eta", "theta", "iota", "number", "result", 
                               "x", "y", "z", "a", "b", "c"]
                    position_operands = equation_position_operands[new_intervention.equation]
                    for pos in position_operands:
                        new_operand_str = random.choice(symbols)
                        new_operand_str = ' ' + new_operand_str
                        new_operand_enc = tokenizer.encode(new_operand_str)[0]
                        new_intervention.base_string_tok[0][pos] = new_operand_enc
                    new_few_shot_string = tokenizer.decode(new_intervention.base_string_tok[0][:intervention.len_few_shots-1])
                    new_intervention.few_shots = new_few_shot_string
                    new_intervention.base_string_tok_list = new_intervention.base_string_tok[0].numpy().tolist()
                    new_result_enc = new_intervention.base_string_tok[0][new_intervention.len_few_shots - 3]

                new_intervention.base_string_tok[0][new_intervention.len_few_shots - 3] = new_result_enc
                new_few_shot_string = tokenizer.decode(new_intervention.base_string_tok[0][:intervention.len_few_shots-1])
                print(new_few_shot_string)
                new_intervention.few_shots = new_few_shot_string
                new_intervention.base_string_tok_list = new_intervention.base_string_tok[0].numpy().tolist()
                counterfactual_list.append(new_intervention)

            elif args.counterfactual and args.representation == "words":
                new_intervention = copy.deepcopy(intervention)
                few_shot_result = tokenizer.decode(new_intervention.base_string_tok[0][new_intervention.len_few_shots - 3])[1:]
                if not args.counterfactual_symbol_result and not args.counterfactual_symbol_operands:
                    few_shot_result_int = int(words_to_n[few_shot_result])
                    new_result = random.randint(1, args.max_n)
                    while new_result == few_shot_result_int:
                        new_result = random.randint(1, args.max_n)
                    new_result_str = str(new_result)
                    if args.consistent_counterfactual:
                        new_result_str = convert_to_words(new_result_str)
                    new_result_str = ' ' + new_result_str
                    new_result_enc = tokenizer.encode(new_result_str)[0]
                elif args.counterfactual_symbol_result and not args.counterfactual_symbol_operands:
                    symbols = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "number", "result", "x", "y", "z", "a", "b", "c"]
                    # draw a random element from symbols
                    few_shot_result_str = random.choice(symbols)
                    few_shot_result_str = ' ' + few_shot_result_str
                    new_result_enc = tokenizer.encode(few_shot_result_str)[0]
                elif args.counterfactual_symbol_operands:
                    symbols = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", 
                               "eta", "theta", "iota", "number", "result", 
                               "x", "y", "z", "a", "b", "c"]
                    position_operands = equation_position_operands[new_intervention.equation]
                    for pos in position_operands:
                        new_operand_str = random.choice(symbols)
                        new_operand_str = ' ' + new_operand_str
                        new_operand_enc = tokenizer.encode(new_operand_str)[0]
                        new_intervention.base_string_tok[0][pos] = new_operand_enc
                    new_few_shot_string = tokenizer.decode(new_intervention.base_string_tok[0][:intervention.len_few_shots-1])
                    new_intervention.few_shots = new_few_shot_string
                    new_intervention.base_string_tok_list = new_intervention.base_string_tok[0].numpy().tolist()
                    new_result_enc = new_intervention.base_string_tok[0][new_intervention.len_few_shots - 3]
                
                new_intervention.base_string_tok[0][new_intervention.len_few_shots - 3] = new_result_enc
                new_few_shot_string = tokenizer.decode(new_intervention.base_string_tok[0][:intervention.len_few_shots-1])
                print(new_few_shot_string)                
                new_intervention.few_shots = new_few_shot_string
                new_intervention.base_string_tok_list = new_intervention.base_string_tok[0].numpy().tolist()
                counterfactual_list.append(new_intervention) 

            elif args.last_counterfactual:
                if args.representation == "words":
                    new_intervention = copy.deepcopy(intervention)
                    few_shot_result = tokenizer.decode(new_intervention.base_string_tok[0][new_intervention.len_few_shots - 3])[1:]
                    few_shot_result_int = int(words_to_n[few_shot_result])
                    new_result_str = str(few_shot_result_int)
                    new_result_str = ' ' + new_result_str
                    new_result_enc = tokenizer.encode(new_result_str)[0]
                else:
                    new_intervention = copy.deepcopy(intervention)
                    few_shot_result = tokenizer.decode(new_intervention.base_string_tok[0][new_intervention.len_few_shots - 3])[1:]
                    few_shot_result_word = convert_to_words(few_shot_result)
                    new_result_str = str(few_shot_result_word)
                    new_result_str = ' ' + new_result_str
                    new_result_enc = tokenizer.encode(new_result_str)[0]
                new_intervention.base_string_tok[0][new_intervention.len_few_shots - 3] = new_result_enc
                new_few_shot_string = tokenizer.decode(new_intervention.base_string_tok[0][:intervention.len_few_shots-1])
                print(new_few_shot_string)                
                new_intervention.few_shots = new_few_shot_string
                new_intervention.base_string_tok_list = new_intervention.base_string_tok[0].numpy().tolist()
                counterfactual_list.append(new_intervention)

            else:
                base_input_tok = base_input_tok.to(model.device)
                alt_input_tok = alt_input_tok.to(model.device) 
                base_output_generate = model.model.generate(base_input_tok, do_sample=False, max_new_tokens=1)
                base_output_generate = base_output_generate[0, -1].cpu().numpy()
                base_output_generate_decoded = tokenizer.decode(base_output_generate)
                # base_output_forward = model.model(base_input_tok)
                # base_logits = base_output_forward.logits
                # predicted_token_id = torch.argmax(base_logits[:, -1, :], dim=-1)
                # predicted_token = tokenizer.decode(predicted_token_id)
                #print(f"Base output generate: {base_output_generate_decoded}, predicted token: {predicted_token}")
                alt_output = model.model.generate(alt_input_tok, do_sample=False, max_new_tokens=1)
                alt_output = alt_output[0, -1].cpu().numpy()
                alt_output_decoded = tokenizer.decode(alt_output)
                print(f"Base output: {base_output_generate_decoded}, alt output: {alt_output_decoded}")
                print(f"few_shot: {intervention.few_shots}, base: {intervention.base_string}")
                if base_output_generate == intervention.res_base_tok[0]:
                    counter += 1
                    print("++++++++++++++++++++++++++++++++++++")
    print(f"Out of {len(data)} examples {counter/len(data[:])} were right.")

    
    if args.counterfactual:
        # save counterfactual data to pickle file
        file_name = args.data_dir
        file_name += '/' + str(args.model)
        file_name += '/intervention_' + n_shots + '_shots_' + 'max_' + max_n + '_' + representation
        file_name += '_counterfactual' if args.counterfactual else ''
        file_name += '_consistent' if args.consistent_counterfactual else '_inconsistent'
        file_name += '_symbol_result' if args.counterfactual_symbol_result else ''
        file_name += '_symbol_operands' if args.counterfactual_symbol_operands else ''
        file_name += '_further_templates' if args.extended_templates else ''
        with open(file_name + '.pkl', 'wb') as f:
            pickle.dump(counterfactual_list, f)
        print(f"Saved counterfactual data to {file_name}.pkl")
    elif args.last_counterfactual:
        file_name = args.data_dir
        file_name += '/' + str(args.model)
        file_name += '/intervention_' + n_shots + '_shots_' + 'max_' + max_n + '_' + representation
        file_name += '_last_counterfactual'
        with open(file_name + '.pkl', 'wb') as f:
            pickle.dump(counterfactual_list, f)
    
    return counter/len(data)

if __name__ == '__main__':
    test()

