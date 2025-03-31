import os
import hydra
import random
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import GPT2Tokenizer, BertTokenizer, AutoTokenizer, BloomTokenizerFast, GPTNeoXTokenizerFast, LlamaTokenizer
from intervention_models.intervention_model import load_model
from tqdm import tqdm
import pickle
import sys
import copy
from utils.number_utils import convert_to_words


@hydra.main(config_path='conf', config_name='config')
def evaluate_performance(args: DictConfig):
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
    representation = ["arabic", "words"]

    words_to_n = {convert_to_words(str(i)): i for i in range(args.max_n + 1)}

    model = load_model(args)
    tokenizer_class = (GPT2Tokenizer if model.is_gpt2 or model.is_gptneo or model.is_opt else
                           BertTokenizer if model.is_bert else
                           AutoTokenizer if model.is_gptj or model.is_flan or model.is_pythia or model.is_mpt else
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
    tokenizer = tokenizer_class.from_pretrained(tokenizer_id)
    model.create_vocab_subset(tokenizer, args)

    for rep in representation:

        f_name = args.data_dir + '/' + str(args.model) + '/intervention_' + n_shots + '_shots_' + 'max_' + max_n + '_' + rep
        f_name += '_counterfactual' if args.counterfactual else ''
        f_name += '_consistent' if args.consistent_counterfactual else '_inconsistent'
        f_name += '_symbol_result' if args.counterfactual_symbol_result else ''
        f_name += '_symbol_operands' if args.counterfactual_symbol_operands else ''
        f_name += '_further_templates' if args.extended_templates else ''

        with open(f_name + '.pkl', 'rb') as f:
            data = pickle.load(f)
        print("read data from", f_name + '.pkl')

        
        right_counter = 0
        wrong_counter = 0
        wrong_representation_right_counter = 0

        for intervention in data:
            base_input_tok = intervention.base_string_tok
            alt_input_tok = intervention.alt_string_tok
            with torch.no_grad():
                base_input_tok = base_input_tok.to(model.device)
                alt_input_tok = alt_input_tok.to(model.device) 
                base_output_generate = model.model.generate(base_input_tok, do_sample=False, max_length=1)
                base_output_generate = base_output_generate[0, -1].cpu().numpy()
                base_output_generate_decoded = tokenizer.decode(base_output_generate)
                alt_output = model.model.generate(alt_input_tok, do_sample=False, max_length=1)
                alt_output = alt_output[0, -1].cpu().numpy()
                alt_output_decoded = tokenizer.decode(alt_output)
                few_shot_result = intervention.base_string_tok[0, intervention.len_few_shots-3]
                few_shot_decoded = tokenizer.decode(few_shot_result)

                if rep == "words":
                    if base_output_generate_decoded != intervention.res_base_string and base_output_generate_decoded != ' ' + intervention.res_base_string:
                        wrong_rep_result = [key for key, value in words_to_n.items() if str(value) == base_output_generate_decoded[1:]]
                        if len(wrong_rep_result) > 0:
                            if wrong_rep_result[0] == intervention.res_base_string:
                                wrong_representation_right_counter += 1
                                #print("WRONG REPRESENTATION:", wrong_rep_result[0], "ANSWER:", intervention.res_base_string)

                if rep == "arabic":
                    if base_output_generate_decoded != intervention.res_base_string and base_output_generate_decoded != ' ' + intervention.res_base_string:
                        wrong_rep_result = words_to_n.get(base_output_generate_decoded[1:], None)
                        if wrong_rep_result != None:
                            if wrong_rep_result == int(intervention.res_base_string):
                                wrong_representation_right_counter += 1
                                print("WRONG REPRESENTATION:", wrong_rep_result, "ANSWER:", intervention.res_base_string)

                print(f"Base output: {base_output_generate_decoded} vs. base result {intervention.res_base_string}")
            
            
                
                if base_output_generate_decoded == intervention.res_base_string or base_output_generate_decoded == ' ' + intervention.res_base_string:
                    right_counter += 1
                else:
                    wrong_counter += 1
        print("Representation:", rep)

        print(f"Right: {right_counter} out of {len(data)}")
        print(f"Wrong representation right: {wrong_representation_right_counter} out of {len(data)}")
        print(f"Wrong: {len(data)-right_counter-wrong_representation_right_counter} out of {len(data)}")

if __name__ == '__main__':
    evaluate_performance()
