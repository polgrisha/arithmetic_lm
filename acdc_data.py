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
def create_acdc_data(args: DictConfig):
    print(OmegaConf.to_yaml(args))
    print("Model:", args.model)

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

    with open(f_name + '.pkl', 'rb') as f:
        data = pickle.load(f)
    
    tokenizer_class = AutoTokenizer
    tokenizer_id = args.model
    tokenizer = tokenizer_class.from_pretrained(tokenizer_id, cache_dir=args.transformers_cache_dir,
                                                padding_side="left")
    model = load_model(args)
    model.create_vocab_subset(tokenizer, args)

    if args.acdc_data:
        counterfactual_list = []
        words_to_n = {convert_to_words(str(i)): i for i in range(args.max_n + 1)}

        for intervention in data:
            base_input_tok = intervention.base_string_tok
            alt_input_tok = intervention.alt_string_tok
            padding_symbol = tokenizer.bos_token
            #padding_token = tokenizer.encode(padding_symbol)[0]
            padding_token = tokenizer.bos_token_id

            with torch.no_grad():
                new_intervention = copy.deepcopy(intervention)
                #print(f"initial base_string {tokenizer.decode(new_intervention.base_string_tok[0])}")
                #print(f"initial intervention alt_string: {tokenizer.decode(new_intervention.alt_string_tok[0])}")
                new_intervention.alt_string_tok = base_input_tok
                new_intervention.alt_string_tok_list = intervention.base_string_tok_list
                
                few_shot_result_pos = intervention.len_few_shots - 3

                new_intervention.alt_string_tok[0][few_shot_result_pos] = padding_token
                new_intervention.alt_string_tok_list[few_shot_result_pos] = padding_symbol
                #print(f"replaced result: {tokenizer.decode(new_intervention.alt_string_tok[0])}")
                
                old_prediction = tokenizer.decode(intervention.pred_res_alt_tok)
                new_prediction = model.model.generate(new_intervention.alt_string_tok.to("cuda"), max_new_tokens=1, do_sample=False)
                new_prediction = tokenizer.decode(new_prediction[0, -1])
                new_intervention.new_pred_res_alt_tok = tokenizer.encode(new_prediction)

                counterfactual_list.append(new_intervention)

        save_name = f_name + '_acdc.pkl'
        with open(save_name, 'wb') as f:
            pickle.dump(counterfactual_list, f)
        print("Saved ACDC data to", save_name)

        return 
    
    else:
        raise NotImplementedError





if __name__ == '__main__':
    create_acdc_data()