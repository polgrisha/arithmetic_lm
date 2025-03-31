from copy import deepcopy
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel, AutoModelForCausalLM, BertForMaskedLM, GPTNeoForCausalLM, BloomForCausalLM, \
    OPTForCausalLM, GPTNeoXForCausalLM, AutoModelForSeq2SeqLM, LlamaForCausalLM, PersimmonForCausalLM
from functools import partial
from interventions.lama_intervention import get_lama_vocab_subset
from utils.number_utils import batch, convert_to_words
from intervention_models.attention_modules import OverrideGPTJAttention, OverrideGPTNeoXAttention
from intervention_models import BaseModel
from peft import PeftModel
import sys
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, GPTNeoXTokenizerFast

def get_tokenizer(Model):
    if Model.is_neox:
        return GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    elif Model.is_pythia:
        return AutoTokenizer.from_pretrained("EleutherAI/pythia-12b-deduped-v0")
    elif Model.is_mistral: 
        return AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    elif Model.is_qwen:
        return AutoTokenizer.from_pretrained("Qwen/Qwen-14B", trust_remote_code=True)
    elif Model.is_bloom:
        return AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
    elif Model.is_persimmon:
        return AutoTokenizer.from_pretrained("/home/aoq559/.cache/huggingface/hub/persimmon/")
    elif Model.is_mpt:
        return AutoTokenizer.from_pretrained("mosaicml/mpt-7b")
    else:
        raise Exception(f'Tokenizer version not supported: {Model.model}')


def load_model(args):
    return Model(device=args.device, random_weights=args.random_weights, model_version=args.model,
                 model_ckpt=args.model_ckpt, transformers_cache_dir=args.transformers_cache_dir, int8=args.int8)


class Model(BaseModel):

    def __init__(self, device='cpu', random_weights=False, model_version='gpt2', model_ckpt=None,
                 transformers_cache_dir=None, int8=False):
        # handle detection of model
        super().__init__(model_version)
        assert (self.is_gpt2 or self.is_bert or self.is_gptj or self.is_gptneo or self.is_bloom or self.is_qwen
                or self.is_opt or self.is_neox or self.is_flan or self.is_pythia or self.is_llama 
                or self.is_mistral or self.is_persimmon or self.is_mpt)

        self.device = device

        model_class = (GPT2LMHeadModel if self.is_gpt2 else
                       AutoModelForCausalLM if self.is_gptj else
                       BertForMaskedLM if self.is_bert else
                       GPTNeoForCausalLM if self.is_gptneo else
                       BloomForCausalLM if self.is_bloom else
                       OPTForCausalLM if self.is_opt else
                       GPTNeoXForCausalLM if self.is_neox else
                       AutoModelForSeq2SeqLM if self.is_flan else
                       GPTNeoXForCausalLM if self.is_pythia else
                       LlamaForCausalLM if self.is_llama else
                       AutoModelForCausalLM if self.is_mistral or self.is_qwen or self.is_mpt else
                       PersimmonForCausalLM if self.is_persimmon else
                       None)
        if self.is_llama and 'goat' in model_version:
            # model = LlamaForCausalLM.from_pretrained(
            #     'decapoda-research/llama-7b-hf',
            #     load_in_8bit=int8,
            #     torch_dtype=torch.float16,
            #     cache_dir=transformers_cache_dir,
            #     device_map="auto",
            # )
            # model = LlamaForCausalLM.from_pretrained(
            #     'baffo32/decapoda-research-llama-7B-hf',
            #     load_in_8bit=int8,
            #     torch_dtype=torch.float16,
            #     cache_dir=transformers_cache_dir,
            #     device_map="auto",
            # )
            model = AutoModelForCausalLM.from_pretrained(
                "baffo32/decapoda-research-llama-7B-hf",
                load_in_8bit=int8,
                torch_dtype=torch.float16,
                cache_dir=transformers_cache_dir,
                device_map="auto")
            self.model = PeftModel.from_pretrained(
                model,
                'tiedong/goat-lora-7b',
                torch_dtype=torch.float16,
                cache_dir=transformers_cache_dir,
                device_map={'': 0},
            )
        elif model_ckpt:
            self.model = model_class.from_pretrained(model_ckpt)
            self.model.to(device)
        elif self.is_mistral:
            self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",
                load_in_8bit=int8,
                torch_dtype=torch.float16,
                cache_dir=transformers_cache_dir,
                device_map="auto"
            )
            if hasattr(self.model, 'hf_device_map'):
                print('hf_device_map:', self.model.hf_device_map)
            else:
                self.model.to(device)
        elif self.is_qwen:
            self.model = model_class.from_pretrained(
                model_version,
                device_map="auto",
                torch_dtype=torch.float16,
                fp16=True,
                trust_remote_code=True
            )
            if hasattr(self.model, 'hf_device_map'):
                print('hf_device_map:', self.model.hf_device_map)
            else:
                self.model.to(device)
        elif self.is_bloom:
            self.model = model_class.from_pretrained(
                model_version,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                cache_dir=transformers_cache_dir,
                device_map="auto"
            )
            if hasattr(self.model, 'hf_device_map'):
                print('hf_device_map:', self.model.hf_device_map)
            else:
                self.model.to(device)
        elif self.is_persimmon:
            self.model = model_class.from_pretrained(
                "/home/aoq559/.cache/huggingface/hub/persimmon/",
                load_in_8bit=False,
                torch_dtype=torch.float32,
                device_map="auto",
            )
            if hasattr(self.model, 'hf_device_map'):
                print('hf_device_map:', self.model.hf_device_map)
            else:
                self.model.to(device)
        elif self.is_mpt:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_version, trust_remote_code=True)
            # config.attn_config["attn_impl"] = "triton"
            #config.torch_dtype = torch.bfloat16

            self.model = model_class.from_pretrained(
                model_version,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                cache_dir=transformers_cache_dir,
                device_map="auto",
                trust_remote_code=True,
                config=config,
            )
            if hasattr(self.model, 'hf_device_map'):
                print('hf_device_map:', self.model.hf_device_map)
            else:
                self.model.to(device)
            print(self.model.config)

        else:
            self.model = model_class.from_pretrained(
                model_version,
                load_in_8bit=int8,
                torch_dtype=torch.float16,
                device_map = 'auto',
                cache_dir=transformers_cache_dir,
                output_attentions=False
            )
            
            if hasattr(self.model, 'hf_device_map'):
                print('hf_device_map:', self.model.hf_device_map)
            else:
                self.model.to(device)

        if random_weights:
            print('Randomizing weights')
            self.model.init_weights()

        # store model details
        if not self.is_mpt:
            self.num_layers = self.model.config.num_hidden_layers # pythia 36
            self.num_neurons = self.model.config.hidden_size # pythia 5120
            self.num_heads = self.model.config.num_attention_heads # pythia 40
        else:
            self.num_layers = self.model.config.n_layers# mpt 32
            self.num_neurons = self.model.config.d_model # mpt 4096
            self.num_heads = self.model.config.n_heads # mpt 32

        if self.is_gpt2 or self.is_gptj or self.is_gptneo:
            # retrieve specific layers from the model
            self.get_attention_layer = lambda layer: self.model.transformer.h[layer].attn
            self.word_emb_layer = self.model.transformer.wte
            self.get_neuron_layer = lambda layer: self.model.transformer.h[layer].mlp
        elif self.is_neox:
            self.get_attention_layer = lambda layer: self.model.gpt_neox.layers[layer].attention
            self.word_emb_layer = self.model.gpt_neox.embed_in
            self.get_neuron_layer = lambda layer: self.model.gpt_neox.layers[layer].mlp
        elif self.is_flan:
            self.get_attention_layer = lambda layer: (self.model.encoder.block + self.model.decoder.block)[layer].layer[
                0]
            self.word_emb_layer = self.model.encoder.embed_tokens
            self.get_neuron_layer = lambda layer: (self.model.encoder.block + self.model.decoder.block)[layer].layer[
                1 if layer < len(self.model.encoder.block) else 2]
        elif self.is_pythia:
            self.get_attention_layer = lambda layer: self.model.gpt_neox.layers[layer].attention
            self.word_emb_layer = self.model.gpt_neox.embed_in
            self.get_neuron_layer = lambda layer: self.model.gpt_neox.layers[layer].mlp
            self.get_hidden_states = lambda layer: self.model.gpt_neox.layers[layer]
        elif self.is_llama:
            if 'goat' in model_version:
                self.get_attention_layer = lambda layer: self.model.model.model.layers[layer].self_attn
                self.word_emb_layer = self.model.model.model.embed_tokens
                self.get_neuron_layer = lambda layer: self.model.model.model.layers[layer].mlp
            else:
                self.get_attention_layer = lambda layer: self.model.model.layers[layer].self_attn
                self.word_emb_layer = self.model.model.embed_tokens
                self.get_neuron_layer = lambda layer: self.model.model.layers[layer].mlp
        elif self.is_opt:
            self.get_attention_layer = lambda layer: self.model.model.decoder.layers[layer].self_attn
            self.word_emb_layer = self.model.model.decoder.embed_tokens
            self.get_neuron_layer = lambda layer: self.model.model.decoder.layers[layer].fc
        elif self.is_mistral:
            self.get_attention_layer = lambda layer: self.model.model.layers[layer].self_attn
            self.word_emb_layer = self.model.model.embed_tokens
            self.get_neuron_layer = lambda layer: self.model.model.layers[layer].mlp
        elif self.is_qwen:
            self.get_attention_layer = lambda layer: self.model.transformer.h[layer].attn
            self.word_emb_layer = self.model.transformer.wte
            self.get_neuron_layer = lambda layer: self.model.transformer.h[layer].mlp
        elif self.is_bloom:
            self.get_attention_layer = lambda layer: self.model.transformer.h[layer].self_attention
            self.word_emb_layer = self.model.transformer.word_embeddings
            self.get_neuron_layer = lambda layer: self.model.transformer.h[layer].mlp
        elif self.is_persimmon:
            self.get_attention_layer = lambda layer: self.model.model.layers[layer].self_attn
            self.word_emb_layer = self.model.model.embed_tokens
            self.get_neuron_layer = lambda layer: self.model.model.layers[layer].mlp
        elif self.is_mpt:
            self.get_attention_layer = lambda layer: self.model.transformer.blocks[layer].attn
            self.word_emb_layer = self.model.transformer.wte
            self.get_neuron_layer = lambda layer: self.model.transformer.blocks[layer].ffn
        else:
            raise Exception(f'Model version not supported: {model_version}')

    def create_vocab_subset(self, tokenizer, args):
        if args.intervention_type == 20:
            vocab_subset = get_lama_vocab_subset(tokenizer, args)
            ids, tokens = zip(*vocab_subset.items())
            self.vocab_subset = ids
            self.word_subset = tokens
        elif self.is_flan:
            if args.max_n > 55:
                raise Exception('flan-t5\'s tokenizer splits some numbers over 55')
            if args.representation == 'arabic':
                self.vocab_subset = tokenizer.convert_tokens_to_ids(['▁' + str(i) for i in range(args.max_n + 1)])
            else:
                raise Exception('Representation unknown: {}'.format(args.representation))
        else:
            if self.is_opt:
                prefix = 'a '
                start_idx = 2
            elif self.is_llama:
                start_idx = 2
                prefix = ''
            elif self.is_mistral:
                start_idx = 2
                prefix = ''
            elif self.is_qwen:
                raise NotImplementedError
            elif self.is_bloom:
                prefix = 'a '
                start_idx = 1
            else:
                prefix = ' '
                start_idx = 0

            if args.representation == 'arabic':
                self.vocab_subset = [tokenizer.encode(prefix + str(i))[start_idx:] for i in range(args.max_n + 1)]
            elif args.representation == 'words':
                self.vocab_subset = [tokenizer.encode(prefix + convert_to_words(str(i)))[start_idx:] for i in
                                     range(args.max_n + 1)]
            else:
                raise Exception('Representation unknown: {}'.format(args.representation))
            
    def double_mixed_intervention_experiment(self, interventions, effect_type, mlp_position_dictionary, intervention_loc, position_fewshot, multitoken=False,
                                get_full_distribution=False, all_tokens=False):
        
        intervention_results = {}
        progress = tqdm(total=len(interventions), desc='performing interventions')
        for idx, intervention in enumerate(interventions):
            mlp_position = mlp_position_dictionary.get(intervention.equation)
            intervention_results[idx] = self.double_mixed_neuron_intervention_single_experiment(intervention=intervention,
                                                                                   effect_type=effect_type,
                                                                                   mlp_position=mlp_position,
                                                                                   intervention_loc='attention_layer_output',
                                                                                   position_fewshot=position_fewshot,
                                                                                   get_full_distribution=get_full_distribution,
                                                                                   all_tokens=all_tokens)
            progress.update()

        return intervention_results
            
    def mixed_intervention_experiment(self, interventions, effect_type, mlp_position_dictionary, intervention_loc, position_fewshot, multitoken=False,
                                get_full_distribution=False, all_tokens=False):
        intervention_results = {}
        progress = tqdm(total=len(interventions), desc='performing interventions')
        for idx, intervention in enumerate(interventions):
            mlp_position = mlp_position_dictionary.get(intervention.equation)
            intervention_results[idx] = self.mixed_neuron_intervention_single_experiment(intervention=intervention,
                                                                                   effect_type=effect_type,
                                                                                   mlp_position=mlp_position,
                                                                                   intervention_loc='attention_layer_output',
                                                                                   position_fewshot=position_fewshot,
                                                                                   get_full_distribution=get_full_distribution,
                                                                                   all_tokens=all_tokens)
            progress.update()

        return intervention_results

    def intervention_experiment(self, interventions, effect_type, intervention_loc, position_fewshot, multitoken=False,
                                get_full_distribution=False, all_tokens=False):
        intervention_results = {}
        progress = tqdm(total=len(interventions), desc='performing interventions')
        for idx, intervention in enumerate(interventions):
            intervention_results[idx] = self.neuron_intervention_single_experiment(intervention=intervention,
                                                                                   effect_type=effect_type,
                                                                                   intervention_loc=intervention_loc,
                                                                                   position_fewshot=position_fewshot,
                                                                                   get_full_distribution=get_full_distribution,
                                                                                   all_tokens=all_tokens)
            progress.update()

        return intervention_results

    def get_representations(self, context, position=-1, is_attention=False):
        # Hook for saving the representation
        def extract_representation_hook(module,
                                        input,
                                        output,
                                        position,
                                        representations,
                                        layer):
            representations[layer] = output[(0, position)]

        def extract_representation_hook_attn(module,
                                             input,
                                             output,
                                             position,
                                             representations,
                                             layer):
            representations[layer] = output[0][0, position]

        handles = []
        representation = {}
        with torch.no_grad():
            # construct all the hooks
            # word embeddings will be layer -1
            if not is_attention:
                handles.append(self.word_emb_layer.register_forward_hook(
                    partial(extract_representation_hook,
                            position=position,
                            representations=representation,
                            layer=-1)))
            # hidden layers
            for layer_n in range(self.num_layers):
                if is_attention:
                    handles.append(self.get_attention_layer(layer_n).register_forward_hook(
                        partial(extract_representation_hook_attn,
                                position=position,
                                representations=representation,
                                layer=layer_n)))
                else:
                    handles.append(self.get_neuron_layer(layer_n).register_forward_hook(
                        partial(extract_representation_hook,
                                position=position,
                                representations=representation,
                                layer=layer_n)))
            if self.is_flan:
                self.model(context.to(self.device), decoder_input_ids=torch.tensor([[0]]).to(self.device))
            else:
                self.model(context.to(self.device))
            for h in handles:
                h.remove()
        return representation

    def get_probability_for_example(self, context, res_tok):
        """Return probabilities of single-token candidates given context"""
        if len(res_tok) > 1:
            raise ValueError(f"Multiple tokens not allowed: {res_tok}")

        if self.is_flan:
            decoder_input_ids = torch.tensor([[0]] * context.shape[0]).to(self.device)
            logits = self.model(context.to(self.device), decoder_input_ids=decoder_input_ids)[0].to(
                'cpu')
        else:
            with autocast():
                logits = self.model(context.to(self.device))[0].to('cpu') # extract logits from model output
        logits = logits[:, -1, :].float()
        probs = F.softmax(logits, dim=-1)
        probs_res = probs[:, res_tok[0]].squeeze().tolist()
        logits_res = logits[:, res_tok[0]].squeeze().tolist()

        # decode the output
        tokenizer = get_tokenizer(self)
        logits_argmax = logits.argmax(dim=-1)
        predicted_token = tokenizer.decode(logits_argmax)

        return probs_res, logits_res, predicted_token

    def get_distribution_for_example(self, context):
        if self.is_flan:
            logits = self.model(context.to(self.device), decoder_input_ids=torch.tensor([[0]]).to(self.device))[0].to(
                'cpu')
        else:
            logits = self.model(context.to(self.device))[0].to('cpu')
        logits = logits[:, -1, :].float()
        probs = F.softmax(logits, dim=-1)
        probs_subset = probs[:, self.vocab_subset].squeeze()
        logits_subset = logits[:, self.vocab_subset].squeeze()
        return probs_subset, logits_subset
    
    def double_mixed_neuron_intervention_single_experiment(self,
                                                intervention, 
                                                effect_type, 
                                                mlp_position,
                                                bsize=200,
                                                intervention_loc='attention_layer_output', # MLP or attention
                                                position_fewshot="all",
                                                get_full_distribution=False,
                                                all_tokens=False): # only last token
        """
        run one full neuron intervention experiment
        - first intervene on the activations of the layer with index -1 at mlp_position
        - second intervene on the attention output of layer with index 5 at mlp_position
        - third intervente on the attention output of every subsequent layer in the actual task"""

        with torch.no_grad():
                
                # Probabilities without intervention (Base case)
                # base: prompt 1
                # alt: prompt 2
                distrib_base, logits_base = self.get_distribution_for_example(intervention.base_string_tok)
                distrib_alt, logits_alt = self.get_distribution_for_example(intervention.alt_string_tok)
                distrib_base, logits_base = distrib_base.numpy(), logits_base.numpy()
                distrib_alt, logits_alt = distrib_alt.numpy(), logits_alt.numpy()
    
                # return difference in distribution without intervention
                if effect_type == 'total':
                    return distrib_base, distrib_alt, None, None
    
                #context = intervention.base_string_tok # base_string_tok includes the few_shot_example
                context = intervention.alt_string_tok

                intervention_on_output = len(intervention_loc.split('_')) > 1 and intervention_loc.split('_')[2] == 'output'
                if intervention_on_output:
                    intervention_loc = 'layer_output'

                # get position of the actual prompt after the few-shot examples
                if position_fewshot:
                    positions = list(range(intervention.len_few_shots)) if all_tokens else [-1]
                else:
                    positions = list(range(intervention.len_few_shots, len(context.squeeze()))) if all_tokens else [-1]
                
                res_base_probs = {}
                res_alt_probs = {}
                res_base_logits = {}
                res_alt_logits = {}

                mlp_position = mlp_position
                mlp_rep = self.get_representations(intervention.base_string_tok, position=mlp_position)

                layers_to_search = [-1]
                neurons_mlp = list(range(self.num_neurons))
                neurons_to_search = [neurons_mlp]

                handle_list_mlp = self.neuron_intervention2(
                    context=context,
                    res_base_tok=intervention.res_base_tok,
                    res_alt_tok=intervention.pred_res_alt_tok,
                    rep=mlp_rep,
                    layers=layers_to_search,
                    neurons=neurons_to_search,
                    position=mlp_position,
                    intervention_loc='layer',
                    get_full_distribution=get_full_distribution,
                    is_attention=False)
 
                x = intervention.base_string_tok[0]
                x_alt = intervention.alt_string_tok[0]
                input = x
                context = x_alt
                batch_size = 1
                seq_len = len(x)
                seq_len_alt = len(x_alt)

                if intervention_on_output:
                    attention_override_layer = self.get_representations(input.unsqueeze(0), position=mlp_position,
                                                                        is_attention=True)
                    assert attention_override_layer[0].shape[0] == self.num_neurons, \
                        f'attention_override[0].shape: {attention_override_layer[0].shape} vs {self.num_neurons}'
                    layers_to_search = [5]
                    handle_list_attn = self.neuron_intervention2(
                        context=context.unsqueeze(0),
                        res_base_tok=intervention.res_base_tok,
                        res_alt_tok=intervention.pred_res_alt_tok,
                        rep=attention_override_layer,
                        layers=layers_to_search,
                        neurons=[list(range(self.num_neurons))],
                        position=mlp_position,
                        intervention_loc='layer',
                        get_full_distribution=get_full_distribution,
                        is_attention=True)
                
                for position in positions:
                    if intervention_on_output:
                        attention_override = self.get_representations(input.unsqueeze(0), position=position,
                                                                  is_attention=True)
                        assert attention_override[0].shape[0] == self.num_neurons, \
                            f'attention_override[0].shape: {attention_override[0].shape} vs {self.num_neurons}'
                        
                    if intervention_loc.startswith('layer'):
                        dimensions = (self.num_layers, 1)
                        res_base_probs[position] = torch.zeros(dimensions)
                        res_alt_probs[position] = torch.zeros(dimensions)
                        res_base_logits[position] = torch.zeros(dimensions)
                        res_alt_logits[position] = torch.zeros(dimensions)

                        for layer in range(self.num_layers):
                            if intervention_on_output:
                                probs, logits = self.neuron_intervention(
                                    context=context.unsqueeze(0),
                                    res_base_tok=intervention.res_base_tok,
                                    res_alt_tok=intervention.pred_res_alt_tok,
                                    rep=attention_override,
                                    layers=[layer],
                                    neurons=[list(range(self.num_neurons))],
                                    position=position,
                                    intervention_loc='layer',
                                    get_full_distribution=get_full_distribution,
                                    is_attention=True)
                                p1, p2 = probs
                                l1, l2 = logits
                                res_base_probs[position][layer][0] = torch.tensor(p1)
                                res_alt_probs[position][layer][0] = torch.tensor(p2)
                                res_base_logits[position][layer][0] = torch.tensor(l1)
                                res_alt_logits[position][layer][0] = torch.tensor(l2)

                    else:
                        raise ValueError(f'Intervention location not supported: {intervention_loc}')
        for handle in handle_list_mlp:
            handle.remove()

        for h in handle_list_attn:
            h.remove()   

        return distrib_base, distrib_alt, logits_base, logits_alt, res_base_probs, res_alt_probs, res_base_logits, res_alt_logits
        

    def mixed_neuron_intervention_single_experiment(self,
                                              intervention, # 1, 2, 3, 11, 20
                                              effect_type, # indirect, total
                                              mlp_position,
                                              bsize=200,
                                              intervention_loc='attention_layer_output', # MLP or attention
                                              position_fewshot=True,
                                              get_full_distribution=False,
                                              all_tokens=False): # only last token
        """
        run one full neuron intervention experiment:
         - first intervene on the activations of the layer with index -1
         - second intervene on the attention output of every subsequent layer
        """

        with torch.no_grad():
                
                # Probabilities without intervention (Base case)
                # base: prompt 1
                # alt: prompt 2
                distrib_base, logits_base = self.get_distribution_for_example(intervention.base_string_tok)
                distrib_alt, logits_alt = self.get_distribution_for_example(intervention.alt_string_tok)
                distrib_base, logits_base = distrib_base.numpy(), logits_base.numpy()
                distrib_alt, logits_alt = distrib_alt.numpy(), logits_alt.numpy()
    
                # return difference in distribution without intervention
                if effect_type == 'total':
                    return distrib_base, distrib_alt, None, None
    
                #context = intervention.base_string_tok # base_string_tok includes the few_shot_example
                context = intervention.alt_string_tok

                intervention_on_output = len(intervention_loc.split('_')) > 1 and intervention_loc.split('_')[2] == 'output'
                if intervention_on_output:
                    intervention_loc = 'layer_output'

                # get position of the actual prompt after the few-shot examples
                if position_fewshot:
                    positions = list(range(intervention.len_few_shots)) if all_tokens else [-1]
                else:
                    positions = list(range(intervention.len_few_shots, len(context.squeeze()))) if all_tokens else [-1]
                
                res_base_probs = {}
                res_alt_probs = {}
                res_base_logits = {}
                res_alt_logits = {}
                hooked_pred_base = {}
                hooked_pred_alt = {}

                mlp_position = mlp_position
                mlp_rep = self.get_representations(intervention.base_string_tok, position=mlp_position)

                layers_to_search = [-1]
                neurons_mlp = list(range(self.num_neurons))
                neurons_to_search = [neurons_mlp]

                handle_list = self.neuron_intervention2(
                    context=context,
                    res_base_tok=intervention.res_base_tok,
                    res_alt_tok=intervention.pred_res_alt_tok,
                    rep=mlp_rep,
                    layers=layers_to_search,
                    neurons=neurons_to_search,
                    position=mlp_position,
                    intervention_loc='layer',
                    get_full_distribution=get_full_distribution,
                    is_attention=False)
 
                x = intervention.base_string_tok[0]
                x_alt = intervention.alt_string_tok[0]
                input = x
                context = x_alt
                batch_size = 1
                seq_len = len(x)
                seq_len_alt = len(x_alt)
                
                for position in positions:
                    if intervention_on_output:
                        attention_override = self.get_representations(input.unsqueeze(0), position=position,
                                                                  is_attention=True)
                        assert attention_override[0].shape[0] == self.num_neurons, \
                            f'attention_override[0].shape: {attention_override[0].shape} vs {self.num_neurons}'
                        
                    if intervention_loc.startswith('layer'):
                        dimensions = (self.num_layers, 1)
                        res_base_probs[position] = torch.zeros(dimensions)
                        res_alt_probs[position] = torch.zeros(dimensions)
                        res_base_logits[position] = torch.zeros(dimensions)
                        res_alt_logits[position] = torch.zeros(dimensions)
                        hooked_pred_base[position] = np.full(dimensions, None)
                        hooked_pred_alt[position] = np.full(dimensions, None)

                        for layer in range(self.num_layers):
                            if intervention_on_output:
                                probs, logits, hooked_predictions = self.neuron_intervention(
                                    context=context.unsqueeze(0),
                                    res_base_tok=intervention.res_base_tok,
                                    res_alt_tok=intervention.pred_res_alt_tok,
                                    rep=attention_override,
                                    layers=[layer],
                                    neurons=[list(range(self.num_neurons))],
                                    position=position,
                                    intervention_loc='layer',
                                    get_full_distribution=get_full_distribution,
                                    is_attention=True)
                                p1, p2 = probs
                                l1, l2 = logits
                                pb, pa = hooked_predictions
                                res_base_probs[position][layer][0] = torch.tensor(p1)
                                res_alt_probs[position][layer][0] = torch.tensor(p2)
                                res_base_logits[position][layer][0] = torch.tensor(l1)
                                res_alt_logits[position][layer][0] = torch.tensor(l2)
                                hooked_pred_base[position][layer][0] = pb
                                hooked_pred_alt[position][layer][0] = pa

                    else:
                        raise ValueError(f'Intervention location not supported: {intervention_loc}')
        for handle in handle_list:
            handle.remove()   

        return distrib_base, distrib_alt, logits_base, logits_alt, res_base_probs, res_alt_probs, res_base_logits, res_alt_logits, hooked_pred_base, hooked_pred_alt
    

    def neuron_intervention_single_experiment(self,
                                              intervention, # 1, 2, 3, 11, 20
                                              effect_type, # indirect, total
                                              bsize=200,
                                              intervention_loc='all', # MLP or attention
                                              position_fewshot=True,
                                              get_full_distribution=False,
                                              all_tokens=False): # only last token
        """
        run one full neuron intervention experiment
        """

        with torch.no_grad():

            # Probabilities without intervention (Base case)
            # base: prompt 1
            # alt: prompt 2
            distrib_base, logits_base = self.get_distribution_for_example(intervention.base_string_tok)
            distrib_alt, logits_alt = self.get_distribution_for_example(intervention.alt_string_tok)
            distrib_base, logits_base = distrib_base.numpy(), logits_base.numpy()
            distrib_alt, logits_alt = distrib_alt.numpy(), logits_alt.numpy()

            # return difference in distribution without intervention
            if effect_type == 'total':
                return distrib_base, distrib_alt, None, None

            #context = intervention.base_string_tok # base_string_tok includes the few_shot_example
            context = intervention.alt_string_tok # edited to insert base intervention into alt string 

            # get position of the actual prompt after the few-shot examples
            # or get position of the last token
            # --> use positions to access the positions where activation is to be intervened
            
            if position_fewshot:
                positions = list(range(intervention.len_few_shots)) if all_tokens else [-1]
            else:
                positions = list(range(intervention.len_few_shots, len(context.squeeze()))) if all_tokens else [-1]


            res_base_probs = {}
            res_alt_probs = {}
            res_base_logits = {}
            res_alt_logits = {}
            hooked_pred_base = {}
            hooked_pred_alt = {}

            for position in positions:
                # assumes effect_type is indirect
                # get activation for the alternative string at the position
                #rep = self.get_representations(intervention.alt_string_tok, position=position) # original
                rep = self.get_representations(intervention.base_string_tok, position=position) # edited

                if intervention_loc == 'all':
                    # Now intervening on potentially biased example
                    # (all layers + embedding layer, number of neurons in each layer)
                    dimensions = (self.num_layers + 1, self.num_neurons)
                    res_base_probs[position] = torch.zeros(dimensions)
                    res_alt_probs[position] = torch.zeros(dimensions)
                    res_base_logits[position] = torch.zeros(dimensions)
                    res_alt_logits[position] = torch.zeros(dimensions)
                    first_layer = -1
                    for layer in tqdm(range(first_layer, self.num_layers)):
                        layers_to_search = [layer]
                        # batch of neurons for efficiency
                        for neurons in batch(range(self.num_neurons), bsize):
                            neurons_to_search = [[i] for i in neurons]
                            # return p1, p2 for base and alternative prompt
                            probs, logits, pred_base, pred_alt = self.neuron_intervention(
                                context=context,
                                res_base_tok=intervention.res_base_tok,
                                #res_alt_tok=intervention.res_alt_tok,
                                res_alt_tok=intervention.pred_res_alt_tok,
                                rep=rep,
                                layers=layers_to_search,
                                neurons=neurons_to_search,
                                position=-1,
                                intervention_loc=intervention_loc,
                                get_full_distribution=get_full_distribution)

                            for neuron, (p1, p2), (l1, l2), (pb, pa) in zip(neurons, probs, logits, pred_base, pred_alt):
                                res_base_probs[position][layer + 1][neuron] = p1
                                res_alt_probs[position][layer + 1][neuron] = p2
                                res_base_logits[position][layer + 1][neuron] = l1
                                res_alt_logits[position][layer + 1][neuron] = l2
                                hooked_pred_base[position][layer + 1][neuron] = pb
                                hooked_pred_alt[position][layer + 1][neuron] = pa

                elif intervention_loc == 'layer':
                    dimensions = (self.num_layers + 1, 1)
                    if get_full_distribution:
                        dimensions = (self.num_layers + 1, 1, len(self.vocab_subset))
                    res_base_probs[position] = torch.zeros(dimensions)
                    res_alt_probs[position] = torch.zeros(dimensions)
                    res_base_logits[position] = torch.zeros(dimensions)
                    res_alt_logits[position] = torch.zeros(dimensions)
                    hooked_pred_base[position] = np.full(dimensions, None)
                    hooked_pred_alt[position] = np.full(dimensions, None)

                    first_layer = -1
                    for layer in range(first_layer, self.num_layers):
                        neurons = list(range(self.num_neurons))
                        neurons_to_search = [neurons]
                        layers_to_search = [layer]

                        probs, logits, predictions = self.neuron_intervention(
                            context=context,
                            res_base_tok=intervention.res_base_tok,
                            #res_alt_tok=intervention.res_alt_tok,
                            res_alt_tok=intervention.pred_res_alt_tok,
                            rep=rep,
                            layers=layers_to_search,
                            neurons=neurons_to_search,
                            position=position,
                            intervention_loc=intervention_loc,
                            get_full_distribution=get_full_distribution)
                        p1, p2 = probs
                        l1, l2 = logits
                        pb, pa = predictions
                        res_base_probs[position][layer + 1][0] = torch.tensor(p1)
                        res_alt_probs[position][layer + 1][0] = torch.tensor(p2)
                        res_base_logits[position][layer + 1][0] = torch.tensor(l1)
                        res_alt_logits[position][layer + 1][0] = torch.tensor(l2)
                        hooked_pred_base[position][layer + 1][0] = pb
                        hooked_pred_alt[position][layer + 1][0] = pa

                elif intervention_loc.startswith('single_layer_'):
                    layer = int(intervention_loc.split('_')[-1])
                    dimensions = (1, self.num_neurons)
                    res_base_probs[position] = torch.zeros(dimensions)
                    res_alt_probs[position] = torch.zeros(dimensions)
                    res_base_logits[position] = torch.zeros(dimensions)
                    res_alt_logits[position] = torch.zeros(dimensions)
                    layers_to_search = [layer]
                    for neurons in batch(range(self.num_neurons), bsize):
                        neurons_to_search = [[i] for i in neurons]

                        probs, logits = self.neuron_intervention(
                            context=context,
                            res_base_tok=intervention.res_base_tok,
                            #res_alt_tok=intervention.res_alt_tok,
                            res_alt_tok=intervention.pred_res_alt_tok,
                            rep=rep,
                            layers=layers_to_search,
                            neurons=neurons_to_search,
                            position=position,
                            intervention_loc=intervention_loc,
                            get_full_distribution=get_full_distribution)

                        for neuron, (p1, p2), (l1, l2) in zip(neurons, probs, logits):
                            res_base_probs[position][0][neuron] = p1
                            res_alt_probs[position][0][neuron] = p2
                            res_base_logits[position][0][neuron] = l1
                            res_alt_logits[position][0][neuron] = l2

                else:
                    raise Exception(f'intervention_loc not defined: {intervention_loc}')

        return distrib_base, distrib_alt, logits_base, logits_alt, res_base_probs, res_alt_probs, res_base_logits, res_alt_logits, hooked_pred_base, hooked_pred_alt

    # targeted intervention on neurons or attention heads
    def neuron_intervention(self,
                            context,
                            res_base_tok,
                            res_alt_tok,
                            rep,
                            layers,
                            neurons,
                            position,
                            intervention_loc,
                            get_full_distribution=False,
                            is_attention=False):
        # Hook for changing representation during forward pass
        def intervention_hook(module, # model layer
                              input, # input to model layer
                              output, # output before intervention
                              position, # position of token to intervene on
                              neurons, # neurons to intervene on
                              intervention, # intervention values
                              is_attention=False):
            # Get the neurons to intervene on
            if is_attention:
                o = output[0]
            else:
                o = output
            out_device = o.device
            neurons = torch.LongTensor(neurons).to(out_device)
            # First grab the position across batch
            # Then, for each element, get correct index w/ gather
            # extract position in the output tensor where the intervention will be applied
            base_slice = (slice(None), position, slice(None))
            # extract the values from the output tensor at the specified position and neurons
            base = o[base_slice].gather(1, neurons)
            # reshape to match the shape of the base tensor
            intervention_view = intervention.view_as(base)
            # set base to intervention
            base = intervention_view

            # Overwrite values in the output
            # First define mask where to overwrite
            scatter_mask = torch.zeros_like(o, dtype=torch.bool)
            for i, v in enumerate(neurons):
                scatter_mask[(i, position, v)] = 1
            # Then take values from base and scatter and overwrite the output
            o.masked_scatter_(scatter_mask, base.flatten())

        # Set up the context as batch
        batch_size = len(neurons)
        # replicate context for each neuron in the batch
        context = context.repeat(batch_size, 1)
        handle_list = []
        for layer in set(layers):
            if intervention_loc == 'all' or intervention_loc.startswith('single_layer_'):
                neuron_loc = np.where(np.array(layers) == layer)[0] # get index of first layer in layers set
                n_list = []
                for n in neurons:
                    unsorted_n_list = [n[i] for i in neuron_loc]
                    n_list.append(list(np.sort(unsorted_n_list)))
            elif intervention_loc == 'layer':
                n_list = neurons
            else:
                raise Exception(f'intervention_loc not defined: {intervention_loc}')
            m_list = n_list
            intervention_rep = rep[layer][m_list]
            if layer == -1:
                if is_attention:
                    raise Exception('Attention not implemented for word embedding intervention')

                handle_list.append(self.word_emb_layer.register_forward_hook(
                    partial(intervention_hook,
                            position=position,
                            neurons=n_list,
                            intervention=intervention_rep)))
            else:
                if is_attention:
                    module = self.get_attention_layer(layer)
                else:
                    module = self.get_neuron_layer(layer)
                handle_list.append(module.register_forward_hook(
                    partial(intervention_hook,
                            position=position,
                            neurons=n_list,
                            intervention=intervention_rep,
                            is_attention=is_attention)))

        if get_full_distribution:
            # new distribution is a list of tensors
            new_distrib, new_logits = self.get_distribution_for_example(context)
            probs = new_distrib, [0. for _ in range(len(new_distrib))]
        else:
            new_base_probability, new_base_logits, hooked_base_pred = self.get_probability_for_example(context, res_base_tok) # get prob for alt prompt + base prediction
            new_alt_probability, new_alt_logits, hooked_alt_pred = self.get_probability_for_example(context, res_alt_tok) # get prob for alt prompt plus wrong alt prediciton

            if type(new_base_probability) is not list:
                probs = new_base_probability, new_alt_probability
                logits = new_base_logits, new_alt_logits
                hooked_predictions = hooked_base_pred, hooked_alt_pred
            else:
                probs = zip(new_base_probability, new_alt_probability)
                logits = zip(new_base_logits, new_alt_logits)
                hooked_predictions = zip(hooked_base_pred, hooked_alt_pred)
        # Remove hooks to restore original model state
        for handle in handle_list:
            handle.remove()

        return probs, logits, hooked_predictions

    def neuron_intervention2(self,
                            context,
                            res_base_tok,
                            res_alt_tok,
                            rep,
                            layers,
                            neurons,
                            position,
                            intervention_loc,
                            get_full_distribution=False,
                            is_attention=False):
        # Hook for changing representation during forward pass
        def intervention_hook(module, # model layer
                              input, # input to model layer
                              output, # output before intervention
                              position, # position of token to intervene on
                              neurons, # neurons to intervene on
                              intervention, # intervention values
                              is_attention=False):
            # Get the neurons to intervene on
            if is_attention:
                o = output[0]
            else:
                o = output
            out_device = o.device
            neurons = torch.LongTensor(neurons).to(out_device)
            # First grab the position across batch
            # Then, for each element, get correct index w/ gather
            # extract position in the output tensor where the intervention will be applied
            base_slice = (slice(None), position, slice(None))
            # extract the values from the output tensor at the specified position and neurons
            base = o[base_slice].gather(1, neurons)
            # reshape to match the shape of the base tensor
            intervention_view = intervention.view_as(base)
            # set base to intervention
            base = intervention_view

            # Overwrite values in the output
            # First define mask where to overwrite
            scatter_mask = torch.zeros_like(o, dtype=torch.bool)
            for i, v in enumerate(neurons):
                scatter_mask[(i, position, v)] = 1
            # Then take values from base and scatter and overwrite the output
            o.masked_scatter_(scatter_mask, base.flatten())

        # Set up the context as batch
        batch_size = len(neurons)
        # replicate context for each neuron in the batch
        context = context.repeat(batch_size, 1)
        handle_list = []
        for layer in set(layers):
            if intervention_loc == 'layer':
                n_list = neurons
            else:
                raise Exception(f'intervention_loc not defined: {intervention_loc}')
            m_list = n_list
            intervention_rep = rep[layer][m_list]
            if layer == -1:
                if is_attention:
                    raise Exception('Attention not implemented for word embedding intervention')

                handle_list.append(self.word_emb_layer.register_forward_hook(
                    partial(intervention_hook,
                            position=position,
                            neurons=n_list,
                            intervention=intervention_rep)))
            else:
                if is_attention:
                    module = self.get_attention_layer(layer)
                else:
                    raise NotImplementedError
                handle_list.append(module.register_forward_hook(
                    partial(intervention_hook,
                            position=position,
                            neurons=n_list,
                            intervention=intervention_rep,
                            is_attention=is_attention)))

        return handle_list

    def attention_experiment(self, interventions, effect_type, intervention_loc, position_fewshot, multitoken=False,
                             get_full_distribution=False, all_tokens=False):
        intervention_results = {}
        progress = tqdm(total=len(interventions), desc='performing interventions')
        for idx, intervention in enumerate(interventions):
            intervention_results[idx] = self.attention_intervention_single_experiment(intervention=intervention,
                                                                                      effect_type=effect_type,
                                                                                      intervention_loc=intervention_loc,
                                                                                      position_fewshot=position_fewshot,
                                                                                      get_full_distribution=get_full_distribution,
                                                                                      all_tokens=all_tokens)
            progress.update()

        return intervention_results

    def attention_intervention_single_experiment(self, intervention, effect_type, intervention_loc, position_fewshot,
                                                 get_full_distribution=False, all_tokens=False):
        """
        Run one full attention intervention experiment
        measuring indirect effect.
        """

        with torch.no_grad():
            # Probabilities without intervention (Base case)
            distrib_base, logits_base = self.get_distribution_for_example(intervention.base_string_tok)
            distrib_alt, logits_alt = self.get_distribution_for_example(intervention.alt_string_tok)
            distrib_base, logits_base = distrib_base.numpy(), logits_base.numpy()
            distrib_alt, logits_alt = distrib_alt.numpy(), logits_alt.numpy()

            # E.g. 4 plus 5 is...
            x = intervention.base_string_tok[0]
            # E.g. 1 plus 2 is...
            x_alt = intervention.alt_string_tok[0]

            #input = x_alt  # Get attention for x_alt
            #context = x
            input = x
            context = x_alt

            intervention_on_output = len(intervention_loc.split('_')) > 1 and intervention_loc.split('_')[1] == 'output'

            batch_size = 1
            seq_len = len(x)
            seq_len_alt = len(x_alt)

            if position_fewshot:
                positions = list(range(intervention.len_few_shots)) if all_tokens else [-1]
            else:
                positions = list(range(intervention.len_few_shots, len(context.squeeze()))) if all_tokens else [-1]

            res_base_probs = {}
            res_alt_probs = {}
            res_base_logits = {}
            res_alt_logits = {}
            hooked_pred_base = {}
            hooked_pred_alt = {}

            for position in positions:
                if intervention_on_output:
                    attention_override = self.get_representations(input.unsqueeze(0), position=position,
                                                                  is_attention=True)
                    assert attention_override[0].shape[0] == self.num_neurons, \
                        f'attention_override[0].shape: {attention_override[0].shape} vs {self.num_neurons}'
                else:
                    batch = input.clone().detach().unsqueeze(0).to(self.device)
                    model_output = self.model(batch, output_attentions=True)
                    #print("model_output:", model_output)
                    attention_override = model_output[-1]
                    #print("attention_override:", attention_override[0].device)
                    #print("attention_override:", attention_override)
                    assert attention_override[0].shape == (batch_size, self.num_heads, seq_len, seq_len), \
                        f'attention_override[0].shape: {attention_override[0].shape} vs ({batch_size}, {self.num_heads}, {seq_len}, {seq_len})'

                assert seq_len == seq_len_alt, f'x: [{x}] vs x_alt: [{x_alt}]'
                assert len(attention_override) == self.num_layers

                # basically generate the mask for the layers_to_adj and heads_to_adj
                if intervention_loc == 'head':
                    candidate1_probs_head = torch.zeros((self.num_layers, self.num_heads))
                    candidate2_probs_head = torch.zeros((self.num_layers, self.num_heads))
                    candidate1_logits_head = torch.zeros((self.num_layers, self.num_heads))
                    candidate2_logits_head = torch.zeros((self.num_layers, self.num_heads))

                    for layer in range(self.num_layers):
                        layer_attention_override = attention_override[layer]
                        device_id = self.get_layer_device_from_layer(layer)
                        #print("device_id:", device_id)
                        layer_attention_override = layer_attention_override.to(device_id)
                        #print("layer_attention_override:", layer_attention_override.device)

                        # one head at a time
                        for head in range(self.num_heads):
                            attention_override_mask = torch.zeros_like(layer_attention_override, dtype=torch.bool)
                            attention_override_mask=attention_override_mask.to(device_id)

                            # Set mask to 1 for single head only
                            # this should have shape (1, n_heads, seq_len, seq_len)
                            attention_override_mask[0][head] = 1

                            head_attn_override_data = [{
                                'layer': layer,
                                'attention_override': layer_attention_override,
                                'attention_override_mask': attention_override_mask
                            }]

                            candidate1_probs_head[layer][head], candidate2_probs_head[layer][
                                head], candidate1_logits_head[layer][head], candidate2_logits_head[layer][head] = self.attention_intervention(
                                context=context.unsqueeze(0),
                                res_base_tok=intervention.res_base_tok,
                                #res_alt_tok=intervention.res_alt_tok,
                                res_alt_tok=intervention.pred_res_alt_tok,
                                attn_override_data=head_attn_override_data)

                elif intervention_loc.startswith('layer'):
                    dimensions = (self.num_layers, 1)
                    res_base_probs[position] = torch.zeros(dimensions)
                    res_alt_probs[position] = torch.zeros(dimensions)
                    res_base_logits[position] = torch.zeros(dimensions)
                    res_alt_logits[position] = torch.zeros(dimensions)
                    hooked_pred_base[position] = np.full(dimensions, None)
                    hooked_pred_alt[position] = np.full(dimensions, None)

                    candidate1_probs_head = torch.zeros(dimensions)
                    candidate2_probs_head = torch.zeros(dimensions)
                    candidate1_logits_head = torch.zeros(dimensions)
                    candidate2_logits_head = torch.zeros(dimensions)

                    for layer in range(self.num_layers):
                        if intervention_on_output:
                            probs, logits, predictions = self.neuron_intervention(
                                context=context.unsqueeze(0),
                                res_base_tok=intervention.res_base_tok,
                                #res_alt_tok=intervention.res_alt_tok,
                                res_alt_tok=intervention.pred_res_alt_tok,
                                rep=attention_override,
                                layers=[layer],
                                neurons=[list(range(self.num_neurons))],
                                position=position,
                                intervention_loc='layer',
                                get_full_distribution=get_full_distribution,
                                is_attention=True)
                            p1, p2 = probs
                            l1, l2 = logits
                            pb, pa = predictions
                            res_base_probs[position][layer][0] = torch.tensor(p1)
                            res_alt_probs[position][layer][0] = torch.tensor(p2)
                            res_base_logits[position][layer][0] = torch.tensor(l1)
                            res_alt_logits[position][layer][0] = torch.tensor(l2)
                            hooked_pred_base[position][layer][0] = pb
                            hooked_pred_alt[position][layer][0] = pa
                        else:
                            layer_attention_override = attention_override[layer]
                            device_id = self.get_layer_device_from_layer(layer)
                            #print("device_id:", device_id)
                            layer_attention_override = layer_attention_override.to(device_id)
                            #print("layer_attention_override:", layer_attention_override.device)

                            # set all the head_masks in layer to 1
                            attention_override_mask = torch.ones_like(layer_attention_override, dtype=torch.bool)

                            head_attn_override_data = [{
                                'layer': layer,
                                'attention_override': layer_attention_override,
                                'attention_override_mask': attention_override_mask
                            }]

                            candidate1_probs_head[layer][0], candidate2_probs_head[layer][0], \
                                 candidate1_logits_head[layer][0], candidate2_logits_head[layer][0] = \
                                self.attention_intervention(
                                    context=context.unsqueeze(0),
                                    res_base_tok=intervention.res_base_tok,
                                    #res_alt_tok=intervention.res_alt_tok,
                                    res_alt_tok=intervention.pred_res_alt_tok,
                                    attn_override_data=head_attn_override_data)
                            res_base_probs[position][layer][0] = candidate1_probs_head[layer][0]
                            res_alt_probs[position][layer][0] = candidate2_probs_head[layer][0]
                            res_base_logits[position][layer][0] = candidate1_logits_head[layer][0]
                            res_alt_logits[position][layer][0] = candidate2_logits_head[layer][0]
                
                elif intervention_loc.startswith('single_layer'):
                    layer = int(intervention_loc.split('_')[-1])
                    dimensions = (1, self.num_heads)
                    res_base_probs[position] = torch.zeros(dimensions)
                    res_alt_probs[position] = torch.zeros(dimensions)
                    res_base_logits[position] = torch.zeros(dimensions)
                    res_alt_logits[position] = torch.zeros(dimensions)
                    hooked_pred_base[position] = np.full(dimensions, None)
                    hooked_pred_alt[position] = np.full(dimensions, None)
                    candidate1_probs_head = torch.zeros(dimensions)
                    candidate2_probs_head = torch.zeros(dimensions)
                    candidate1_logits_head = torch.zeros(dimensions)
                    candidate2_logits_head = torch.zeros(dimensions)

                    layer_attention_override = attention_override[layer]
                    device_id = self.get_layer_device_from_layer(layer)
                    layer_attention_override = layer_attention_override.to(device_id)

                    # one head at a time
                    for head in range(self.num_heads):
                        attention_override_mask = torch.zeros_like(layer_attention_override, dtype=torch.bool)
                        attention_override_mask=attention_override_mask.to(device_id)    
                        # Set mask to 1 for single head only
                        # this should have shape (1, n_heads, seq_len, seq_len)
                        attention_override_mask[0][head] = 1
                        head_attn_override_data = [{
                            'layer': layer,
                            'attention_override': layer_attention_override,
                            'attention_override_mask': attention_override_mask
                        }   ]
                        candidate1_probs_head[0][head], candidate2_probs_head[0][head], candidate1_logits_head[0][head], candidate2_logits_head[0][head] = self.attention_intervention(
                            context=context.unsqueeze(0),
                            res_base_tok=intervention.res_base_tok,
                            #res_alt_tok=intervention.res_alt_tok,
                            res_alt_tok=intervention.pred_res_alt_tok,
                            attn_override_data=head_attn_override_data)
                        
                        res_base_probs[position][0][head] = candidate1_probs_head[0][head]
                        res_alt_probs[position][0][head] = candidate2_probs_head[0][head]
                        res_base_logits[position][0][head] = candidate1_logits_head[0][head]
                        res_alt_logits[position][0][head] = candidate2_logits_head[0][head]



                else:
                    raise ValueError(f"Invalid intervention_loc: {intervention_loc}")

        return distrib_base, distrib_alt, logits_base, logits_alt, res_base_probs, res_alt_probs, res_base_logits, res_alt_logits, hooked_pred_base, hooked_pred_alt
    
    def get_layer_device_from_layer(self, layer):
        # Look up the device in hf_device_map
        key_to_search = f'gpt_neox.layers.{layer}'
        device_id = self.model.hf_device_map.get(key_to_search)
        return f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'


    def get_layer_device(self, module):
        # Iterate through named modules
        for name, mod in self.model.named_modules():
            if mod is module:
                #print(f"Module found: {name}")  # Debug print, can be removed later

                # Parse the layer number from the module name
                # Assuming the structure is 'gpt_neox.layers.X.attention'
                split_name = name.split('.')
                if split_name[0] == 'gpt_neox' and split_name[1].startswith('layers'):
                    layer_number = split_name[2]
                    key_to_search = f'gpt_neox.layers.{layer_number}'

                    # Look up the device in hf_device_map
                    device_id = self.model.hf_device_map.get(key_to_search)
                    if device_id is not None:
                        return f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
                    else:
                        # If the module is found but not in the device map, handle this case
                        print(f"Module {name} found, but not in hf_device_map")
                        break  # Stop searching if the module is found but not in the device map
                    
        # If the module's name is not found in the device map or the module is not found at all
        raise ValueError(f"Device for module not found in hf_device_map")




    def attention_intervention(self,
                               context,
                               res_base_tok,
                               res_alt_tok,
                               attn_override_data):
        """ Override attention values in specified layer
        Args:
            context: context text
            attn_override_data: list of dicts of form:
                {
                    'layer': <index of layer on which to intervene>,
                    'attention_override': <values to override the computed attention weights.
                           Shape is [batch_size, num_heads, seq_len, seq_len]>,
                    'attention_override_mask': <indicates which attention weights to override.
                                Shape is [batch_size, num_heads, seq_len, seq_len]>
                }
        """

        def intervention_hook(module, input, kwargs, outputs, attn_override, attn_override_mask):
            # attention_override_module = AttentionOverride(
            #    module, attn_override, attn_override_mask)
            attention_override_module_class = (OverrideGPTJAttention if self.is_gptj else
                                               OverrideGPTNeoXAttention if self.is_pythia else
                                               None)
            if attention_override_module_class is None:
                raise ValueError("Invalid model type")
            layer_device = self.get_layer_device(module)
            attn_override.to(layer_device)
            #print("attn_override before device:", attn_override.device)
            attn_override_mask.to(layer_device)

            attention_override_module = attention_override_module_class(
                module, attn_override, attn_override_mask
            )
            #print("module:", module)
            #print("attention_override:", attn_override)
            #print("attention_override_module:", attention_override_module)
            #print("input shape:", input[0].shape)
            #print("input 1", input[1])
            #print("kwargs:", kwargs)
            #print("outputs:", outputs)

            #print("layer_device:", layer_device)
            attention_override_module.to(layer_device)
            #print("input device:", input[0].device)

            #attention_override_module.to(self.device)

            return attention_override_module(*input, **kwargs)

        with torch.no_grad():
            hooks = []
            for d in attn_override_data:
                attn_override = d['attention_override']
                attn_override_mask = d['attention_override_mask']
                layer = d['layer']
                hooks.append(self.get_attention_layer(layer).register_forward_hook(
                    partial(intervention_hook,
                            attn_override=attn_override,
                            attn_override_mask=attn_override_mask), with_kwargs=True))

            # new probabilities are scalar
            new_base_probabilities, new_base_logits, hooked_base_pred = self.get_probability_for_example(
                context,
                res_base_tok)

            new_alt_probabilities, new_alt_logits, hooked_alt_pred = self.get_probability_for_example(
                context,
                res_alt_tok)

            for hook in hooks:
                hook.remove()

            return new_base_probabilities, new_alt_probabilities, new_base_logits, new_alt_logits, hooked_base_pred, hooked_alt_pred
        
    # summarize results
    #@staticmethod
    def process_intervention_results(self, interventions, list_of_words, intervention_results,
                                     tokenizer, args):
        results = []
        for example in intervention_results:
            distrib_base, distrib_alt, \
                logits_base, logits_alt, \
                    res_base_probs, res_alt_probs, \
                        res_base_logits, res_alt_logits, \
                             hooked_pred_base, hooked_pred_alt = intervention_results[example]

            intervention = interventions[example]

            if args.intervention_type == 20:
                res_base = intervention.res_base_string
                res_alt = intervention.res_alt_string
                res_base_idx = list_of_words.index(res_base)
                res_alt_idx = list_of_words.index(res_alt)

                res_base_base_prob = distrib_base[res_base_idx]
                res_alt_base_prob = distrib_base[res_alt_idx]
                res_base_alt_prob = distrib_alt[res_base_idx]
                res_alt_alt_prob = distrib_alt[res_alt_idx]

                pred_base_idx = np.argmax(distrib_base)
                pred_alt_idx = np.argmax(distrib_alt)
                pred_base = list_of_words[pred_base_idx]
                pred_alt = list_of_words[pred_alt_idx]

            else:
                if args.representation == 'arabic':
                    words_to_n = {convert_to_words(str(i)): i for i in range(args.max_n + 1)}
                    if intervention.res_base_string.startswith(' '):
                        adjusted_res_base_string = intervention.res_base_string[1:]
                        res_base = int(adjusted_res_base_string)
                    else:
                        res_base = int(intervention.res_base_string)
                    if intervention.pred_alt_string.startswith(' '):
                        adjusted_res_alt_string = intervention.pred_alt_string[1:]
                        try:
                            res_alt = int(adjusted_res_alt_string)
                            if res_alt > 20:
                                res_alt = self.vocab_subset.index(intervention.pred_res_alt_tok)
                        except ValueError:
                            res_alt = -1

                        if res_alt == -1:
                            res_alt = self.vocab_subset.index(intervention.pred_res_alt_tok)
                        # else:
                        #     res_alt = int(words_to_n.get(adjusted_res_alt_string, -1))
                        #     if res_alt == -1:
                        #         res_alt = int(self.vocab_subset.index(intervention.pred_res_alt_tok))
                    else:
                        try:
                            res_alt = int(intervention.pred_alt_string)
                        except ValueError:
                            res_alt = -1

                        if res_alt == -1:
                            res_alt = self.vocab_subset.index(intervention.pred_res_alt_tok)

                else:
                    words_to_n = {convert_to_words(str(i)): i for i in range(args.max_n + 1)}
                    if intervention.res_base_string.startswith(' '):
                        adjusted_res_base_string = intervention.res_base_string[1:]
                        res_base = int(words_to_n[adjusted_res_base_string])
                    else:
                        res_base = int(words_to_n[intervention.res_base_string])
                    if intervention.pred_alt_string.startswith(' '):
                        adjusted_res_alt_string = intervention.pred_alt_string[1:]
                        res_alt = int(words_to_n.get(adjusted_res_alt_string,
                                                     -1))
                        if res_alt == -1:
                            res_alt = self.vocab_subset.index(intervention.pred_res_alt_tok)
                    else:
                        # return either the integer or invalid index
                        res_alt = int(words_to_n.get(intervention.pred_alt_string, 
                                                     -1))
                        if res_alt == -1:
                            res_alt = self.vocab_subset.index(intervention.pred_res_alt_tok)
                            
                        
                        #res_alt = int(words_to_n[intervention.pred_alt_string])

                res_base_base_prob = distrib_base[res_base]
                res_alt_base_prob = distrib_base[res_alt]
                res_base_alt_prob = distrib_alt[res_base]
                res_alt_alt_prob = distrib_alt[res_alt]

                res_base_base_logits = logits_base[res_base]
                res_alt_base_logits = logits_base[res_alt]
                res_base_alt_logits = logits_alt[res_base]
                res_alt_alt_logits = logits_alt[res_alt]

                pred_base = np.argmax(distrib_base)
                pred_alt = np.argmax(distrib_alt)

                if pred_alt > args.max_n:
                    pred_alt = tokenizer.decode(self.vocab_subset[res_alt])
                pred_base = str(pred_base)
                pred_alt = str(pred_alt)

            # accuracy_10 = int(res_base in top_10_preds_base) * 0.5 + int(res_alt in top_10_preds_alt) * 0.5
            accuracy = int(pred_base == res_base) * 0.5 + int(pred_alt == res_alt) * 0.5

            metric_dict = {
                'example': example,
                'template_id': intervention.template_id,
                'n_vars': intervention.n_vars,
                'base_string': intervention.base_string,
                'alt_string': intervention.alt_string,
                'few_shots': intervention.few_shots,
                'equation': intervention.equation,
                'res_base': intervention.res_base_string,
                #'res_alt': intervention.res_alt_string,
                'res_alt': intervention.pred_alt_string,
                # base probs
                'res_base_base_prob': float(res_base_base_prob),
                'res_alt_base_prob': float(res_alt_base_prob),
                'res_base_alt_prob': float(res_base_alt_prob),
                'res_alt_alt_prob': float(res_alt_alt_prob),
                # logits
                'res_base_base_logit': float(res_base_base_logits),
                'res_alt_base_logit': float(res_alt_base_logits),
                'res_base_alt_logit': float(res_base_alt_logits),
                'res_alt_alt_logit': float(res_alt_alt_logits),
                # distribs
                'distrib_base': distrib_base,
                'distrib_alt': distrib_alt,
                # preds
                'pred_base': pred_base,
                'pred_alt': pred_alt,
                'accuracy': accuracy,
                # operands
                'operands_base': intervention.operands_base,
                'operands_alt': intervention.operands_alt
            }

            if args.all_tokens:
                if args.intervention_type == 11:
                    metric_dict.update({
                        'e1_first_pos': intervention.e1_first_pos,
                        'e1_last_pos': intervention.e1_last_pos,
                        'e2_first_pos': intervention.e2_first_pos,
                        'e2_last_pos': intervention.e2_last_pos,
                        'entity_q_first': intervention.entity_q_first,
                        'entity_q_last': intervention.entity_q_last,
                    })
                else:
                    if args.n_operands == 2:
                        metric_dict.update({
                            'op1_pos': intervention.op1_pos,
                            'op2_pos': intervention.op2_pos,
                            'operator_pos': intervention.operator_pos,
                            'operation': intervention.equation.split()[1],
                        })
                    elif args.n_operands == 3:
                        operations = intervention.equation.replace('{x}', '').replace('{y}', '').replace('{z}', '')
                        operations = operations.replace('(', '').replace(')', '')
                        metric_dict.update({
                            'op1_pos': intervention.op1_pos,
                            'op2_pos': intervention.op2_pos,
                            'op3_pos': intervention.op3_pos,
                            'operation': operations,
                        })
                    else:
                        raise NotImplementedError

            if res_base_probs is None:
                results.append(metric_dict)
            else:
                for position in res_base_probs.keys():
                    for layer in range(res_base_probs[position].size(0)):
                        if args.intervention_loc.startswith('single_layer_'):
                            layer_number = int(args.intervention_loc.split('_')[-1])
                        else:
                            layer_number = layer
                        for neuron in range(res_base_probs[position].size(1)):
                            c1_prob, c2_prob = res_base_probs[position][layer][neuron], res_alt_probs[position][layer][
                                neuron]
                            c1_logit, c2_logit = res_base_logits[position][layer][neuron], res_alt_logits[position][layer][neuron]
                            hooked_pb, hooked_pa = hooked_pred_base[position][layer][neuron], hooked_pred_alt[position][layer][neuron]
                            results_single = deepcopy(metric_dict)
                            results_single.update({
                                'position': position,
                                'layer': layer_number,
                                'neuron': neuron})
                            if args.get_full_distribution:
                                results_single['distrib_alt'] = c1_prob.numpy()
                            else:
                                results_single.update({  # strings
                                    # intervention probs
                                    'res_base_prob': float(c1_prob),
                                    'res_alt_prob': float(c2_prob),
                                    'res_base_logit': float(c1_logit),
                                    'res_alt_logit': float(c2_logit),
                                    'hooked_pred_base': hooked_pb,
                                    'hooked_pred_alt': hooked_pa
                                })
                                if 'distrib_base' in metric_dict:
                                    metric_dict.pop('distrib_base')
                                    metric_dict.pop('distrib_alt')
                            if 'few_shots' in metric_dict:
                                metric_dict.pop('few_shots')

                            results.append(results_single)

        return pd.DataFrame(results)
     