from itertools import count
from typing import cast
import torch
from transformers import AutoTokenizer
from transformers import GenerationConfig
from constants import HF_CACHE_DIR
from model.modeling import LSWTForCausalLM
from train_utils import add_special_tokens

def get_user_input( prompt: str, input: str ):
    prompt += (
        f'<|im_start|>user\n{input}<|im_end|>\n'
        '<|im_start|>assistant\n'
    )

    return prompt

def add_response( prompt: str, response: str ):
    prompt += f'{response}<|im_end|>\n'

    return prompt

def get_candidate( prompt, model, tokenizer, generation_config ):
    prompt_tokenized = tokenizer( prompt, add_special_tokens=False )[ 'input_ids' ]

    outputs = model.generate(
        torch.LongTensor( [ prompt_tokenized ] ).cuda(),
        generation_config=generation_config,
        use_cache=True,
        max_key_values=4095,
        return_dict_in_generate=True
    )

    return tokenizer.decode( outputs.sequences[ 0, len( prompt_tokenized ) : ].cpu(), skip_special_tokens=True )

def get_system_prompt( prompt: str ):
    return prompt + (
        '<|im_start|>system\nBelow is an instruction that describes a task. '
        'Write a response that appropriately completes the request.<|im_end|>\n'
    )

def generate( model_dir: str ):
    tokenizer = AutoTokenizer.from_pretrained( 'facebook/opt-125m', cache_dir=HF_CACHE_DIR, use_fast=True )
    add_special_tokens( tokenizer )

    model = cast(
        LSWTForCausalLM,
        LSWTForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map='cuda'
        ).eval() # type: ignore
    )

    generation_config = GenerationConfig(
        do_sample=True,
        num_beams=4,
        eos_token_id=50266,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=8.0,
        typical_p=0.95,
        no_repeat_ngram_size=5,
        # renormalize_logits=True,
        length_penalty=-0.5,
        temperature=0.707,
        max_new_tokens=1024,
        early_stopping='never',
    )

    prompt = get_system_prompt( '</s>' )

    with torch.inference_mode():
        while True:
            prompt = get_user_input( prompt, input( 'USER: ' ) )
            while True:
                response = get_candidate( prompt, model, tokenizer, generation_config )
                print( f'LOVELACE: {response}' )
                print()

                user_input = input( 'USER: ' )

                match user_input:
                    case '!retry':
                        continue
                    case '!exit':
                        exit()
                    case '!system':
                        prompt = add_response( prompt, response )
                        prompt = get_system_prompt( prompt )
                        break
                    case '!restart':
                        prompt = get_system_prompt( '</s>' )
                        break
                    case '!debug':
                        print( prompt )
                    case _:
                        prompt = add_response( prompt, response )
                        prompt = get_user_input( prompt, user_input )
                        continue



if __name__ == '__main__':
    generate( './checkpoints/lswt_medium_2k4k_sftc' )
