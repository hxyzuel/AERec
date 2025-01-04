from openai import OpenAI
import openai
import json
import random
import os
import sys
import torch
import transformers
# from peft import PeftModel
from transformers import  AutoTokenizer, AutoModelForCausalLM,GenerationConfig
from typing import Union

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

class GPT4(object):
    def __init__(self, temperature):
        super().__init__()
        self.client =OpenAI(
            api_key = ''
        )
        self.temperature = temperature
    def get_completion(self,prompt, model="gpt-4"):
        response = self.client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=self.temperature # this is the degree of randomness of the model's output
        )
        return response.choices[0].message.content


class GPT3(object):
    def __init__(self, temperature):
        super().__init__()
        self.client = OpenAI(
            api_key=''
        )
        self.temperature = temperature
    def get_completion(self, prompt, model="gpt-3.5-turbo"):
        response = self.client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=self.temperature,n=1)  # this is the degree of randomness of the model's output
        return response.choices[0].message.content

load_8bit = False
base_model = 'meta-llama/Llama-2-7b-chat-hf'
model_path = 'meta-llama/Llama-2-7b-chat-hf'
model_name = 'LLaMA2'

if torch.cuda.is_available():
    device = "cuda"

# class Prompter_LLaMA2(object):
#     __slots__ = ("template", "_verbose")
#
#     def __init__(self, verbose: bool = False):
#         super().__init__()
#
#     def generate_prompt(
#             self,
#             instruction: str,
#             input: Union[None, str] = None,
#     ) -> str:
#         if input:
#             prompt = instruction + input
#         else:
#             prompt = instruction
#         system_message = "Only return 1 or 2. just return the number,dont return else!!! Do not say 'I don't know."
#         prompt_template = f'''[INST] <<SYS>>
# {system_message}
# <</SYS>>
#
# {prompt} [/INST]'''
#
#         return prompt_template


def llama3(information):
    model_id = "../data/llms/llama3_8B_Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    messages = information
    '''
        [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": question},
    ]'''

    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    return outputs[0]["generated_text"]


# class LLama2():
#     def __init__(self,base_model,lora,lora_weights):
#         base_model = base_model
#         # model_path = model_path
#         # model_name = model_name
#         if lora ==False:
#             print('Loading tokenizer...')
#             self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
#
#             print('Loading model...')
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 base_model,
#                 load_in_8bit=load_8bit,
#                 torch_dtype=torch.float16,
#                 device_map="auto",
#                 trust_remote_code=True
#             )
#         else:
#             print('Loading tokenizer...')
#             self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
#             print('Loading model...')
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 base_model,
#                 load_in_8bit=load_8bit,
#                 torch_dtype=torch.float16,
#                 device_map="auto",
#                 trust_remote_code=True
#             )
#
#             print(f"using lora {lora_weights}")
#             self.model = PeftModel.from_pretrained(
#                 self.model,
#                 lora_weights,
#                 torch_dtype=torch.float16,
#             )
#             if not load_8bit:
#                 self.model.half()
#         self.model.eval()
#         if torch.__version__ >= "2" and sys.platform != "win32":
#             self.model = torch.compile(self.model)
#
#     def generate_prompt(
#             self,
#             instruction: str,
#             input: Union[None, str] = None,
#     ) -> str:
#         if input:
#             prompt = instruction + input
#         else:
#             prompt = instruction
#         system_message = ""
#         prompt_template = f'''[INST] <<SYS>>
# {system_message}
# <</SYS>>
#
# {prompt} [/INST]'''
#
#         return prompt_template
#     def get_response(self, output: str) -> str:
#         return output.split('[/INST]')[-1].strip(self.tokenizer.eos_token).strip()
#     def get_completion(self,
#                        instruction,
#                        input=None,
#                        temperature=0.9,
#                        top_p=0.75,
#                        top_k=40,
#                        num_beams=1,
#                        max_new_tokens=4096,
#                        **kwargs,
#                        ):
#         # streamer = TextStreamer(self.tokenizer)
#         prompt = self.generate_prompt(instruction, input)
#         inputs = self.tokenizer(prompt, return_tensors="pt")
#         input_ids = inputs["input_ids"].to(device)
#         generation_config = GenerationConfig(
#             do_sample=True,
#             temperature=temperature,
#             top_p=top_p,
#             top_k=top_k,
#             num_beams=num_beams,
#             **kwargs,
#         )
#         with torch.no_grad():
#             generation_output = self.model.generate(
#                 input_ids=input_ids,
#                 generation_config=generation_config,
#                 return_dict_in_generate=True,
#                 output_scores=True,
#                 max_new_tokens=max_new_tokens,
#                 # streamer=streamer,
#             )
#         s = generation_output.sequences[0]
#         output = self.tokenizer.decode(s, skip_special_tokens=True)
#         # return instruction, self.get_response(output)
#         return self.get_response(output)
# class discriminator_LLama2():
#     def __init__(self,base_model,lora,lora_weights):
#         base_model = base_model
#         # model_path = model_path
#         # model_name = model_name
#         if lora ==False:
#             print('Loading tokenizer...')
#             self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
#
#             print('Loading model...')
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 base_model,
#                 load_in_8bit=load_8bit,
#                 torch_dtype=torch.float16,
#                 device_map="auto",
#                 trust_remote_code=True
#             )
#         else:
#             print('Loading tokenizer...')
#             self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
#             print('Loading model...')
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 base_model,
#                 load_in_8bit=load_8bit,
#                 torch_dtype=torch.float16,
#                 device_map="auto",
#                 trust_remote_code=True
#             )
#
#             print(f"using lora {lora_weights}")
#             self.model = PeftModel.from_pretrained(
#                 self.model,
#                 lora_weights,
#                 torch_dtype=torch.float16,
#             )
#             if not load_8bit:
#                 self.model.half()
#         self.model.eval()
#         if torch.__version__ >= "2" and sys.platform != "win32":
#             self.model = torch.compile(self.model)
#
#     def generate_prompt(
#             self,
#             instruction: str,
#             input: Union[None, str] = None,
#     ) -> str:
#         if input:
#             prompt = instruction + input
#         else:
#             prompt = instruction
#         system_message = "You are a discriminator that judges whether the explainability of the recommendation system is good or bad. You should judge which of the 2 interpretability opinions generated based on the following Instruction is better. Return 1 if you think the first one is better, and 2 if you think the second one is better. Only the number 1 or 2 should be returned. Do not return any other characters."
#         prompt_template = f'''[INST] <<SYS>>
#     {system_message}
#     <</SYS>>
#
#     {prompt}Based on the above instructions, decide which explanation better explains why the recommendation system recommends this item to the customer.Please return 1 or 2 to show your choice. Only return 1 or 2. Do not return any other information. [/INST]'''
#
#         return prompt_template
#
#     def get_response(self, output: str) -> str:
#         return output.split('[/INST]')[-1].strip(self.tokenizer.eos_token).strip()
#     def get_completion(self,
#                        instruction,
#                        input=None,
#                        temperature=0.9,
#                        top_p=0.75,
#                        top_k=40,
#                        num_beams=1,
#                        max_new_tokens=4096,
#                        **kwargs,
#                        ):
#         # streamer = TextStreamer(self.tokenizer)
#         prompt = self.generate_prompt(instruction, input)
#         inputs = self.tokenizer(prompt, return_tensors="pt")
#         input_ids = inputs["input_ids"].to(device)
#         generation_config = GenerationConfig(
#             do_sample=True,
#             temperature=temperature,
#             top_p=top_p,
#             top_k=top_k,
#             num_beams=num_beams,
#             **kwargs,
#         )
#         with torch.no_grad():
#             generation_output = self.model.generate(
#                 input_ids=input_ids,
#                 generation_config=generation_config,
#                 return_dict_in_generate=True,
#                 output_scores=True,
#                 max_new_tokens=max_new_tokens,
#                 # streamer=streamer,
#             )
#         s = generation_output.sequences[0]
#         output = self.tokenizer.decode(s, skip_special_tokens=True)
#         # return instruction, self.get_response(output)
#         return self.get_response(output)
if __name__ == '__main__':
    # mygpt4 = GPT4(temperature=0.9)
    #mygpt3 = GPT3(temperature=0.9)
    # myllama2 =LLama2(base_model='/data/llms/Llama-2-7b-chat-hf',lora=False,lora_weights='')
    #myllama2 = LLama2(base_model='/data/llms/Llama-2-7b-chat-hf', lora=False, lora_weights='')
    #myllama2_fintune = LLama2(base_model='/data/llms/Llama-2-7b-chat-hf',lora=True,lora_weights='/data/llms/portrait_generation_Video_Games/Lora')
    input = [{"role": "user",
          "content": "What's you name? How are you?"}]
    # print(mygpt4.get_completion(prompt=input))
    #print(mygpt3.get_completion(prompt=input))
    print(llama3(input))
    # time.sleep(100)
