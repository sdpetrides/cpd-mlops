#!/usr/bin/env python

import os
import sys
import json

from datetime import datetime, timezone
from typing import Union

import redhareapiversion
from redhareapi import Kernel

# from utils import install_requirements

dir_user = os.environ['REDHARE_MODEL_PATH']

print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %z')}: Setting Up ...")

# # Setup environment variables and paths
# deploy_name = os.environ['REDHARE_MODEL_NAME']
# dir_user = os.environ['REDHARE_MODEL_PATH']

# # Change directory
# os.chdir(dir_user)

# # Setup working directory
# dir_work = f"/opt/wml-edi/work/{deploy_name}"

# # Setup python package directory
# dir_python_pkg = f"{dir_work}/python_packages"
# os.makedirs(dir_python_pkg, exist_ok=True)

# # Setup up pip cache directory
# dir_pip_cache = f"{dir_python_pkg}/pip_cache"
# os.environ['XDG_CACHE_HOME'] = dir_pip_cache
# os.environ['PIP_CACHE_HOME'] = dir_pip_cache
# os.makedirs(dir_pip_cache, exist_ok=True)

# # Install packages and add python install directory into PATH
# install_requirements(dir_python_pkg)
# sys.path.insert(0, dir_python_pkg)

# # Import packages
# import torch

# print(f"CUDA version: {torch.version.cuda}")

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import torch
# import transformers

# import accelerate
# import bitsandbytes

# from peft import PeftModel
# from transformers import (
#     GenerationConfig,
#     LlamaForCausalLM,
#     LlamaTokenizer,
#     BitsAndBytesConfig
# )


# class Prompter(object):
#     __slots__ = ("template", "_verbose")

#     def __init__(self, template_name: str = "", verbose: bool = False):
#         self._verbose = verbose
#         if not template_name:
#             # Enforce the default here, so the constructor
#             # can be called with '' and will not break.
#             template_name = "alpaca"
#         file_name = os.path.join("./templates", f"{template_name}.json")
#         if not os.path.exists(file_name):
#             raise ValueError(f"Can't read {file_name}")
#         with open(file_name) as fp:
#             self.template = json.load(fp)
#         if self._verbose:
#             print(
#                 f"Using prompt template {template_name}:"
#                 f"{self.template['description']}"
#             )

#     def generate_prompt(
#         self,
#         instruction: str,
#         input: Union[None, str] = None,
#         label: Union[None, str] = None,
#     ) -> str:
#         # returns the full prompt from instruction and optional input
#         # if a label (=response, =output) is provided, it's also appended.
#         if input:
#             res = self.template["prompt_input"].format(
#                 instruction=instruction, input=input
#             )
#         else:
#             res = self.template["prompt_no_input"].format(
#                 instruction=instruction
#             )
#         if label:
#             res = f"{res}{label}"
#         if self._verbose:
#             print(res)
#         return res

#     def get_response(self, output: str) -> str:
#         return output.split(self.template["response_split"])[1].strip()


class MatchKernel(Kernel):

    def on_kernel_start(self, kernel_context):
        Kernel.log_info("MatchKernel on_kernel_start")
        Kernel.log_info(str(kernel_context))

        self.device = "cuda:0"
        self.base_model = "yahma/llama-7b-hf"
        self.lora_weights = "yahma/alpaca-7b-lora"
        self.cache_directory = "/mnts/llm"

        if os.path.exists(self.cache_directory):
            Kernel.log_info(f"{self.cache_directory} exists")
        else:
            Kernel.log_error(f"{self.cache_directory} does not exist")

#         try:
#             self.prompter = Prompter("")
#             self.tokenizer = LlamaTokenizer.from_pretrained(self.base_model)

#             Kernel.log_info("MatchKernel on_kernel_start before from_pretrained")

#             model = LlamaForCausalLM.from_pretrained(
#                 self.base_model,
#                 load_in_8bit=True,
#                 torch_dtype=torch.float16,
#                 device_map="auto",
#                 cache_dir=self.cache_directory,
#             )
#             Kernel.log_info("MatchKernel on_kernel_start after from_pretrained")
#             Kernel.log_info("MatchKernel on_kernel_start before from_pretrained")

#             model = PeftModel.from_pretrained(model, self.lora_weights)

#             Kernel.log_info("MatchKernel on_kernel_start after from_pretrained")

#             model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
#             model.config.bos_token_id = 1
#             model.config.eos_token_id = 2

#             self.model = torch.compile(model)
#         except Exception e:
#             Kernel.log_error(str(e))

    def on_task_invoke(self, task_context):
        Kernel.log_info("MatchKernel on_task_invoke")
        task_context.set_output_data(json.dumps({}))
        # output_data = {}
#         try:
#             # Read input
#             input_data = json.loads(task_context.get_input_data())
#             instruction = input_data.get("instruction", "")
#             temperature = input_data.get("temperature", 0.1)
#             top_p = input_data.get("top_p", 0.75)
#             top_k = input_data.get("top_k", 40)
#             num_beams = input_data.get("num_beams", 4)
#             max_new_tokens = input_data.get("max_new_tokens", 128)

#             # Generate input tokens
#             prompt = self.prompter.generate_prompt(instruction, None)
#             inputs = self.tokenizer(prompt, return_tensors="pt")
#             input_ids = inputs["input_ids"].to(self.device)

#             # Set params
#             generation_config = GenerationConfig(
#                 temperature=temperature,
#                 top_p=top_p,
#                 top_k=top_k,
#                 num_beams=num_beams,
#             )

#             with torch.no_grad():
#                 generation_output = self.model.generate(
#                     input_ids=input_ids,
#                     generation_config=generation_config,
#                     return_dict_in_generate=True,
#                     output_scores=True,
#                     max_new_tokens=max_new_tokens,
#                 )

#             s = generation_output.sequences[0]
#             output = self.tokenizer.decode(s)
#             output_data["text"] = output

#             task_context.set_output_data(json.dumps(output_data))

#         except Exception as e:
#             traceback.print_exc()
#             output_data['msg'] = str(e)
#             task_context.set_output_data(json.dumps(output_data))

    def on_kernel_shutdown(self):
        pass


if __name__ == '__main__':
    obj_kernel = MatchKernel()
    obj_kernel.run()
