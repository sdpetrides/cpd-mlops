#!/usr/bin/env python

import os
import sys
import json
import importlib
import traceback
import subprocess

from datetime import datetime, timezone

import redhareapiversion
from redhareapi import Kernel

from utils import install_requirements
from utils.prompter import Prompter

dir_user = os.environ['REDHARE_MODEL_PATH']

print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %z')}: Setting Up ...")

# Setup environment variables and paths
deploy_name = os.environ['REDHARE_MODEL_NAME']
dir_user = os.environ['REDHARE_MODEL_PATH']

# Change directory
os.chdir(dir_user)

# Setup working directory
dir_work = f"/opt/wml-edi/work/{deploy_name}"

# Setup python package directory
dir_python_pkg = f"{dir_work}/python_packages"
os.makedirs(dir_python_pkg, exist_ok=True)

# Setup up pip cache directory
dir_pip_cache = f"{dir_python_pkg}/pip_cache"
os.environ['XDG_CACHE_HOME'] = dir_pip_cache
os.environ['PIP_CACHE_HOME'] = dir_pip_cache
os.makedirs(dir_pip_cache, exist_ok=True)

# Install packages and add python install directory into PATH
install_requirements(dir_python_pkg)
sys.path.insert(0, dir_python_pkg)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Import packages
import torch
import transformers

import accelerate
import bitsandbytes

from peft import PeftModel
from transformers import (
    GenerationConfig, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
)

print(f"CUDA version: {torch.version.cuda}")


class MatchKernel(Kernel):

    def on_kernel_start(self, kernel_context):
        Kernel.log_info("on_kernel_start kernel id: " + kernel_context.get_id())
        if os.path.exists("/mnts/llm"):
            Kernel.log_info("/mnts/llm exists")
        else:
            Kernel.log_error("/mnts/llm does not exist")

    def on_task_invoke(self, task_context):
        Kernel.log_info("on_task_invoke")
        output_data = {}
        try:
            input_data = json.loads(task_context.get_input_data())

            # request tasks go here
            output_data["response"] = "Success!"

            task_context.set_output_data(json.dumps(output_data))

        except Exception as e:
            Kernel.log_error("something went wrong")
            traceback.print_exc()
            output_data['msg'] = str(e)
            task_context.set_output_data(json.dumps(output_data))

    def on_kernel_shutdown(self):
        pass


if __name__ == '__main__':
    obj_kernel = MatchKernel()
    obj_kernel.run()

