import os
import shutil
import sys
import unittest

import torch
import torch.utils.cpp_extension

here = os.path.abspath(__file__)
pytorch_root = os.path.dirname(os.path.dirname(here))
api_include = os.path.join(pytorch_root, 'torch', 'csrc', 'api', 'include')
module = torch.utils.cpp_extension.load(
   name='cpp_api_extension',
   sources='cpp_extensions/cpp_api_extension.cpp',
   extra_include_paths=api_include,
   verbose=True)

net = module.Net(3, 5)
