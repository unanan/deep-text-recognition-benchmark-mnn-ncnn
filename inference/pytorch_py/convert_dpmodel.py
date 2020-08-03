### Note: Use this script to convert the data-parallel model to single-parallel model
##  Refer to:

import fire
import torch
from collections import OrderedDict


def convert_dpmodel(model_path, output_path):
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = k[7:]
        new_state_dict[name]=v
    torch.save(new_state_dict,output_path)

if __name__ == "__main__":
    fire.Fire(convert_dpmodel)