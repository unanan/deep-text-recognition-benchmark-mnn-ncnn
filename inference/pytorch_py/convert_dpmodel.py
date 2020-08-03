### Note: Use this script to convert the data-parallel model to single-parallel model
##  Refer to:

import fire
import torch
from collections import OrderedDict


# from model import Model
# from utils import AttnLabelConverter
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# imgW=100
# imgH=32
# batch_max_length=25
# character = "0123456789abcdefghijklmnopqrstuvwxyz"
# converter = AttnLabelConverter(character)
# num_class = len(converter.character)
# model = Model(imgW, imgH, num_class,batch_max_length)

# ,model=model
def convert_dpmodel(model_path, output_path):
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = k[7:]
        new_state_dict[name]=v
    # model.load_state_dict(new_state_dict)
    torch.save(new_state_dict,output_path)

if __name__ == "__main__":
    fire.Fire(convert_dpmodel)