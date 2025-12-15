import os 

import torch 
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save, torch_save_to_dcp


def convert_pth_to_fsdp(pth_path, new_fdsp_folder_path):
    torch_save_to_dcp(pth_path, new_fdsp_folder_path)

# convert to fsdp after renaming
def rename_pth_file(pth_path, new_pth_path):
    oldpth = torch.load(pth_path)    
    print(f'oldpth keys: {oldpth.keys()}')
    oldpth = oldpth['model']
    newpth = {}   
    for old_key in oldpth:

        if 'prediction_head' in old_key:
            continue

        new_key = 'decoder.'+old_key
        # if 'blocks' not in old_key:
        #     new_key = 'decoder.'+old_key
        # else:
        #     new_key = old_key.replace('blocks', 'decoder.blocks.0')
        newpth[new_key] = oldpth[old_key]
    
                
        if 'dec_blocks.' in old_key:
            new_key = old_key.replace('dec_blocks.', 'dec_blocks2.')
            new_key = 'decoder.'+new_key
            newpth[new_key] = oldpth[old_key]

    #newpth = {'model': newpth}
    torch.save(newpth, new_pth_path)


if __name__ == '__main__':
    rename_pth_file('../pretrained/CroCo_V2_ViTLarge_BaseDecoder.pth', '../pretrained/renamed-croco.pth')
    convert_pth_to_fsdp('../pretrained/renamed-croco.pth', '../pretrained/croco_pretrain_fsdp')