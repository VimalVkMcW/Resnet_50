import torch
import sys
import numpy as np
import pytest
from compare_pcc import PCC

sys.path.append("../")
from Resnet.Resnet import resnet
from torchvision.models.resnet import Bottleneck
from Resnet.Resnet_Bottleneck import Bottleneck as Bottleneck_R

model_path = '/data/TV/dataset/resnet50-0676ba61.pth'

@pytest.fixture
def load_model():
    model = resnet(model_path)
    return model

def get_new_state_dict(model, layer_name):
    new_state_dict = PCC.modify_state_dict_with_prefix(model, layer_name)
    return new_state_dict

@pytest.mark.parametrize("layer_name", ['layer1.1.'])
def test_pcc_of_bottleneck_blocks(load_model, layer_name):
    model = load_model
    new_state_dict = get_new_state_dict(model, layer_name)
    
    Golden_input = Bottleneck(256, 64)
    reference_input = Bottleneck_R(256, 64)

    Golden_input.load_state_dict(new_state_dict)
    reference_input.load_state_dict(new_state_dict)

    input_ids = torch.randn(1, 256, 56, 56)

    Golden_output = Golden_input(input_ids)
    reference_output = reference_input(input_ids)

    output1_flat = PCC.flatten_tuple(Golden_output)
    output2_flat = PCC.flatten_tuple(reference_output)

    is_similar, pcc_message = PCC.comp_pcc(output1_flat, output2_flat)
    
    assert is_similar, f"PCC comparison failed: {pcc_message}"

if __name__ == "__main__":
    pytest.main()