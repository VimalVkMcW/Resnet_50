import torch
import sys
import pytest
from compare_pcc import PCC

sys.path.append("../")
from torchvision.models.resnet import resnet50
from Resnet.Resnet import resnet

model_path = 'dataset/resnet50-0676ba61.pth'

@pytest.fixture
def load_models():
    torch_model = resnet50(weights='IMAGENET1K_V1')
    reference_model = resnet(model_path)
    return torch_model, reference_model

def test_pcc_of_resnet_models(load_models):
    torch_model, reference_model = load_models

    input_ids = torch.rand(1, 3, 224, 224)

    golden_output = torch_model(input_ids)
    reference_output = reference_model(input_ids)

    output1_flat = PCC.flatten_tuple(golden_output)
    output2_flat = PCC.flatten_tuple(reference_output)

    is_similar, pcc_value = PCC.comp_pcc(output1_flat, output2_flat)

    assert is_similar, f"PCC comparison failed: {pcc_value}"

if __name__ == "__main__":
    pytest.main()