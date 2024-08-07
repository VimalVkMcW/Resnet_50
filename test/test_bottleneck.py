import torch
import pytest
import sys
from compare_pcc import PCC
sys.path.append("../")
from torchvision.models.resnet import resnet50
from Resnet.Resnet import resnet

@pytest.fixture(scope="module")
def models():
    torch_model = resnet50(weights='IMAGENET1K_V1')
    ref_model = resnet('/data/TV/dataset/resnet50-0676ba61.pth')
    return torch_model, ref_model

@pytest.mark.parametrize(
    "idx, layer, in_channels, height, width",
    [
        (0, 'layer1', 64, 56, 56),
        (1, 'layer1', 256, 56, 56),
        (2, 'layer1', 256, 56, 56),
        (0, 'layer2', 256, 28, 28),
        (1, 'layer2', 512, 28, 28),
        (2, 'layer2', 512, 28, 28),
        (3, 'layer2', 512, 28, 28),
        (0, 'layer3', 512, 14, 14),
        (1, 'layer3', 1024, 14, 14),
        (2, 'layer3', 1024, 14, 14),
        (3, 'layer3', 1024, 14, 14),
        (4, 'layer3', 1024, 14, 14),
        (5, 'layer3', 1024, 14, 14),
        (0, 'layer4', 1024, 7, 7),
        (1, 'layer4', 2048, 7, 7),
        (2, 'layer4', 2048, 7, 7),
    ]
)
def test_resnet50_bottlenecks(models, idx, layer, in_channels, height, width):
    torch_model, ref_model = models
    torch_layer = getattr(torch_model, layer)[idx]
    ref_layer = getattr(ref_model, layer)[idx]

    torch_layer.eval()
    ref_layer.eval()

    input_tensor = torch.randn(1, in_channels, height, width)

    with torch.no_grad():
        torch_output = torch_layer(input_tensor)
        ref_output = ref_layer(input_tensor)

    result, pcc_value = PCC.comp_pcc(torch_output, ref_output)
    assert result, f"Bottleneck submodule {layer}[{idx}] does not match. PCC value: {pcc_value}"
    print(f"Bottleneck submodule {layer}[{idx}] PCC value: {pcc_value}")

if __name__ == "__main__":
    pytest.main()