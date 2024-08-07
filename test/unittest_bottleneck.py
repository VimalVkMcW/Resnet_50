import unittest
import torch
from compare_pcc import PCC
import sys
sys.path.append("../")
from torchvision.models.resnet import resnet50
from Resnet.Resnet import resnet

class TestResNet50Bottlenecks(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.torch_model = resnet50(weights='IMAGENET1K_V1')
        cls.ref_model = resnet('/data/TV/dataset/resnet50-0676ba61.pth')
    
    def test_resnet50_bottlenecks(self):
        test_cases = [
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
        
        for idx, layer, in_channels, height, width in test_cases:
            with self.subTest(idx=idx, layer=layer, in_channels=in_channels, height=height, width=width):
                Bottleneck = getattr(self.torch_model, layer)[idx]
                Bottleneck_R = getattr(self.ref_model, layer)[idx]

                input_tensor = torch.randn(1, in_channels, height, width)

                with torch.no_grad():
                    torch_output = Bottleneck(input_tensor)
                    ref_output = Bottleneck_R(input_tensor)

                is_similiar, pcc_value = PCC.comp_pcc(torch_output, ref_output)
                self.assertTrue(is_similiar, f"Bottleneck submodules {layer}[{idx}] does not match  ||  PCC value: {pcc_value}")
                print(f"Bottleneck submodules {layer}[{idx}] matched    ||  PCC value: {pcc_value}")

if __name__ == "__main__":
    unittest.main()
