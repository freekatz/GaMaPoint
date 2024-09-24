import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from scannetv2.configs import scannetv2_s
# from scannetv2.configs import scannetv2_b
# from scannetv2.configs import scannetv2_l
# from scannetv2.configs import scannetv2_xl

model_configs = {
    's': (scannetv2_s.ScanNetV2Config(), scannetv2_s.ScanNetV2WarmupConfig(), scannetv2_s.GaMaConfig()),
    # 'b': (scannetv2_b.ScanNetV2Config(), scannetv2_b.ScanNetV2WarmupConfig(), scannetv2_b.GaMaConfig()),
    # 'l': (scannetv2_l.ScanNetV2Config(), scannetv2_l.ScanNetV2WarmupConfig(), scannetv2_l.GaMaConfig()),
    # 'xl': (scannetv2_xl.ScanNetV2Config(), scannetv2_xl.ScanNetV2WarmupConfig(), scannetv2_xl.GaMaConfig()),
}