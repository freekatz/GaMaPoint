import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from scannetv2.configs import scannetv2_s
# from s3dis.configs import s3dis_b
# from s3dis.configs import s3dis_l
# from s3dis.configs import s3dis_xl

model_configs = {
    's': (scannetv2_s.ScanNetV2Config(), scannetv2_s.ScanNetV2WarmupConfig(), scannetv2_s.GaMaConfig()),
    # 'b': (s3dis_b.ScanNetV2Config(), s3dis_b.ScanNetV2WarmupConfig(), s3dis_b.GaMaConfig()),
    # 'l': (s3dis_l.ScanNetV2Config(), s3dis_l.ScanNetV2WarmupConfig(), s3dis_l.GaMaConfig()),
    # 'xl': (s3dis_xl.ScanNetV2Config(), s3dis_xl.ScanNetV2WarmupConfig(), s3dis_xl.GaMaConfig()),
}