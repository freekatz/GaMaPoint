import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from scanobjectnn.configs import scanobjectnn_s
# from scanobjectnn.configs import scanobjectnn_b
# from scanobjectnn.configs import scanobjectnn_l
# from scanobjectnn.configs import scanobjectnn_xl

model_configs = {
    's': (scanobjectnn_s.ScanObjectNNConfig(), scanobjectnn_s.ScanObjectNNWarmupConfig(), scanobjectnn_s.GaMaConfig()),
    # 'b': (scanobjectnn_b.ScanObjectNNConfig(), scanobjectnn_b.ScanObjectNNWarmupConfig(), scanobjectnn_b.GaMaConfig()),
    # 'l': (scanobjectnn_l.ScanObjectNNConfig(), scanobjectnn_l.ScanObjectNNWarmupConfig(), scanobjectnn_l.GaMaConfig()),
    # 'xl': (scanobjectnn_xl.ScanObjectNNConfig(), scanobjectnn_xl.ScanObjectNNWarmupConfig(), scanobjectnn_xl.GaMaConfig()),
}