import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from scanobjectnn.configs import scanobjectnn_s, scanobjectnn_l


model_configs = {
    's': scanobjectnn_s.ModelConfig(),
    'l': scanobjectnn_l.ModelConfig(),
}
