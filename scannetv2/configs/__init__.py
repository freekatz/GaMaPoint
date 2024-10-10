import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from scannetv2.configs import scannetv2_s

model_configs = {
    's': scannetv2_s.ModelConfig(),
}
