import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from modelnet40.configs import modelnet40_s, modelnet40_custom

model_configs = {
    's': modelnet40_s.ModelConfig(),
    'c': modelnet40_custom.ModelConfig(),
}
