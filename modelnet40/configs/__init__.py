import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from modelnet40.configs import modelnet40_s, modelnet40_l

model_configs = {
    's': modelnet40_s.ModelConfig(),
    'l': modelnet40_l.ModelConfig(),
}
