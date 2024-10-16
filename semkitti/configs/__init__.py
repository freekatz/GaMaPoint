import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from semkitti.configs import semkitti_s, semkitti_custom

model_configs = {
    's': semkitti_s.ModelConfig(),
    'c': semkitti_custom.ModelConfig(),
}
