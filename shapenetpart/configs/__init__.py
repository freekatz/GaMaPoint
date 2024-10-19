import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from shapenetpart.configs import shapenetpart_s, shapenetpart_l, shapenetpart_xl, shapenetpart_custom

model_configs = {
    's': shapenetpart_s.ModelConfig(),
    'l': shapenetpart_l.ModelConfig(),
    'xl': shapenetpart_xl.ModelConfig(),
    'c': shapenetpart_custom.ModelConfig(),
}
