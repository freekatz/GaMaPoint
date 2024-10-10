import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from shapenetpart.configs import shapenetpart_s

model_configs = {
    's': shapenetpart_s.ModelConfig(),
}
