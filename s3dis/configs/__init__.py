import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from s3dis.configs import s3dis_s

model_configs = {
    's': s3dis_s.ModelConfig(),
}
