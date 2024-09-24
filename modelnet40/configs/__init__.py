import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from modelnet40.configs import modelnet40_s
# from modelnet40.configs import modelnet40_b
# from modelnet40.configs import modelnet40_l
# from modelnet40.configs import modelnet40_xl

model_configs = {
    's': (modelnet40_s.ModelNet40Config(), modelnet40_s.ModelNet40WarmupConfig(), modelnet40_s.GaMaConfig()),
    # 'b': (modelnet40_b.ModelNet40Config(), modelnet40_b.ModelNet40WarmupConfig(), modelnet40_b.GaMaConfig()),
    # 'l': (modelnet40_l.ModelNet40Config(), modelnet40_l.ModelNet40WarmupConfig(), modelnet40_l.GaMaConfig()),
    # 'xl': (modelnet40_xl.ModelNet40Config(), modelnet40_xl.ModelNet40WarmupConfig(), modelnet40_xl.GaMaConfig()),
}