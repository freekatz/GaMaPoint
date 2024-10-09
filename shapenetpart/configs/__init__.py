import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from shapenetpart.configs import shapenetpart_s
# from shapenetpart.configs import shapenetpart_b
# from shapenetpart.configs import shapenetpart_l
# from shapenetpart.configs import shapenetpart_xl

model_configs = {
    's': (shapenetpart_s.ShapeNetPartConfig(), shapenetpart_s.ShapeNetPartWarmupConfig(), shapenetpart_s.GaMaConfig()),
    # 'b': (shapenetpart_b.ShapeNetPartConfig(), shapenetpart_b.ShapeNetPartWarmupConfig(), shapenetpart_b.GaMaConfig()),
    # 'l': (shapenetpart_l.ShapeNetPartConfig(), shapenetpart_l.ShapeNetPartWarmupConfig(), shapenetpart_l.GaMaConfig()),
    # 'xl': (shapenetpart_xl.ShapeNetPartConfig(), shapenetpart_xl.ShapeNetPartWarmupConfig(), shapenetpart_xl.GaMaConfig()),
}