import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from s3dis.configs import s3dis_s
# from s3dis.configs import s3dis_b
# from s3dis.configs import s3dis_l
# from s3dis.configs import s3dis_xl

model_configs = {
    's': (s3dis_s.S3disConfig(), s3dis_s.S3disWarmupConfig(), s3dis_s.GaMaConfig()),
    # 'b': (s3dis_b.S3disConfig(), s3dis_b.S3disWarmupConfig(), s3dis_b.GaMaConfig()),
    # 'l': (s3dis_l.S3disConfig(), s3dis_l.S3disWarmupConfig(), s3dis_l.GaMaConfig()),
    # 'xl': (s3dis_xl.S3disConfig(), s3dis_xl.S3disWarmupConfig(), s3dis_xl.GaMaConfig()),
}