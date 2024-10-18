from dataclasses import dataclass, field

from utils.dict_utils import ObjDict


@dataclass
class MambaConfig(dict):

    d_model: int = 256
    d_intermediate: int = 0
    n_layer: int = 4
    ssm_cfg: ObjDict = field(default_factory=ObjDict)
    attn_cfg: ObjDict = field(default_factory=ObjDict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    scan_method: str = ''
    use_mask: bool = False
    use_pos: bool = False

    @classmethod
    def default(cls):
        return MambaConfig(
                ssm_cfg=ObjDict(layer="Mamba2"),
                attn_cfg=ObjDict(num_heads=4),
                rms_norm=True,
                residual_in_fp32=True,
                scan_method='',
                use_mask=False,
                use_pos=False,
            )
