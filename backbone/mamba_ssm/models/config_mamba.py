from dataclasses import dataclass, field


@dataclass
class MambaConfig(dict):

    d_model: int = 256
    d_intermediate: int = 0
    n_layer: int = 4
    ssm_cfg: dict = field(default_factory=dict)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    scan_method: str = ''  # ["random", "random2", "bidi"]
    use_mask: bool = True
    use_gs_fuse: bool = True

    @classmethod
    def default(cls):
        return MambaConfig(
                ssm_cfg=dict(layer="Mamba2"),
                attn_cfg=dict(num_heads=4),
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=True,
                scan_method='',
                use_mask=True,
                use_gs_fuse=True,
            )
