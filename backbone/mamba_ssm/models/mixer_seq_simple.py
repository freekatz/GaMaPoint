# Copyright (c) 2023, Albert Gu, Tri Dao.

import copy
import math
from functools import partial

import torch
import torch.nn as nn
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mlp import GatedMLP

from utils.dict_utils import ObjDict
from ..custom.order import Order
from ..custom.structured_mask import StructuredMask
from ..modules.mamba2 import Mamba2
from ..modules.mha import MHA
from ...gs_3d import NaiveGaussian3D

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def create_block(
        d_model,
        d_intermediate,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
) -> (Block, str):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    block_type = 'Mamba1'
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba2")
        if ssm_layer not in ["Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba2")

        mixer_cls = partial(
            Mamba2,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )

        factory_kwargs = {"device": device, "dtype": dtype}
        block_type = 'Mamba2' if ssm_layer == "Mamba2" else 'Mamba1'
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
        block_type = 'MHA'
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block, block_type


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            d_intermediate: int,
            ssm_cfg=None,
            attn_layer_idx=None,
            attn_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.n_layer = n_layer
        self.layers = nn.ModuleList()
        self.layers_name = ObjDict()
        for i in range(n_layer):
            layer_idx = i
            block, block_type = create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=layer_idx,
                    **factory_kwargs,
            )
            layer_name = f'layer_{i}_{layer_idx}_{block_type}_1'
            self.layers.add_module(
                layer_name,
                block
            )
            self.layers_name[f'{i}_block_1'] = layer_name

            # block, block_type = create_block(
            #     d_model,
            #     d_intermediate=d_intermediate,
            #     ssm_cfg=ssm_cfg,
            #     attn_layer_idx=attn_layer_idx,
            #     attn_cfg=attn_cfg,
            #     norm_epsilon=norm_epsilon,
            #     rms_norm=rms_norm,
            #     residual_in_fp32=residual_in_fp32,
            #     fused_add_norm=fused_add_norm,
            #     layer_idx=layer_idx,
            #     **factory_kwargs,
            # )
            # layer_name = f'layer_{i}_{layer_idx}_{block_type}_2'
            # self.layers.add_module(
            #     layer_name,
            #     block
            # )
            # self.layers_name[f'{i}_block_2'] = layer_name

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @staticmethod
    def inverse_permutation(perm):
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.size(0), device=perm.device)
        return inv

    def __get_layer_by_name(self, layer_shortname):
        return self.layers.get_submodule(self.layers_name[layer_shortname])

    def forward(self, input_ids, pos_embed=None, inference_params=None,
                mask: StructuredMask = None, gs: NaiveGaussian3D = None,
                order: Order = None, **mixer_kwargs):
        hidden_states = input_ids
        residual = None

        for idx in range(self.n_layer):
            block1 = self.__get_layer_by_name(f'{idx}_block_1')
            if pos_embed is not None:
                hidden_states = hidden_states + pos_embed
            hidden_states1, residual1 = block1(
                hidden_states, residual, inference_params=inference_params, mask=mask
            )
            if order is not None:
                block2 = self.__get_layer_by_name(f'{idx}_block_2')
                hidden_states2, residual2 = block2(
                    order.sort(hidden_states), None if residual is None else order.sort(residual), inference_params=inference_params, mask=mask
                )
                hidden_states2 = order.inv_sort(hidden_states2)
                residual2 = order.inv_sort(residual2)
                hidden_states = hidden_states1 + hidden_states2
                residual = residual1 + residual2
            else:
                hidden_states = hidden_states1
                residual = residual1

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states
