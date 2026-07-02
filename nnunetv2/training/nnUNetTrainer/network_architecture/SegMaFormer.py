import math
import yaml
import torch
from torch import Tensor, nn
import torch.nn.functional as F

SUPPORTED_BLOCK_MODES = {"hybrid", "hybrid_prereduce", "mamba", "mamba_prereduce", "conv"}
SUPPORTED_ENCODER_BLOCK_TYPES = {"attention", *SUPPORTED_BLOCK_MODES}


def _normalize_block_mode(block_mode: str) -> str:
    mode = str(block_mode).strip().lower()
    if mode not in SUPPORTED_BLOCK_MODES:
        raise ValueError(
            f"Unsupported block_mode '{block_mode}'. Expected one of {sorted(SUPPORTED_BLOCK_MODES)}."
        )
    return mode


def _normalize_encoder_block_type(block_type: str) -> str:
    mode = str(block_type).strip().lower()
    if mode not in SUPPORTED_ENCODER_BLOCK_TYPES:
        raise ValueError(
            f"Unsupported encoder block type '{block_type}'. Expected one of {sorted(SUPPORTED_ENCODER_BLOCK_TYPES)}."
        )
    return mode


def _normalize_explicit_stage_schedule(stage_block_types, depths):
    if stage_block_types is None:
        return None

    if len(stage_block_types) != len(depths):
        raise ValueError(
            f"stage_block_types must have {len(depths)} entries, got {len(stage_block_types)}."
        )

    normalized_schedule = []
    for stage_idx, (stage_spec, depth) in enumerate(zip(stage_block_types, depths)):
        if isinstance(stage_spec, (list, tuple)):
            if len(stage_spec) != depth:
                raise ValueError(
                    f"Stage {stage_idx + 1} block schedule must have depth {depth}, got {len(stage_spec)}."
                )
            normalized_schedule.append([
                _normalize_encoder_block_type(block_type)
                for block_type in stage_spec
            ])
        else:
            normalized_schedule.append([
                _normalize_encoder_block_type(stage_spec)
            ] * depth)
    return normalized_schedule


def _build_encoder_block_schedule(
    depths,
    default_stage_block_mode: str,
    stage_block_types=None,
    num_mamba_replacements=None,
    replacement_block_mode="mamba",
):
    explicit_stage_schedule = _normalize_explicit_stage_schedule(stage_block_types, depths)
    if explicit_stage_schedule is not None:
        if num_mamba_replacements is not None:
            raise ValueError(
                "stage_block_types and num_mamba_replacements are mutually exclusive schedule controls."
            )
        return None, explicit_stage_schedule

    total_blocks = sum(depths)
    if num_mamba_replacements is None:
        return [
            default_stage_block_mode,
            default_stage_block_mode,
            default_stage_block_mode,
            "attention",
        ], None

    num_mamba_replacements = int(num_mamba_replacements)
    if not 0 <= num_mamba_replacements <= total_blocks:
        raise ValueError(
            f"num_mamba_replacements must be in [0, {total_blocks}], got {num_mamba_replacements}."
        )

    replacement_block_mode = _normalize_block_mode(replacement_block_mode)
    flat_schedule = ["attention"] * total_blocks
    # Replace deepest blocks first so the ablation progressively swaps global mixers
    # starting from lower-resolution stages where sequence modeling is cheapest.
    for idx in range(num_mamba_replacements):
        flat_schedule[total_blocks - 1 - idx] = replacement_block_mode

    stage_schedule = []
    start = 0
    for depth in depths:
        stage_schedule.append(flat_schedule[start:start + depth])
        start += depth
    return None, stage_schedule


def _build_encoder_block(
    block_type: str,
    dim: int,
    num_heads: int,
    sr_ratio: int,
    d_state: int,
    d_conv: int,
    expand: int,
    mlp_ratio: int,
):
    block_type = _normalize_encoder_block_type(block_type)
    if block_type == "attention":
        return TransformerBlock(dim, num_heads, mlp_ratio, sr_ratio)

    # Keep spatial-reduction strictly as an attention-only mechanism.
    # Mamba and hybrid blocks always operate at full stage resolution.
    return MambaBlock3D(
        dim=dim,
        sr_ratio=1,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        mlp_ratio=mlp_ratio,
        block_mode=block_type,
    )


def _normalize_spatial_shape_args(*spatial_shape):
    if len(spatial_shape) == 1:
        shape = spatial_shape[0]
        if isinstance(shape, (tuple, list)):
            if len(shape) != 3:
                raise ValueError(f"Expected 3D spatial shape, got {shape}.")
            return tuple(shape)
    elif len(spatial_shape) == 3:
        return tuple(spatial_shape)

    raise ValueError(f"Expected spatial shape as (D, H, W) or D, H, W, got {spatial_shape}.")


def _normalize_sr_ratio(sr_ratio):
    if isinstance(sr_ratio, int):
        if sr_ratio < 1:
            raise ValueError(f"sr_ratio must be >= 1, got {sr_ratio}.")
        return (sr_ratio, sr_ratio, sr_ratio)

    if isinstance(sr_ratio, (tuple, list)) and len(sr_ratio) == 3:
        ratio = tuple(int(v) for v in sr_ratio)
        if any(v < 1 for v in ratio):
            raise ValueError(f"sr_ratio entries must be >= 1, got {sr_ratio}.")
        return ratio

    raise ValueError(f"Expected sr_ratio as int or 3D tuple/list, got {sr_ratio}.")


class AuxiliaryHead3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        mid = max(in_channels // 2, 1)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, mid, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, num_classes, kernel_size=1),
        )

    def forward(self, x, output_size=None):
        x = self.block(x)
        if output_size is not None:
            x = F.interpolate(x, size=output_size, mode="trilinear", align_corners=False)
        return x


class RoPE3D(nn.Module):
    """
    3D RoPE that works for any channel size.
    It rotates the largest valid prefix and leaves remainder channels unchanged.
    """
    def __init__(self, theta: float = 10000.0):
        super().__init__()
        self.theta = theta
        self.register_buffer("cos_z", None, persistent=False)
        self.register_buffer("sin_z", None, persistent=False)
        self.register_buffer("cos_y", None, persistent=False)
        self.register_buffer("sin_y", None, persistent=False)
        self.register_buffer("cos_x", None, persistent=False)
        self.register_buffer("sin_x", None, persistent=False)
        self.register_buffer("cache_D", torch.tensor(-1), persistent=False)
        self.register_buffer("cache_H", torch.tensor(-1), persistent=False)
        self.register_buffer("cache_W", torch.tensor(-1), persistent=False)
        self.register_buffer("cache_half", torch.tensor(-1), persistent=False)

    def _rotary_dims(self, c: int):
        # Largest multiple of 6 not exceeding c, with even per-axis split.
        c_rot = (c // 6) * 6
        if c_rot == 0:
            return 0, 0, c
        c_axis = c_rot // 3
        half = c_axis // 2
        c_axis = half * 2
        c_rot = c_axis * 3
        remainder = c - c_rot
        return c_rot, half, remainder

    def _update_cache(self, D: int, H: int, W: int, half_dim: int, device):
        if half_dim == 0:
            return
        if (
            self.cos_z is not None
            and self.cache_D.item() == D
            and self.cache_H.item() == H
            and self.cache_W.item() == W
            and self.cache_half.item() == half_dim
            and self.cos_z.device == device
        ):
            return

        pos_z = torch.arange(D, device=device, dtype=torch.float32)
        pos_y = torch.arange(H, device=device, dtype=torch.float32)
        pos_x = torch.arange(W, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(half_dim, device=device, dtype=torch.float32) / max(half_dim, 1))
        )

        freqs_z = pos_z[:, None] * inv_freq[None, :]
        freqs_y = pos_y[:, None] * inv_freq[None, :]
        freqs_x = pos_x[:, None] * inv_freq[None, :]

        self.cos_z = freqs_z.cos()[:, None, None, :]
        self.sin_z = freqs_z.sin()[:, None, None, :]
        self.cos_y = freqs_y.cos()[None, :, None, :]
        self.sin_y = freqs_y.sin()[None, :, None, :]
        self.cos_x = freqs_x.cos()[None, None, :, :]
        self.sin_x = freqs_x.sin()[None, None, :, :]

        self.cache_D.copy_(torch.tensor(D, device=self.cache_D.device))
        self.cache_H.copy_(torch.tensor(H, device=self.cache_H.device))
        self.cache_W.copy_(torch.tensor(W, device=self.cache_W.device))
        self.cache_half.copy_(torch.tensor(half_dim, device=self.cache_half.device))

    def forward(self, x, D=None, H=None, W=None):
        if x.dim() == 5:
            B, D, H, W, C = x.shape
            x = x.view(B, -1, C)
        else:
            B, N, C = x.shape
            assert D is not None and H is not None and W is not None
            assert N == D * H * W

        c_rot, half, remainder = self._rotary_dims(C)
        if c_rot == 0:
            return x

        self._update_cache(D, H, W, half, x.device)
        x = x.view(B, D, H, W, C)

        x_rot = x[..., :c_rot]
        x_tail = x[..., c_rot:] if remainder > 0 else None

        c_axis = c_rot // 3
        xz, xy, xx = torch.split(x_rot, c_axis, dim=-1)

        def apply_rope(xi, cos, sin):
            x1, x2 = xi[..., :half], xi[..., half:]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        out = torch.cat(
            [
                apply_rope(xz, self.cos_z, self.sin_z),
                apply_rope(xy, self.cos_y, self.sin_y),
                apply_rope(xx, self.cos_x, self.sin_x),
            ],
            dim=-1,
        )
        if x_tail is not None:
            out = torch.cat([out, x_tail], dim=-1)
        return out.view(B, -1, C)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channel, embed_dim, kernel_size, stride, padding, use_rope: bool = True):
        super().__init__()
        self.proj = nn.Conv3d(in_channel, embed_dim, kernel_size, stride, padding)
        self.norm = nn.LayerNorm(embed_dim)
        self.rope = RoPE3D() if use_rope else None

    def forward(self, x):
        x = self.proj(x)
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        if self.rope is not None:
            x = self.rope(x, D, H, W)
        else:
            x = x.view(B, -1, C)
        x = self.norm(x)
        return x, (D, H, W)


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, groups=dim, bias=True)
        self.bn = nn.InstanceNorm3d(dim, affine=True)

    def forward(self, x, spatial_shape):
        B, N, C = x.shape
        D, H, W = spatial_shape
        x = x.transpose(1, 2).contiguous().reshape(B, C, D, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class _MLP(nn.Module):
    def __init__(self, in_feature, mlp_ratio=2, dropout=0.0):
        super().__init__()
        hidden = int(in_feature * mlp_ratio)
        self.fc1 = nn.Linear(in_feature, hidden)
        self.dwconv = DWConv(hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, in_feature)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, spatial_shape):
        x = self.fc1(x)
        x = self.dwconv(x, spatial_shape)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        sr_ratio=1,
        qkv_bias=False,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.key_value = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.sr_ratio = _normalize_sr_ratio(sr_ratio)
        self.use_sr = any(v > 1 for v in self.sr_ratio)
        if self.use_sr:
            self.sr = nn.Conv3d(embed_dim, embed_dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, spatial_shape):
        B, N, C = x.shape
        D, H, W = spatial_shape

        q = self.query(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        if self.use_sr:
            x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, D, H, W)
            x_ = self.sr(x_)
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.sr_norm(x_)
            kv = self.key_value(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.key_value(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=2, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = _MLP(embed_dim, mlp_ratio)

    def forward(self, x, *spatial_shape):
        spatial_shape = _normalize_spatial_shape_args(*spatial_shape)
        x = x + self.attn(self.norm1(x), spatial_shape)
        x = x + self.mlp(self.norm2(x), spatial_shape)
        return x


class LocalConvTokenBranch3D(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.dw = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.norm = nn.InstanceNorm3d(dim, affine=True)
        self.act = nn.GELU()
        self.pw = nn.Conv3d(dim, dim, kernel_size=1, bias=False)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, spatial_shape):
        B, N, C = x.shape
        D, H, W = spatial_shape
        x = x.transpose(1, 2).contiguous().view(B, C, D, H, W)
        x = self.dw(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.drop(x)
        return x.flatten(2).transpose(1, 2).contiguous()


class MambaTokenWrapper(nn.Module):
    def __init__(self, dim, d_state=8, d_conv=3, expand=2):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as e:
            raise ImportError("Please install mamba-ssm: pip install mamba-ssm") from e
        self.block = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class MambaBlock3D(nn.Module):
    """
    Local-global Mamba block:
    - Mamba branch captures long-range/global token dependencies
    - 3D depthwise conv branch captures local volumetric structure
    - learned fusion adaptively balances global and local features
    - residual MLP refines fused tokens
    """
    def __init__(
        self,
        dim,
        sr_ratio=1,
        d_state=8,
        d_conv=3,
        expand=2,
        mlp_ratio=2,
        dropout=0.0,
        layer_scale_init=1e-4,
        block_mode="hybrid",
    ):
        super().__init__()
        self.block_mode = _normalize_block_mode(block_mode)
        self.sr_ratio = _normalize_sr_ratio(sr_ratio)
        self.use_sr = any(v > 1 for v in self.sr_ratio)
        self.use_prereduce = self.block_mode in {"hybrid_prereduce", "mamba_prereduce"}
        self.use_mamba = self.block_mode in {"hybrid", "hybrid_prereduce", "mamba", "mamba_prereduce"}
        self.use_conv = self.block_mode in {"hybrid", "hybrid_prereduce", "conv"}

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        if self.use_prereduce and self.use_sr:
            self.prereduce = nn.Conv3d(
                dim,
                dim,
                kernel_size=self.sr_ratio,
                stride=self.sr_ratio,
            )
            self.prereduce_norm = nn.LayerNorm(dim)
        else:
            self.prereduce = None
            self.prereduce_norm = None

        if self.use_mamba:
            self.mamba = MambaTokenWrapper(
                dim=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mamba = None

        if self.use_conv:
            self.conv_branch = LocalConvTokenBranch3D(
                dim=dim,
                dropout=dropout,
            )
        else:
            self.conv_branch = None

        if self.block_mode in {"hybrid", "hybrid_prereduce"}:
            self.gate = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

        self.fuse = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

        self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(dim))

        self.mlp = _MLP(
            in_feature=dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(dim))

    def _reduce_tokens(self, x: Tensor, spatial_shape):
        if self.prereduce is None:
            return x, spatial_shape

        B, _, C = x.shape
        D, H, W = spatial_shape
        x_3d = x.transpose(1, 2).contiguous().reshape(B, C, D, H, W)
        x_3d = self.prereduce(x_3d)
        reduced_shape = x_3d.shape[2:]
        x_tokens = x_3d.flatten(2).transpose(1, 2).contiguous()
        x_tokens = self.prereduce_norm(x_tokens)
        return x_tokens, reduced_shape

    def _upsample_tokens(self, x: Tensor, source_shape, target_shape):
        if source_shape == target_shape:
            return x

        B, _, C = x.shape
        x_3d = x.transpose(1, 2).contiguous().reshape(B, C, *source_shape)
        x_3d = F.interpolate(
            x_3d,
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )
        return x_3d.flatten(2).transpose(1, 2).contiguous()

    def _run_mamba_branch(self, x: Tensor):
        if self.mamba is None:
            raise RuntimeError("Mamba branch requested but not initialized.")
        return self.mamba(x)

    def forward(self, x: Tensor, *spatial_shape) -> Tensor:
        D, H, W = _normalize_spatial_shape_args(*spatial_shape)
        z = self.norm1(x)

        if self.block_mode == "hybrid":
            y_m = self._run_mamba_branch(z)
            y_c = self.conv_branch(z, (D, H, W))
            if self.gate is not None:
                gate = self.gate(torch.cat([y_m, y_c], dim=-1))
                y = gate * y_m + (1.0 - gate) * y_c
            else:
                y = 0.5 * (y_m + y_c)
            y = self.fuse(y)
            y = self.drop(y)
        elif self.block_mode == "hybrid_prereduce":
            z_reduced, reduced_shape = self._reduce_tokens(z, (D, H, W))
            y_m = self.mamba(z_reduced)
            y_c = self.conv_branch(z_reduced, reduced_shape)
            if self.gate is not None:
                gate = self.gate(torch.cat([y_m, y_c], dim=-1))
                y = gate * y_m + (1.0 - gate) * y_c
            else:
                y = 0.5 * (y_m + y_c)
            y = self.fuse(y)
            y = self.drop(y)
            y = self._upsample_tokens(y, reduced_shape, (D, H, W))
        elif self.block_mode == "mamba_prereduce":
            z_reduced, reduced_shape = self._reduce_tokens(z, (D, H, W))
            y = self.mamba(z_reduced)
            y = self.fuse(y)
            y = self.drop(y)
            y = self._upsample_tokens(y, reduced_shape, (D, H, W))
        elif self.block_mode == "mamba":
            y = self._run_mamba_branch(z)
            y = self.fuse(y)
            y = self.drop(y)
        else:
            y = self.conv_branch(z, (D, H, W))
            y = self.fuse(y)
            y = self.drop(y)

        x = x + y * self.gamma1.view(1, 1, -1)

        z2 = self.norm2(x)
        if self.use_prereduce and self.prereduce is not None:
            z2_reduced, reduced_shape = self._reduce_tokens(z2, (D, H, W))
            y2 = self.mlp(z2_reduced, reduced_shape)
            y2 = self._upsample_tokens(y2, reduced_shape, (D, H, W))
        else:
            y2 = self.mlp(z2, (D, H, W))
        x = x + y2 * self.gamma2.view(1, 1, -1)

        return x


class MixVisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_dims,
        patch_kernel_size,
        patch_stride,
        patch_padding,
        mlp_ratios,
        num_heads,
        sr_ratios,
        depths,
        d_state,
        d_conv,
        expand,
        block_mode="hybrid",
        use_rope=True,
        stage_block_types=None,
        num_mamba_replacements=None,
        replacement_block_mode="mamba",
        **kwargs,
    ):
        super().__init__()
        self.default_stage_block_mode = _normalize_block_mode(block_mode)
        self.default_stage_schedule, self.explicit_stage_schedule = _build_encoder_block_schedule(
            depths=depths,
            default_stage_block_mode=self.default_stage_block_mode,
            stage_block_types=stage_block_types,
            num_mamba_replacements=num_mamba_replacements,
            replacement_block_mode=replacement_block_mode,
        )

        self.embed_1 = PatchEmbedding(
            in_channels, embed_dims[0], patch_kernel_size[0], patch_stride[0], patch_padding[0], use_rope=use_rope
        )
        self.embed_2 = PatchEmbedding(
            embed_dims[0], embed_dims[1], patch_kernel_size[1], patch_stride[1], patch_padding[1], use_rope=use_rope
        )
        self.embed_3 = PatchEmbedding(
            embed_dims[1], embed_dims[2], patch_kernel_size[2], patch_stride[2], patch_padding[2], use_rope=use_rope
        )
        self.embed_4 = PatchEmbedding(
            embed_dims[2], embed_dims[3], patch_kernel_size[3], patch_stride[3], patch_padding[3], use_rope=use_rope
        )

        self.stage1 = self._make_stage(
            stage_index=0,
            depth=depths[0],
            embed_dim=embed_dims[0],
            num_heads=num_heads[0],
            sr_ratio=sr_ratios[0],
            d_state=d_state[0],
            d_conv=d_conv[0],
            expand=expand[0],
            mlp_ratio=mlp_ratios[0],
        )
        self.stage2 = self._make_stage(
            stage_index=1,
            depth=depths[1],
            embed_dim=embed_dims[1],
            num_heads=num_heads[1],
            sr_ratio=sr_ratios[1],
            d_state=d_state[1],
            d_conv=d_conv[1],
            expand=expand[1],
            mlp_ratio=mlp_ratios[1],
        )
        self.stage3 = self._make_stage(
            stage_index=2,
            depth=depths[2],
            embed_dim=embed_dims[2],
            num_heads=num_heads[2],
            sr_ratio=sr_ratios[2],
            d_state=d_state[2],
            d_conv=d_conv[2],
            expand=expand[2],
            mlp_ratio=mlp_ratios[2],
        )
        self.stage4 = self._make_stage(
            stage_index=3,
            depth=depths[3],
            embed_dim=embed_dims[3],
            num_heads=num_heads[3],
            sr_ratio=sr_ratios[3],
            d_state=d_state[3],
            d_conv=d_conv[3],
            expand=expand[3],
            mlp_ratio=mlp_ratios[3],
        )

        self.norm4 = nn.LayerNorm(embed_dims[3])
        self.post_gn = nn.ModuleList([nn.GroupNorm(1, ed) for ed in embed_dims])

    def _stage_block_types(self, stage_index: int, depth: int):
        if self.explicit_stage_schedule is not None:
            return self.explicit_stage_schedule[stage_index]

        if stage_index < 3:
            return [self.default_stage_schedule[stage_index]] * depth
        return ["attention"] * depth

    def _make_stage(
        self,
        stage_index,
        depth,
        embed_dim,
        num_heads,
        sr_ratio,
        d_state,
        d_conv,
        expand,
        mlp_ratio,
    ):
        block_types = self._stage_block_types(stage_index, depth)
        return nn.ModuleList([
            _build_encoder_block(
                block_type=block_types[block_idx],
                dim=embed_dim,
                num_heads=num_heads,
                sr_ratio=sr_ratio,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                mlp_ratio=mlp_ratio,
            )
            for block_idx in range(depth)
        ])

    def forward(self, x):
        outs = []

        x, (d, h, w) = self.embed_1(x)
        B, N, C = x.shape
        for blk in self.stage1:
            x = blk(x, d, h, w)
        x = x.view(B, d, h, w, C).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(self.post_gn[0](x))

        x, (d, h, w) = self.embed_2(x)
        B, N, C = x.shape
        for blk in self.stage2:
            x = blk(x, d, h, w)
        x = x.view(B, d, h, w, C).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(self.post_gn[1](x))

        x, (d, h, w) = self.embed_3(x)
        B, N, C = x.shape
        for blk in self.stage3:
            x = blk(x, d, h, w)
        x = x.view(B, d, h, w, C).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(self.post_gn[2](x))

        x, (D, H, W) = self.embed_4(x)
        B, N, C = x.shape
        for blk in self.stage4:
            x = blk(x, (D, H, W))
        x = self.norm4(x)
        x = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(self.post_gn[3](x))
        return outs


class MLP_(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return self.norm(x)


class SegMaFormerDecoderHead(nn.Module):
    def __init__(self, input_feature_dims, decoder_head_embedding_dim, num_classes, dropout=0.0):
        super().__init__()
        self.linears = nn.ModuleList([MLP_(dim, decoder_head_embedding_dim) for dim in input_feature_dims])
        self.fuse = nn.Sequential(
            nn.Conv3d(decoder_head_embedding_dim * 4, decoder_head_embedding_dim, 1, bias=False),
            nn.InstanceNorm3d(decoder_head_embedding_dim, affine=True),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)
        self.pred = nn.Conv3d(decoder_head_embedding_dim, num_classes, 1)

    def forward(self, c1, c2, c3, c4, full_input_size):
        B = c4.shape[0]
        c_feats = []
        for i, c in enumerate([c1, c2, c3, c4]):
            proj = self.linears[i](c)
            D, H, W = c.shape[2:]
            proj = proj.permute(0, 2, 1).contiguous().reshape(B, -1, D, H, W)
            proj = F.interpolate(proj, size=c1.shape[2:], mode="trilinear", align_corners=False)
            c_feats.append(proj)

        x = torch.cat(c_feats, dim=1)
        x = self.fuse(x)
        x = self.dropout(x)
        x = self.pred(x)
        x = F.interpolate(x, size=full_input_size, mode="trilinear", align_corners=False)
        return x


class SegMaFormer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        encoder_params = {
            k: kwargs[k]
            for k in [
                "in_channels",
                "embed_dims",
                "patch_kernel_size",
                "patch_stride",
                "patch_padding",
                "mlp_ratios",
                "num_heads",
                "sr_ratios",
                "depths",
                "d_state",
                "d_conv",
                "expand",
                "block_mode",
                "use_rope",
                "stage_block_types",
                "num_mamba_replacements",
                "replacement_block_mode",
            ]
            if k in kwargs
        }
        self.encoder = MixVisionTransformer(**encoder_params)
        self.decoder = SegMaFormerDecoderHead(
            kwargs["embed_dims"],
            kwargs["decoder_head_embedding_dim"],
            kwargs["num_classes"],
            kwargs.get("decoder_dropout", 0.0),
        )

    def forward(self, x, inference: bool = False):
        full_input_size = x.shape[2:]
        c1, c2, c3, c4 = self.encoder(x)
        return self.decoder(c1, c2, c3, c4, full_input_size)


def build_segmaformer_model_from_yaml(yaml_path, overrides=None):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "model_parameters" not in cfg:
        raise ValueError("YAML file must contain 'model_parameters' key")

    params = dict(cfg["model_parameters"])
    if overrides:
        params.update(overrides)
    required_params = [
        "in_channels",
        "embed_dims",
        "patch_kernel_size",
        "patch_stride",
        "patch_padding",
        "mlp_ratios",
        "num_heads",
        "sr_ratios",
        "depths",
        "d_state",
        "d_conv",
        "expand",
        "decoder_head_embedding_dim",
        "num_classes",
    ]
    missing_params = [p for p in required_params if p not in params]
    if missing_params:
        raise ValueError(f"Missing required parameters: {missing_params}")

    return SegMaFormer(**params)


# Backward-compatible aliases for older trainer imports and experiment code.
SegMaFormerDecoderHead = SegMaFormerDecoderHead
SegMaFormer = SegMaFormer
build_segmaformer_model_from_yaml = build_segmaformer_model_from_yaml


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 1, 48, 192, 192).to(device)
    model = SegMaFormer(
        in_channels=1,
        embed_dims=[32, 64, 160, 256],
        patch_kernel_size=[7, 3, 3, 3],
        patch_stride=[4, 2, 2, 2],
        patch_padding=[3, 1, 1, 1],
        mlp_ratios=[2, 2, 2, 2],
        num_heads=[1, 2, 4, 8],
        sr_ratios=[8, 4, 2, 1],
        depths=[1, 2, 2, 2],
        d_state=[8, 8, 8, 8],
        d_conv=[3, 3, 3, 3],
        expand=[2, 2, 2, 2],
        decoder_head_embedding_dim=256,
        num_classes=14,
        decoder_dropout=0.1,
        deep_supervision=True,
    ).to(device)

    out = model(x)
    print(f"Output shape: {tuple(out.shape)}")
    print("SegMaFormer created successfully!")
