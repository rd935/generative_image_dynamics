#!/usr/bin/env python

import torch
from typing import Optional
from .softsplat import softsplat_func  # reuse their CUDA bilinear splat


# You can tune this if you want later
GAUSSIAN_SIGMA = 1.0


def gaussian_splat(
    tenIn: torch.Tensor,
    tenFlow: torch.Tensor,
    tenMetric: Optional[torch.Tensor],
    strMode: str
) -> torch.Tensor:
    """
    Gaussian splatting wrapper that mimics the API of softsplat.softsplat.

    We keep:
      - same signature: (tenIn, tenFlow, tenMetric, strMode)
      - same use of softsplat_func (CUDA kernel for bilinear splatting)
    But we change the weighting:
      - For 'soft' / 'gauss' / 'gaussian' modes, we use Gaussian weights:
            weight = exp( - metric^2 / (2 * sigma^2) )

    Args:
        tenIn:    (B, C, H, W)
        tenFlow:  (B, 2, H, W)
        tenMetric:
            Per-pixel metric, same semantics as original softsplat.
            Required for 'soft' / 'gauss' / 'gaussian' / 'linear'.
        strMode:
            Base modes supported:
              'sum'       -> sum splatting (no metric)
              'avg'       -> average splatting (no metric)
              'linear'    -> linear weighting (same as original)
              'soft'      -> **Gaussian** weighting (we override original softmax)
              'gauss'     -> same as 'soft'
              'gaussian'  -> same as 'soft'
            Optional suffix:
              '-addeps', '-zeroeps', '-clipeps' for epsilon handling.
    """
    assert tenIn.dim() == 4 and tenFlow.dim() == 4
    base = strMode.split('-')[0]

    # Treat 'soft' as Gaussian too, since we are "replacing softmax splatting"
    # for the experiment.
    valid_bases = ['sum', 'avg', 'linear', 'soft', 'gauss', 'gaussian']
    assert base in valid_bases, f'Unknown mode: {strMode}'

    if base in ['sum', 'avg']:
        assert tenMetric is None, 'tenMetric must be None for sum/avg modes'
    else:
        assert tenMetric is not None, 'tenMetric is required for linear/soft/gauss modes'

    # -------------------------------------------------------------------------
    # Build augmented input for splatting (exact same style as original)
    # -------------------------------------------------------------------------
    if strMode == 'avg':
        # Append a channel of ones for averaging denominator
        tenIn = torch.cat(
            [tenIn, tenIn.new_ones([tenIn.shape[0], 1, tenIn.shape[2], tenIn.shape[3]])],
            1
        )

    elif base == 'linear':
        # Same as original: linear weighting by metric
        tenIn = torch.cat([tenIn * tenMetric, tenMetric], 1)

    elif base in ['soft', 'gauss', 'gaussian']:
        # ---------------------------------------------------------------------
        # GAUSSIAN WEIGHTING:
        #   Instead of exp(metric) we use exp( - metric^2 / (2 * sigma^2) )
        #   If tenMetric is (B,1,H,W) it broadcasts over channels.
        # ---------------------------------------------------------------------
        sigma = GAUSSIAN_SIGMA
        weight = torch.exp(-0.5 * (tenMetric / sigma) ** 2)
        tenIn = torch.cat([tenIn * weight, weight], 1)

    # 'sum' mode: we just pass tenIn through with no extra channel.

    # -------------------------------------------------------------------------
    # Forward splat using the *same* CUDA kernel as original softsplat
    # -------------------------------------------------------------------------
    tenOut = softsplat_func.apply(tenIn, tenFlow)

    # -------------------------------------------------------------------------
    # Normalization for modes that appended a denominator channel
    # (avg / linear / soft / gauss / gaussian)
    # -------------------------------------------------------------------------
    if base in ['avg', 'linear', 'soft', 'gauss', 'gaussian']:
        tenNormalize = tenOut[:, -1:, :, :]

        parts = strMode.split('-')
        suffix = parts[1] if len(parts) > 1 else 'addeps'

        if suffix == 'addeps':
            tenNormalize = tenNormalize + 0.0000001
        elif suffix == 'zeroeps':
            tenNormalize = tenNormalize.clone()
            tenNormalize[tenNormalize == 0.0] = 1.0
        elif suffix == 'clipeps':
            tenNormalize = tenNormalize.clamp(0.0000001, None)
        else:
            raise ValueError(f'Unknown epsilon mode in strMode: {strMode}')

        tenOut = tenOut[:, :-1, :, :] / tenNormalize

    return tenOut
