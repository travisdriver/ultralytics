
import math
from typing import Tuple, Optional

import torch
import matplotlib.pyplot as plt

from ultralytics.utils.metrics import probiou


def random_ellipses(
    N: int,
    *,
    center_range: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),  # (xmin, xmax, ymin, ymax)
    semi_major_range: Tuple[float, float] = (0.25, 1.0),                        # a in [amin, amax]
    aspect_range: Tuple[float, float] = (0.3, 1.0),                             # r=b/a in [rmin, rmax] (≤ 1)
    orientation_range: Tuple[float, float] = (0.0, math.pi),                    # θ in [θmin, θmax)
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample N random ellipses parameterized as (cx, cy, a, b, theta).

    Args:
        N: Number of ellipses.
        center_range: (xmin, xmax, ymin, ymax) for the center coordinates.
        semi_major_range: (amin, amax) range for semi-major axis a.
        aspect_range: (rmin, rmax) for aspect ratio r = b/a (must be ≤ 1).
        orientation_range: (thetamin, thetamax) for orientation in radians.
        device, dtype: Optional torch device/dtype for the output.
        generator: Optional torch.Generator for reproducibility.

    Returns:
        Tensor of shape (N, 5): columns are (cx, cy, a, b, theta).
    """
    xmin, xmax, ymin, ymax = center_range
    amin, amax = semi_major_range
    rmin, rmax = aspect_range
    tmin, tmax = orientation_range

    if not (0.0 < amin <= amax):
        raise ValueError("semi_major_range must satisfy 0 < amin ≤ amax.")
    if not (0.0 < rmin <= rmax <= 1.0):
        raise ValueError("aspect_range must satisfy 0 < rmin ≤ rmax ≤ 1.")
    if not (tmin < tmax):
        raise ValueError("orientation_range must satisfy thetamin < thetamax.")

    # Convenience uniform sampler
    def urand(low, high, shape):
        return (low + (high - low) * torch.rand(shape, device=device, dtype=dtype, generator=generator))

    # Centers
    cx = urand(xmin, xmax, (N,))
    cy = urand(ymin, ymax, (N,))

    # Semi-major and aspect ratio
    a  = urand(amin, amax, (N,))
    r  = urand(rmin, rmax, (N,))  # r = b/a, guaranteed ≤ 1
    b  = a * r

    # Orientation
    theta = urand(tmin, tmax, (N,))

    ellipses = torch.stack([cx, cy, a, b, theta], dim=-1)
    return ellipses


def gaussian_angle_metric_torch(
    ep0: torch.Tensor,
    ep1: torch.Tensor,
    *,
    degrees: bool = False,
    eps: float = 1e-12,
    take_arccos: bool = True,
) -> torch.Tensor:
    """
    Batched Gaussian angle metric between pairs of ellipses.

    Args:
        ep0: (N, 5) tensor of ellipse params (cx, cy, a, b, theta).
        ep1: (N, 5) tensor of ellipse params (cx, cy, a, b, theta).
        degrees: If True, theta is in degrees (matches the numpy version).
        eps: Small epsilon for numerical stability.

    Returns:
        (N,) tensor of angles (in radians).
    """
    if ep0.ndim != 2 or ep0.size(-1) != 5:
        raise ValueError("ep0 must be shape (N, 5)")
    if ep1.ndim != 2 or ep1.size(-1) != 5:
        raise ValueError("ep1 must be shape (N, 5)")

    # Unpack
    cx0, cy0, a0, b0, th0 = ep0.unbind(dim=-1)
    cx1, cy1, a1, b1, th1 = ep1.unbind(dim=-1)

    # Centers (N,2,1)
    c0 = torch.stack([cx0, cy0], dim=-1).unsqueeze(-1)
    c1 = torch.stack([cx1, cy1], dim=-1).unsqueeze(-1)
    d  = c0 - c1  # (N,2,1)

    # Angles -> radians if needed
    if degrees:
        th0 = torch.deg2rad(th0)
        th1 = torch.deg2rad(th1)

    # Rotation matrices (N,2,2)
    c0s, s0s = torch.cos(th0), torch.sin(th0)
    c1s, s1s = torch.cos(th1), torch.sin(th1)

    R0 = torch.stack([
        torch.stack([ c0s, -s0s], dim=-1),
        torch.stack([ s0s,  c0s], dim=-1)
    ], dim=-2)  # (N,2,2)

    R1 = torch.stack([
        torch.stack([ c1s, -s1s], dim=-1),
        torch.stack([ s1s,  c1s], dim=-1)
    ], dim=-2)  # (N,2,2)

    # Diagonal precision matrices (inverse covariance in ellipse metric space)
    # D = diag([4/a^2, 4/b^2]), shape (N,2,2)
    inv_a0_2 = 4 * (a0.clamp_min(eps)).reciprocal()**2
    inv_b0_2 = 4 * (b0.clamp_min(eps)).reciprocal()**2
    inv_a1_2 = 4 * (a1.clamp_min(eps)).reciprocal()**2
    inv_b1_2 = 4 * (b1.clamp_min(eps)).reciprocal()**2

    D0 = torch.zeros(ep0.size(0), 2, 2, dtype=ep0.dtype, device=ep0.device)
    D1 = torch.zeros_like(D0)
    D0[:, 0, 0] = inv_a0_2
    D0[:, 1, 1] = inv_b0_2
    D1[:, 0, 0] = inv_a1_2
    D1[:, 1, 1] = inv_b1_2

    # Y = R D R^T  (N,2,2)
    Y0 = R0 @ D0 @ R0.transpose(-1, -2)
    Y1 = R1 @ D1 @ R1.transpose(-1, -2)
    S  = Y0 + Y1

    # t1 = 4 * sqrt(det(Y0)*det(Y1)) / det(Y0 + Y1)
    detY0 = torch.linalg.det(Y0).clamp_min(eps)
    detY1 = torch.linalg.det(Y1).clamp_min(eps)
    detS  = torch.linalg.det(S).clamp_min(eps)
    t1 = 4.0 * torch.sqrt(detY0 * detY1) / detS

    # t2 = exp(-0.5 * d^T Y0 S^{-1} Y1 d)
    # Compute v = S^{-1} (Y1 d) using a solve (more stable than explicit inverse)
    Y1d = Y1 @ d                        # (N,2,1)
    v   = torch.linalg.solve(S, Y1d)    # (N,2,1)
    q   = (d.transpose(-1, -2) @ (Y0 @ v)).squeeze(-1).squeeze(-1)  # (N,)
    t2  = torch.exp(-0.5 * q)

    # acos argument, clamp for numerical safety
    arg = (t1 * t2).clamp(-1.0, 1.0)
    if take_arccos:
        return torch.arccos(arg)
    else:
        return arg


obb1 = random_ellipses(5000, device='cpu')
obb2 = random_ellipses(5000, device='cpu')

obb1_scaled = obb1.clone()
obb1_scaled[:, :4] *= 1.2  # Scale semi-major and semi-minor axes by 1.2
obb2_scaled = obb2.clone()
obb2_scaled[:, :4] *= 1.2  # Scale semi-major and semi-minor axes by 1.2

piou_hd = probiou(obb1, obb2)
#print("Probiou between random ellipses:\n", piou_hd.flatten())

piou_ga = gaussian_angle_metric_torch(obb1, obb2, take_arccos=True)
piou_ga = 1 - 2 * piou_ga / torch.pi
#print(piou_ga)

#print(torch.max(piou))
# print("Probiou between random ellipses:\n", piou)
#print("Probiou between random ellipses:\n", 1 - 2 * piou / torch.pi)
# 
# 
# piou = probiou(obb1_scaled, obb2_scaled)
# print("Probiou between random ellipses:\n", piou)

plt.plot(piou_hd, piou_ga, "r.")
plt.plot([0, 1], [0, 1], "b-")
plt.xlabel("Hellinger Distance")
plt.ylabel("Gaussian Angle Metric")
plt.savefig("test.png")