# Copyright (c) 2025, ETH Zurich, Rafael Cathomen
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
import scipy.special
import torch
from torch.autograd import Function


# Bessel function with scipy
class BesselIv(Function):
    """Custom torch.autograd.Function to compute I_v(x) using SciPy."""

    @staticmethod
    def forward(ctx, v, x):
        """Compute I_v(x) elementwise."""
        x_cpu = x.detach().cpu().double().numpy()
        y_cpu = scipy.special.iv(v, x_cpu)
        y = torch.from_numpy(y_cpu).to(x.device).to(x.dtype)

        ctx.save_for_backward(x, y)
        ctx.v = v
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """d/dx of I_v(x) = (I_{v-1}(x) + I_{v+1}(x)) / 2"""
        x, ivx = ctx.saved_tensors
        v = ctx.v

        x_cpu = x.detach().cpu().double().numpy()
        ivp1_cpu = scipy.special.iv(v + 1.0, x_cpu)
        ivm1_cpu = scipy.special.iv(v - 1.0, x_cpu)

        ivp1 = torch.from_numpy(ivp1_cpu).to(x.device).to(x.dtype)
        ivm1 = torch.from_numpy(ivm1_cpu).to(x.device).to(x.dtype)

        d_iv_dx = 0.5 * (ivm1 + ivp1)
        grad_x = grad_output * d_iv_dx

        return None, grad_x


def bessel_iv(v, x):
    """Helper function to call our custom BesselIv autograd op."""
    return BesselIv.apply(v, x)


class LogBesselIv2TermFunction(torch.autograd.Function):
    """
    Computes log(I_v(x)) using piecewise approximation for numerical stability.
    Large x: I_v(x) ~ e^x / sqrt(2 pi x) * (1 - alpha/x) where alpha = (4v^2 - 1)/8
    """

    @staticmethod
    def forward(ctx, v, x, threshold=10.0) -> torch.Tensor:
        alpha = (4.0 * (v**2) - 1.0) / 8.0
        mask_small = x < threshold
        x_small = x[mask_small]
        x_large = x[~mask_small]
        log_iv_out = torch.empty_like(x, dtype=x.dtype, device=x.device)

        # Small x: use direct iv from SciPy
        if x_small.numel() > 0:
            x_small_cpu = x_small.detach().cpu().double().numpy()
            iv_small_cpu = scipy.special.iv(v, x_small_cpu)
            iv_small_cpu = np.clip(iv_small_cpu, 1e-38, None)
            log_iv_small_cpu = np.log(iv_small_cpu)
            log_iv_small = torch.from_numpy(log_iv_small_cpu).to(x_small.device).to(x_small.dtype)
            log_iv_out[mask_small] = log_iv_small

        # Large x: use two-term asymptotic approximation
        if x_large.numel() > 0:
            log_iv_approx = x_large - 0.5 * torch.log(2.0 * math.pi * x_large) - alpha / x_large
            log_iv_out[~mask_small] = log_iv_approx

        ctx.save_for_backward(x, log_iv_out)
        ctx.v = v
        ctx.threshold = threshold
        ctx.alpha = alpha
        return log_iv_out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Small x: d/dx[log(I_v(x))] = I_v'(x)/I_v(x) where I_v'(x) = (I_{v-1}(x)+I_{v+1}(x))/2
        Large x: derivative of asymptotic = 1 - 0.5/x + alpha/x^2
        """
        x, log_iv = ctx.saved_tensors
        v = ctx.v
        threshold = ctx.threshold
        alpha = ctx.alpha

        mask_small = x < threshold
        x_small = x[mask_small]
        x_large = x[~mask_small]
        grad_small = grad_output[mask_small]
        grad_large = grad_output[~mask_small]
        grad_x = torch.zeros_like(x)

        # Small x: exact derivative
        if x_small.numel() > 0:
            iv_small = log_iv[mask_small].exp()
            x_small_cpu = x_small.detach().cpu().double().numpy()
            ivm1_cpu = scipy.special.iv(v - 1.0, x_small_cpu)
            ivp1_cpu = scipy.special.iv(v + 1.0, x_small_cpu)

            ivm1_t = torch.from_numpy(ivm1_cpu).to(x_small.device).to(x_small.dtype)
            ivp1_t = torch.from_numpy(ivp1_cpu).to(x_small.device).to(x_small.dtype)

            iv_prime_small = 0.5 * (ivm1_t + ivp1_t)
            dlogiv_dx_small = iv_prime_small / (iv_small + 1e-38)
            grad_x[mask_small] = grad_small * dlogiv_dx_small

        # Large x: derivative of asymptotic approximation
        if x_large.numel() > 0:
            dlogiv_dx_large = 1.0 - 0.5 / x_large + alpha / (x_large**2)
            grad_x[~mask_small] = grad_large * dlogiv_dx_large

        return None, grad_x, None


def log_bessel_iv_2term(v: float, x: torch.Tensor, threshold: float = 10.0) -> torch.Tensor:
    """Stable log(I_v(x)) using piecewise approach: exact for small x, asymptotic for large x."""
    return LogBesselIv2TermFunction.apply(v, x, threshold)


# Full PyTorch implementation
def _log_bessel_i_small_nu(nu, z, max_iter=50):
    """Series expansion of I_nu(z) for smaller z: I_nu(z) = (z/2)^nu * sum[(z/2)^(2k) / (k! * Gamma(k+nu+1))]"""
    half_z = 0.5 * z
    log_half_z = torch.log(half_z)
    log_prefactor = nu * log_half_z

    k_vals = torch.arange(max_iter, device=z.device, dtype=z.dtype)
    log_power = 2.0 * k_vals.unsqueeze(-1) * log_half_z
    log_factorial = torch.lgamma(k_vals + 1.0).unsqueeze(-1)
    log_gamma = torch.lgamma(k_vals.unsqueeze(-1) + (nu + 1.0))
    log_terms = log_power - log_factorial - log_gamma

    max_log_terms, _ = torch.max(log_terms, dim=0, keepdim=True)
    stable_sum = torch.exp(log_terms - max_log_terms).sum(dim=0)
    log_series = max_log_terms.squeeze(0) + torch.log(stable_sum)

    return log_prefactor + log_series


def _log_bessel_i_large_nu(nu, z):
    """Asymptotic approximation: I_nu(z) ~ e^z / sqrt(2 pi z) * [1 - (nu - 1/2)/(2z)]"""
    return z - 0.5 * torch.log(2.0 * math.pi * z) - (nu - 0.5) / (2.0 * z)


def log_bessel_i_nu(nu, z, cutoff=15.0, max_series=50):
    """Compute log(I_nu(z)) piecewise: series expansion for z < cutoff, asymptotic for z >= cutoff."""
    z_clamped = torch.clamp(z, min=1e-30)
    log_i_small = _log_bessel_i_small_nu(nu, z_clamped, max_iter=max_series)
    log_i_large = _log_bessel_i_large_nu(nu, z_clamped)
    return torch.where(z_clamped < cutoff, log_i_small, log_i_large)


def vmf_log_prob(x, mu, kappa, cutoff=15.0, max_series=50):
    """von Mises-Fisher log probability, fully differentiable."""
    d = x.shape[-1]
    nu = 0.5 * d - 1.0
    dot = (x * mu).sum(dim=-1)
    logC = nu * torch.log(kappa) - 0.5 * d * math.log(2.0 * math.pi) - log_bessel_i_nu(nu, kappa, cutoff, max_series)
    return logC + kappa * dot


def vmf_log_prob_scipy(x, mu, kappa):
    """von Mises-Fisher log probability via SciPy (CPU & numpy). No gradients."""
    x_np = x.detach().cpu().numpy()
    mu_np = mu.detach().cpu().numpy()
    kappa_np = kappa.detach().cpu().numpy()

    d = x.shape[-1]
    nu = 0.5 * d - 1.0
    dot_np = np.sum(x_np * mu_np, axis=-1)
    log_kappa_np = np.log(kappa_np)

    iv_vals = scipy.special.iv(nu, kappa_np)
    iv_vals = np.clip(iv_vals, a_min=1e-300, a_max=None)
    log_iv_np = np.log(iv_vals)

    logC_np = nu * log_kappa_np - 0.5 * d * math.log(2.0 * math.pi) - log_iv_np
    logp_np = logC_np + kappa_np * dot_np

    return torch.from_numpy(logp_np)
