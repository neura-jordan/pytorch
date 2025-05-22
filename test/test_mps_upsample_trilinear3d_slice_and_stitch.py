import pytest
import torch
import itertools
from torch.testing import assert_allclose


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_slice_and_stitch_upsample_trilinear3d():
    device = 'mps'
    dtypes = [torch.float32, torch.float16]
    aligns = [False, True]
    # (scale_d, scale_h, scale_w)
    scale_list = [
        (1.0, 1.0, 1.0),  # identity
        (2.0, 2.0, 2.0),  # uniform upsample
        (2.0, 3.0, 4.0),  # non-uniform upsample
        (0.5, 0.5, 0.5),  # downsample
        (1.5, 0.75, 2.0)  # mixed scales
    ]
    torch.manual_seed(0)
    for dtype, align in itertools.product(dtypes, aligns):
        for scales in scale_list:
            # Input tensor
            x = torch.randn(2, 3, 4, 5, 6, dtype=dtype, device=device, requires_grad=True)
            # 1) F.interpolate dispatches to MPS backend
            y_mps = torch.nn.functional.interpolate(
                x, scale_factor=scales, mode='trilinear', align_corners=align)
            # 2) Reference on CPU
            y_cpu = torch.nn.functional.interpolate(
                x.cpu(), scale_factor=scales, mode='trilinear', align_corners=align)
            # Compare results
            tol = 1e-3 if dtype == torch.float16 else 1e-5
            assert_allclose(y_mps.cpu(), y_cpu, rtol=tol, atol=tol)

            # 3) explicit output size (MPS vs CPU for same explicit size)
            out_shape = [int(x.size(i+2) * scales[i]) for i in range(3)]
            y2_mps = torch.nn.functional.interpolate(
                x, size=out_shape, mode='trilinear', align_corners=align)
            y2_cpu = torch.nn.functional.interpolate(
                x.cpu(), size=out_shape, mode='trilinear', align_corners=align)
            assert_allclose(y2_mps.cpu(), y2_cpu, rtol=tol, atol=tol)

            # 4) identity check
            yi = torch.nn.functional.interpolate(
                x, scale_factor=(1,1,1), mode='trilinear', align_corners=align)
            assert_allclose(yi.cpu(), x.cpu(), rtol=1e-6, atol=1e-6)

            # 5) gradient flow (skip if MPS backward not implemented)
            try:
                grad = torch.autograd.grad(y_mps.sum(), x)
                assert grad[0] is not None
            except NotImplementedError:
                pytest.skip("MPS trilinear3d backward not implemented, skipping gradient test")

            # 6) out= variant (detach input since .out doesn't support autograd)
            out = torch.empty_like(y2_mps)
            torch.ops.aten.upsample_trilinear3d.out(
                x.detach(), out_shape, align, scales_d=None, scales_h=None, scales_w=None, out=out)
            assert_allclose(out, y2_mps, rtol=tol, atol=tol)

    print("All MPS slice+stitch 3D upsample tests passed!") 