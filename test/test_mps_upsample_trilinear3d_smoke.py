import torch
import pytest

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_trilinear3d_mps_explicit_size():
    # Random input tensor on MPS
    input = torch.randn(2, 3, 4, 5, 6, device='mps', dtype=torch.float32)
    # Explicit output size
    out_mps = torch.ops.aten._upsample_trilinear3d_mps(
        input, [8, 10, 12], False, None, None, None)
    # Reference CPU result
    out_ref = torch.nn.functional.interpolate(
        input.cpu(), size=[8, 10, 12], mode='trilinear', align_corners=False)
    out_ref = out_ref.to('mps')
    assert torch.allclose(out_mps, out_ref, atol=1e-5, rtol=1e-5)

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_trilinear3d_mps_scale_factors():
    # Random input tensor on MPS
    input = torch.randn(2, 3, 4, 5, 6, device='mps', dtype=torch.float32)
    # Scale factors
    scales = (2.0, 3.0, 4.0)
    out_mps = torch.ops.aten._upsample_trilinear3d_mps(
        input, None, True, scales[0], scales[1], scales[2])
    # Reference CPU result
    out_ref = torch.nn.functional.interpolate(
        input.cpu(), scale_factor=scales, mode='trilinear', align_corners=True)
    out_ref = out_ref.to('mps')
    assert torch.allclose(out_mps, out_ref, atol=1e-5, rtol=1e-5)

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_trilinear3d_mps_backward():
    # Test backward for explicit size case
    inp = torch.randn(1, 2, 3, 4, 5, device='mps', dtype=torch.float32, requires_grad=True)
    out = torch.ops.aten._upsample_trilinear3d_mps(
        inp, [6, 8, 10], False, None, None, None)
    loss = out.sum()
    loss.backward()
    # Check gradient presence and shape
    assert inp.grad is not None
    assert inp.grad.shape == inp.shape
    assert torch.isfinite(inp.grad).all()

    # Test backward for scale factors case
    inp2 = torch.randn(1, 2, 3, 4, 5, device='mps', dtype=torch.float32, requires_grad=True)
    scales = (1.5, 2.5, 3.5)
    out2 = torch.ops.aten._upsample_trilinear3d_mps(
        inp2, None, True, scales[0], scales[1], scales[2])
    loss2 = out2.sum()
    loss2.backward()
    assert inp2.grad is not None
    assert inp2.grad.shape == inp2.shape
    assert torch.isfinite(inp2.grad).all() 