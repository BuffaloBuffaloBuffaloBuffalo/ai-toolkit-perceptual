"""Small real-CUDA regression for the shipped Minimax convrot8 training path."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA + Triton')
def test_int8_activation_kernel_matches_reference_and_pads():
    pytest.importorskip('triton')
    from toolkit.util.convrot_quant import _int8_act_quant_op, quantize_int8_rows
    torch.manual_seed(7)
    x = torch.randn(3, 256, device='cuda', dtype=torch.bfloat16)
    x[0].zero_()
    expected, expected_scales = quantize_int8_rows(x)
    actual, scales = _int8_act_quant_op(x, 127)
    torch.cuda.synchronize()
    assert actual.shape == (32, 256)
    assert torch.equal(actual[:3], expected)
    assert torch.allclose(scales[:3], expected_scales)
    assert actual[3:].count_nonzero() == 0
    assert torch.all(scales[3:] == 1)
