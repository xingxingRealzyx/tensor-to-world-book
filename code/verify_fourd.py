"""Numerically verify the book's four-dimensional diagrams and trained example."""
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tensor_figures_4d_attention import case_data


def verify():
    torch.set_num_threads(1)
    checks = []
    x = torch.arange(48, dtype=torch.float64).reshape(2, 3, 2, 4)
    assert x.ndim == 4 and x.numel() == 48 and x[1, 2, 1, 3] == 47
    assert x[1].shape == (3, 2, 4) and x[1:2].shape == (1, 3, 2, 4)
    assert x.stride() == (24, 8, 4, 1)
    checks.append("all 48 addresses, rank-preserving slicing, contiguous strides")
    nhwc = x.permute(0, 2, 3, 1)
    assert nhwc.shape == (2, 2, 4, 3) and nhwc.stride() == (24, 4, 1, 8)
    assert nhwc[1, 0, 1, 2] == 41 and x.reshape(2, 2, 4, 3)[1, 0, 1, 2] == 29
    for b in range(2):
        for c in range(3):
            for y in range(2):
                for w in range(4):
                    assert nhwc[b, y, w, c] == x[b, c, y, w]
    torch.testing.assert_close(nhwc.permute(0, 3, 1, 2), x)
    checks.append("NCHW to NHWC mapping and non-corner reshape counterexample")
    bias = x.new_tensor([10, 100, 1000])
    assert (x + bias[None, :, None, None])[1, 2, 1, 3] == 1047
    try:
        x + bias
    except RuntimeError:
        pass
    else:
        raise AssertionError("Unshaped channel bias should fail when W=4 and C=3")
    checks.append("channel broadcasting versus erroneous last-axis addition")
    assert x.mean(1, keepdim=True).shape == (2, 1, 2, 4)
    assert x.mean(1, keepdim=True)[1, 0, 1, 3] == 39
    assert x.mean((2, 3), keepdim=True).shape == (2, 3, 1, 1)
    assert x.mean((2, 3), keepdim=True)[1, 2, 0, 0] == 43.5
    checks.append("channel versus spatial reduction and retained singleton axes")

    weight = x.new_tensor([[1, -2, .5], [-.5, 1, 2]])
    offset = x.new_tensor([.25, -.75])
    conv = F.conv2d(x, weight[:, :, None, None], offset)
    matrix_form = (nhwc @ weight.T + offset).permute(0, 3, 1, 2)
    scalar_form = torch.empty_like(conv)
    for b in range(2):
        for o in range(2):
            for y in range(2):
                for w in range(4):
                    scalar_form[b, o, y, w] = sum(x[b, c, y, w]*weight[o, c] for c in range(3)) + offset[o]
    torch.testing.assert_close(conv, scalar_form, rtol=0, atol=0)
    torch.testing.assert_close(conv, matrix_form, rtol=0, atol=0)
    torch.testing.assert_close(conv[1, :, 1, 3], x.new_tensor([-23.25, 116.75]))
    checks.append("1x1 convolution, channel-last matmul and scalar sums agree")
    # A legal [4,2] matrix multiplies spatial width in NCHW, not channels.
    spatial_weight = x.new_tensor([[1, 0], [0, 1], [1, 0], [0, 1]])
    assert (x @ spatial_weight).shape == (2, 3, 2, 2)
    torch.testing.assert_close((x @ spatial_weight)[1, 2, 1], x.new_tensor([90, 92]))
    checks.append("four-dimensional matmul contracts the last spatial axis")
    kernel = weight[:, :, None, None].expand(2, 3, 2, 2) / 4
    patch_conv = F.conv2d(x, kernel, offset, stride=2)
    expected = x.new_tensor([[[[-9, -10]], [[45.5, 50.5]]],
                             [[[-21, -22]], [[105.5, 110.5]]]])
    torch.testing.assert_close(patch_conv, expected, rtol=0, atol=0)
    patch_tokens = patch_conv.flatten(2).transpose(1, 2)
    assert patch_tokens.shape == (2, 2, 2)
    torch.testing.assert_close(patch_tokens[1, 1], x.new_tensor([-22, 110.5]))
    checks.append("patch convolution and flatten/transpose preserve batch identity")

    base = torch.arange(24).reshape(2, 3, 4)
    split = base.reshape(2, 3, 2, 2)
    heads = split.transpose(1, 2)
    assert heads.shape == (2, 2, 3, 2) and heads[1, 1, 0, 0] == 14
    torch.testing.assert_close(heads.transpose(1, 2).reshape(2, 3, 4), base)
    checks.append("two samples split into two heads and merge without losing elements")
    case = case_data()
    q, k, v = [torch.tensor(case[name], dtype=torch.float64) for name in ("Q", "K", "V")]
    raw = q @ k.transpose(-2, -1)
    torch.testing.assert_close(raw, torch.tensor(case["raw_scores"], dtype=torch.float64))
    checks.append("four independent batch/head dot-product matrices match scalar loops")
    causal = torch.ones(3, 3, dtype=torch.bool).tril()[None, None]
    padding = torch.tensor(case["key_valid"])[:, None, None, :]
    allowed = causal & padding
    a = (raw / 2**.5).masked_fill(~allowed, float("-inf")).softmax(-1)
    torch.testing.assert_close(a, torch.tensor(case["A"], dtype=torch.float64), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(a.sum(-1), torch.ones(2, 2, 3, dtype=torch.float64))
    assert torch.all(a[1, :, :, 2] == 0)
    checks.append("causal and per-sample padding masks broadcast along correct axes")
    o = a @ v
    torch.testing.assert_close(o, torch.tensor(case["O"], dtype=torch.float64), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(o[1, 0, 2, 0], torch.tensor(2.339523098653314, dtype=torch.float64))
    assert torch.any(o[1, :, 2] != 0)
    checks.append("masked padding queries may still produce nonzero outputs")
    merged = o.transpose(1, 2).reshape(2, 3, 4)
    torch.testing.assert_close(merged, torch.tensor(case["merged"], dtype=torch.float64), rtol=1e-12, atol=1e-12)
    altered_v = v.clone(); altered_v[1, :, 2] += 1000
    torch.testing.assert_close(a @ altered_v, o, rtol=0, atol=0)
    changed_head = v.clone(); changed_head[1, 1, 0] += 10
    torch.testing.assert_close((a @ changed_head)[0], o[0], rtol=0, atol=0)
    torch.testing.assert_close((a @ changed_head)[1, 0], o[1, 0], rtol=0, atol=0)
    checks.append("merge order, ignored padding keys and isolated batch/head computation")

    prediction = json.loads((ROOT / "runs/fourd-channels/prediction.json").read_text())
    learned = torch.tensor(prediction["values"], dtype=torch.float64)
    max_error = (learned - scalar_form).abs().max().item()
    assert prediction["output_shape"] == [2, 2, 2, 4] and max_error < 1e-4
    checks.append("stored deployment output matches the paper map")
    return {"status": "PASS", "checks": checks, "x_shape": list(x.shape),
            "nhwc_shape": list(nhwc.shape), "conv_output": conv.tolist(),
            "patch_output": patch_conv.tolist(), "attention_raw_scores": raw.tolist(),
            "attention_weights": a.tolist(), "attention_outputs": o.tolist(),
            "attention_merged": merged.tolist(), "trained_book_input_max_error": max_error}


if __name__ == "__main__":
    result = verify()
    (ROOT / "code/fourd-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k in ("status", "checks", "trained_book_input_max_error")}, indent=2))
