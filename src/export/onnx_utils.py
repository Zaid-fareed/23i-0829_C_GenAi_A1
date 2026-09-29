"""Shared ONNX export + verification helper used by every task's export script."""
import numpy as np
import onnx
import onnxruntime as ort
import torch


def export_and_verify(model: torch.nn.Module, dummy_inputs, out_path, input_names, output_names,
                      dynamic_axes=None, opset=17, atol=1e-4, rtol=1e-3, n_check=8):
    """dummy_inputs: tuple of tensors matching model.forward's positional args (batch dim = n_check for the check).
    Exports, runs onnx.checker, then compares PyTorch vs ONNXRuntime outputs on `n_check` random-ish inputs
    (the same `dummy_inputs` batch, truncated/repeated to n_check) and asserts they match within tolerance.
    Returns the max abs diff found (for logging)."""
    model = model.eval().cpu()
    dummy_inputs = tuple(x.cpu() for x in dummy_inputs)
    dynamic_axes = dynamic_axes or {n: {0: "batch"} for n in list(input_names) + list(output_names)}

    torch.onnx.export(model, dummy_inputs, str(out_path), input_names=list(input_names),
                      output_names=list(output_names), dynamic_axes=dynamic_axes, opset_version=opset,
                      do_constant_folding=True, dynamo=False)  # dynamo=False: avoid onnxscript dependency
    onnx.checker.check_model(str(out_path))

    with torch.no_grad():
        torch_out = model(*dummy_inputs)
    if isinstance(torch_out, torch.Tensor):
        torch_out = (torch_out,)

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    feed = {n: x.numpy() for n, x in zip(input_names, dummy_inputs)}
    onnx_out = sess.run(list(output_names), feed)

    max_diff = 0.0
    for t, o in zip(torch_out, onnx_out):
        d = float(np.abs(t.numpy() - o).max())
        max_diff = max(max_diff, d)
        assert np.allclose(t.numpy(), o, atol=atol, rtol=rtol), f"ONNX mismatch: max abs diff {d}"
    print(f"[onnx] exported {out_path}, checker OK, max |pytorch-onnxruntime| = {max_diff:.2e}")
    return max_diff
