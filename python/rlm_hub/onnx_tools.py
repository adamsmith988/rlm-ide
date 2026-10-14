"""ONNX export and quantization for downloaded models.

  export   --src DIR --out DIR --kind chat|embed [--trust-remote-code]
  quantize --src DIR --out DIR --method dynamic-int8|dynamic-uint8|int4-weight [--onnx-file REL]

The output folder is a complete model folder (config, tokenizer, ONNX files) that serve.py loads with
--backend onnx. Progress lines start with "STEP " so the IDE can show them.
"""
import argparse
import shutil
import sys
from pathlib import Path

import import_guard

import_guard.install()

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".h5", ".msgpack", ".ot", ".gguf")


def step(msg):
    print(f"STEP {msg}", flush=True)


def copy_support_files(src, out):
    """Config, tokenizer and pooling files travel with the ONNX weights; original weights do not."""
    out_abs = Path(out).resolve()
    for p in Path(src).rglob("*"):
        rel = p.relative_to(src)
        # Outputs live under the model folder (.rlm/), so skip them and any other dot folder.
        if out_abs == p.resolve() or out_abs in p.resolve().parents or any(part.startswith(".") for part in rel.parts[:-1]):
            continue
        if p.is_dir() or rel.parts[0] == "onnx" or p.suffix in WEIGHT_SUFFIXES or p.suffix == ".onnx" or p.name.endswith(".onnx_data"):
            continue
        dest = Path(out) / rel
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)


def export(a):
    from optimum.exporters.onnx import main_export

    task = "feature-extraction" if a.kind == "embed" else "text-generation-with-past"
    step(f"exporting to ONNX (task {task})")
    try:
        main_export(a.src, output=a.out, task=task, trust_remote_code=a.trust_remote_code, do_validation=False)
    except Exception as e:  # noqa: BLE001
        # The library optimum detects (e.g. sentence-transformers) may be a version its exporter does not
        # handle; the plain transformers model exports the same weights and serve.py applies the pooling.
        step(f"export via the detected library failed ({type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}); exporting the transformers model")
        shutil.rmtree(a.out, ignore_errors=True)
        main_export(a.src, output=a.out, task=task, trust_remote_code=a.trust_remote_code, do_validation=False, library_name="transformers")
    copy_support_files(a.src, a.out)
    step("export done")


def onnx_files(src, chosen):
    if chosen:
        return [Path(src) / chosen]
    files = sorted(Path(src).rglob("*.onnx"))
    if not files:
        raise SystemExit(f"no .onnx files in {src}")
    return files


def quantize(a):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    Path(a.out).mkdir(parents=True, exist_ok=True)
    files = onnx_files(a.src, a.onnx_file)
    for i, f in enumerate(files, 1):
        rel = f.relative_to(a.src)
        dest = Path(a.out) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        step(f"[{i}/{len(files)}] {a.method} {rel}")
        external = f.with_name(f.name + "_data").exists() or any(f.parent.glob(f.stem + "*.data"))
        if a.method in ("dynamic-int8", "dynamic-uint8"):
            quantize_dynamic(str(f), str(dest), weight_type=QuantType.QInt8 if a.method == "dynamic-int8" else QuantType.QUInt8, use_external_data_format=external)
        elif a.method == "int4-weight":
            import onnx

            try:
                from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer as Q
            except ImportError:
                from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer as Q
            model = onnx.load(str(f))
            q = Q(model, block_size=32, is_symmetric=True)
            q.process()
            q.model.save_model_to_file(str(dest), use_external_data_format=external)
        else:
            raise SystemExit(f"unknown method {a.method}")
    copy_support_files(a.src, a.out)
    step("quantize done")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--src", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--kind", choices=["chat", "embed"], required=True)
    e.add_argument("--trust-remote-code", action="store_true")
    q = sub.add_parser("quantize")
    q.add_argument("--src", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--method", required=True)
    q.add_argument("--onnx-file", default="")
    a = p.parse_args()
    try:
        export(a) if a.cmd == "export" else quantize(a)
    except SystemExit:
        raise
    except Exception as ex:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"ERROR {type(ex).__name__}: {str(ex).splitlines()[0] if str(ex) else ''}", file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
