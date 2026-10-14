"""Reports what this Python environment can do for model deployment, as one JSON line.

Nothing here knows about particular models: it lists installed packages, the model types the
installed transformers / optimum can load or export, the ONNX quantizers available, and the GPU.
"""
import importlib
import importlib.metadata as md
import json
import sys

import import_guard

import_guard.install()

PACKAGES = [
    "torch", "transformers", "accelerate", "safetensors", "tokenizers", "sentencepiece", "protobuf",
    "bitsandbytes", "optimum", "optimum-onnx", "onnx", "onnxruntime", "onnxruntime-gpu", "gguf", "numpy",
]


def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def main():
    out = {"python": sys.version.split()[0], "executable": sys.executable, "packages": {p: version(p) for p in PACKAGES}}
    try:
        import torch

        cuda = torch.cuda.is_available()
        gpu = None
        if cuda:
            free, total = torch.cuda.mem_get_info()
            gpu = {"name": torch.cuda.get_device_name(0), "freeMb": free // 2**20, "totalMb": total // 2**20}
        out["cuda"] = cuda
        out["gpu"] = gpu
    except Exception as e:  # noqa: BLE001
        out["cuda"] = False
        out["torchError"] = str(e)[:300]
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

        out["transformersModelTypes"] = sorted(CONFIG_MAPPING_NAMES.keys())
    except Exception as e:  # noqa: BLE001
        out["transformersModelTypes"] = []
        out["transformersError"] = str(e)[:300]
    try:
        from optimum.exporters.tasks import TasksManager

        try:
            # optimum >= 2 registers the ONNX model configs only once the exporter is imported.
            importlib.import_module("optimum.exporters.onnx.model_configs")
        except Exception:  # noqa: BLE001
            pass
        types = getattr(TasksManager, "_SUPPORTED_MODEL_TYPE", {})
        out["onnxExportModelTypes"] = sorted(t for t, backends in types.items() if "onnx" in backends)
    except Exception:  # noqa: BLE001
        out["onnxExportModelTypes"] = []
    quantizers = []
    try:
        q = importlib.import_module("onnxruntime.quantization")
        if hasattr(q, "quantize_dynamic"):
            quantizers += ["dynamic-int8", "dynamic-uint8"]
        try:
            mb = importlib.import_module("onnxruntime.quantization.matmul_nbits_quantizer")
            if hasattr(mb, "MatMulNBitsQuantizer"):
                quantizers.append("int4-weight")
        except Exception:  # noqa: BLE001
            try:
                mb = importlib.import_module("onnxruntime.quantization.matmul_4bits_quantizer")
                if hasattr(mb, "MatMul4BitsQuantizer"):
                    quantizers.append("int4-weight")
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    out["onnxQuantizers"] = quantizers
    print(json.dumps(out))


if __name__ == "__main__":
    main()
