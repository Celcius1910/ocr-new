#!/usr/bin/env python3
"""
CUDA Environment Diagnostic Script
Checks NVIDIA driver, CUDA availability, PyTorch, EasyOCR, and Ultralytics YOLO
"""

import sys
import subprocess


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def check_nvidia_driver():
    """Check NVIDIA driver using nvidia-smi"""
    print_section("1. NVIDIA Driver Check")
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.split("\n")
            for line in lines[:10]:  # Print first 10 lines (header + GPU info)
                print(line)
            print("\n✓ NVIDIA Driver: INSTALLED")

            # Extract CUDA version
            for line in lines:
                if "CUDA Version" in line:
                    cuda_ver = line.split("CUDA Version:")[-1].strip().split()[0]
                    print(f"✓ CUDA Version (Driver): {cuda_ver}")
                    break
            return True
        else:
            print("✗ nvidia-smi command failed")
            return False
    except FileNotFoundError:
        print("✗ NVIDIA Driver: NOT FOUND")
        print(
            "  → Install NVIDIA driver dari: https://www.nvidia.com/Download/index.aspx"
        )
        return False
    except Exception as e:
        print(f"✗ Error checking driver: {e}")
        return False


def check_pytorch():
    """Check PyTorch CUDA support"""
    print_section("2. PyTorch CUDA Check")
    try:
        import torch

        print(f"✓ PyTorch version: {torch.__version__}")

        cuda_available = torch.cuda.is_available()
        print(f"  torch.cuda.is_available(): {cuda_available}")

        if cuda_available:
            print(f"  torch.version.cuda: {torch.version.cuda}")
            print(f"  CUDA device count: {torch.cuda.device_count()}")
            print(f"  Current CUDA device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
            print(f"  Device capability: {torch.cuda.get_device_capability(0)}")
            print("\n✓ PyTorch: CUDA ENABLED")
            return True
        else:
            print("\n✗ PyTorch: CPU ONLY")
            print("  → Install PyTorch GPU build:")
            print(
                "     pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            )
            return False
    except ImportError:
        print("✗ PyTorch: NOT INSTALLED")
        print(
            "  → Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        )
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def check_easyocr():
    """Check EasyOCR presence and note GPU follows PyTorch"""
    print_section("3. EasyOCR Check")
    try:
        import easyocr  # noqa: F401
        import torch

        print("✓ EasyOCR: INSTALLED")
        print(
            "  Note: EasyOCR uses PyTorch; GPU availability follows torch.cuda.is_available()."
        )
        print(f"  PyTorch CUDA available: {torch.cuda.is_available()}")
        return True
    except ImportError:
        print("✗ EasyOCR: NOT INSTALLED")
        print("  → Install: pip install easyocr")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def check_ultralytics():
    """Check Ultralytics YOLO"""
    print_section("4. Ultralytics YOLO Check")
    try:
        import ultralytics

        print(f"✓ Ultralytics version: {ultralytics.__version__}")
        print("  (YOLO akan otomatis pakai PyTorch device)")
        return True
    except ImportError:
        print("✗ Ultralytics: NOT INSTALLED")
        print("  → Install: pip install ultralytics")
        return False


def main():
    print("\n" + "=" * 60)
    print("  CUDA DIAGNOSTIC TOOL - RTX 4070 Super")
    print("=" * 60)

    # Run all checks
    driver_ok = check_nvidia_driver()
    torch_ok = check_pytorch()
    easyocr_ok = check_easyocr()
    yolo_ok = check_ultralytics()

    # Summary
    print_section("SUMMARY")
    print(f"NVIDIA Driver:  {'✓ OK' if driver_ok else '✗ MISSING'}")
    print(f"PyTorch CUDA:   {'✓ OK' if torch_ok else '✗ NEEDS INSTALL'}")
    print(f"EasyOCR:        {'✓ OK' if easyocr_ok else '✗ NEEDS INSTALL'}")
    print(f"Ultralytics:    {'✓ OK' if yolo_ok else '✗ NEEDS INSTALL'}")

    if driver_ok and torch_ok:
        print("\n🎉 CUDA READY! Semua komponen siap untuk GPU acceleration.")
        print("\nJalankan dengan (PowerShell):")
        print(
            '  python run_ocr.py --mode folder --input "datasets\\sample_ocr_ktp_axa" --output "outputs\\results.gpu.json" --yolo-device cuda --donut-device cuda --easyocr-use-gpu'
        )
    else:
        print("\n⚠️  Perlu instalasi komponen GPU. Ikuti instruksi di atas.")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
