#!/usr/bin/env python3
"""
Model conversion script - Download YOLO model and convert to ONNX.

Usage:
    python convert_model.py yolo26s
    python convert_model.py yolo11n
"""

import argparse
from pathlib import Path


def convert_to_onnx(model_name: str, output_dir: Path, imgsz: int = 640):
    """Download YOLO model and export to ONNX format."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: uv add ultralytics")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pt_path = output_dir / f"{model_name}.pt"
    onnx_path = output_dir / f"{model_name}.onnx"
    
    print(f"📥 Downloading {model_name}.pt...")
    model = YOLO(f"{model_name}.pt")
    print(f"✅ Downloaded to {pt_path}")
    
    print(f"🔄 Exporting to ONNX...")
    model.export(format="onnx", simplify=True, dynamic=False, imgsz=imgsz)
    print(f"✅ Exported to {onnx_path}")
    
    # Show file sizes
    pt_size = pt_path.stat().st_size / (1024 * 1024)
    onnx_size = onnx_path.stat().st_size / (1024 * 1024)
    print(f"\nFile sizes:")
    print(f"  {model_name}.pt:   {pt_size:.1f} MB")
    print(f"  {model_name}.onnx: {onnx_size:.1f} MB")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Download and convert YOLO models to ONNX")
    parser.add_argument("model", help="Model name (e.g., yolo26s, yolo11n, yolo8m)")
    parser.add_argument("--output", "-o", default="src/yolo", help="Output directory (default: src/yolo)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    print(f"🚀 Model Conversion Tool")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")
    print(f"Image size: {args.imgsz}\n")
    
    success = convert_to_onnx(args.model, output_dir, args.imgsz)
    
    if success:
        print(f"\n✨ Conversion complete!")
        print(f"\nTo use this model, update src/yolo/main.py:")
        print(f'  MODEL_PATH = os.path.join(os.path.dirname(__file__), "{args.model}.onnx")')
    else:
        print("\n❌ Conversion failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
