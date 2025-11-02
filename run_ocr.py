import cv2
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
from paddleocr import PaddleOCR
import difflib
import json
import time
from datetime import datetime
import re
import argparse
import os
import numpy as np
from process_ktp_json import parse_ktp_text, create_response, compute_field_confidence

# Config
MODEL_DIR = "models/donut-ktp-v3"
YOLO_MODEL = "models/best.pt"
CAMERA_INDEX = 0
DROIDCAM_URL = "http://192.168.1.3:4747/video"


def normalize_ocr_text(text: str) -> str:
    t = text.upper()
    # Common typos
    t = re.sub(r"\bPHUVINSI\b", "PROVINSI", t)
    t = re.sub(r"\bPRVINSI\b", "PROVINSI", t)
    t = re.sub(r"\bPROVNSI\b", "PROVINSI", t)
    t = re.sub(r"\bPROUINSI\b", "PROVINSI", t)
    t = re.sub(r"PHUVINSI", "PROVINSI", t)  # merged
    t = re.sub(r"PRVINSI", "PROVINSI", t)  # merged
    t = re.sub(r"SUNATE", "SUMATE", t)
    # Split merged keywords
    t = re.sub(r"(KOTA)([A-Z])", r"\1 \2", t)
    t = re.sub(r"(KABUPATEN)([A-Z])", r"\1 \2", t)
    t = re.sub(r"(PROVINSI)([A-Z])", r"\1 \2", t)
    t = re.sub(r"(PROPINSI)([A-Z])", r"\1 \2", t)
    t = re.sub(r"(SUMATERA)([A-Z])", r"\1 \2", t)
    return t


def canonical_province(name_candidate: str):
    provinces = [
        "DKI JAKARTA",
        "DAERAH ISTIMEWA YOGYAKARTA",
        "JAWA BARAT",
        "JAWA TENGAH",
        "JAWA TIMUR",
        "BANTEN",
        "BALI",
        "NUSA TENGGARA BARAT",
        "NUSA TENGGARA TIMUR",
        "ACEH",
        "SUMATERA UTARA",
        "SUMATERA BARAT",
        "RIAU",
        "KEPULAUAN RIAU",
        "JAMBI",
        "BENGKULU",
        "SUMATERA SELATAN",
        "LAMPUNG",
        "KEPULAUAN BANGKA BELITUNG",
        "KALIMANTAN BARAT",
        "KALIMANTAN TENGAH",
        "KALIMANTAN SELATAN",
        "KALIMANTAN TIMUR",
        "KALIMANTAN UTARA",
        "SULAWESI UTARA",
        "SULAWESI TENGAH",
        "SULAWESI SELATAN",
        "SULAWESI TENGGARA",
        "GORONTALO",
        "MALUKU",
        "MALUKU UTARA",
        "PAPUA",
        "PAPUA BARAT",
    ]
    cand = (name_candidate or "").upper().strip()
    cand_ns = cand.replace(" ", "")
    best, best_s = None, 0.0
    for p in provinces:
        s = difflib.SequenceMatcher(None, cand_ns, p.replace(" ", "")).ratio()
        if s > best_s:
            best_s, best = s, p
    return best if best_s >= 0.75 else None


def extract_header_from_crop(crop: np.ndarray, paddle_ocr: PaddleOCR):
    """Run OCR on top region and parse provinsi/kota.
    Returns: header_full_text, provinsi_text, kota_text
    """
    header_full_text = None
    provinsi_text = None
    kota_text = None

    if crop is None or crop.size == 0:
        return header_full_text, provinsi_text, kota_text

    h_total = crop.shape[0]
    header_h = max(120, int(h_total * 0.30))
    header_crop = crop[0:header_h, :]
    header_lines = []
    try:
        res = paddle_ocr.predict(header_crop)
        if res and len(res) > 0 and isinstance(res[0], dict):
            for text, score in zip(
                res[0].get("rec_texts", []), res[0].get("rec_scores", [])
            ):
                if float(score) > 0.5:
                    header_lines.append(normalize_ocr_text(text))
    except Exception:
        pass

    if not header_lines:
        return header_full_text, provinsi_text, kota_text

    header_full_text = " ".join(header_lines)
    up = re.sub(r"[^A-Z\s/]", "", header_full_text.upper())
    up = re.sub(r"\s+", " ", up).strip()

    m_p = re.search(
        r"(PROVINSI|PROPINSI)\s+([A-Z]+(?:\s+[A-Z]+){0,2})(?=\s+(KOTA|KABUPATEN)\b|$)",
        up,
    )
    if m_p:
        name = m_p.group(2).strip()
        canon = canonical_province(name)
        provinsi_text = f"PROVINSI {canon}" if canon else f"PROVINSI {name}"

    m_k = re.search(
        r"(KOTA|KABUPATEN)\s+([A-Z]+(?:\s+[A-Z]+){0,2})(?=\s+(PROVINSI|PROPINSI|NIK|NAMA|TEMPAT|TTL|LAHIR|JENIS|KELAMIN|ALAMAT)\b|$)",
        up,
    )
    if m_k:
        kota_text = m_k.group(0).strip()

    return header_full_text, provinsi_text, kota_text


def process_image(
    image_input, processor, model, yolo, device, paddle_ocr=None, enable_fallback=True
):
    """Process a single image (path or frame array) and return OCR results"""
    t0 = time.time()

    # Read frame
    if isinstance(image_input, str):
        frame = cv2.imread(image_input)
        if frame is None:
            return create_response(False, error=f"Failed to read image: {image_input}")
    else:
        frame = image_input

    try:
        # YOLO detection
        results = yolo(frame, verbose=False)
        largest = None
        largest_area = 0
        largest_conf = None
        for res in results:
            if not hasattr(res, "boxes") or res.boxes is None:
                continue
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            confs = (
                res.boxes.conf.cpu().numpy()
                if hasattr(res.boxes, "conf") and res.boxes.conf is not None
                else []
            )
            for box, conf in zip(boxes_xyxy, confs):
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest = (x1, y1, x2, y2)
                    largest_conf = float(conf)

        if largest is None:
            return create_response(False, error="No card detected in image")

        x1, y1, x2, y2 = largest
        # Increased margin especially bottom to capture lower fields (agama, pekerjaan)
        margin_horizontal = 10
        margin_top = 10
        margin_bottom = 30  # Larger bottom margin to include agama field
        h, w = frame.shape[:2]
        x1 = max(0, x1 - margin_horizontal)
        y1 = max(0, y1 - margin_top)
        x2 = min(w, x2 + margin_horizontal)
        y2 = min(h, y2 + margin_bottom)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return create_response(False, error="Invalid crop region")

        # Donut OCR
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pixel_values = processor(pil_img, return_tensors="pt").pixel_values.to(device)
        outputs = model.generate(
            pixel_values,
            max_length=512,
            num_beams=4,
            decoder_start_token_id=processor.tokenizer.bos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        decoded_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Parse base fields
        ktp_fields = parse_ktp_text(decoded_text)

        # Header OCR via PaddleOCR (single region parse)
        header_full_text = None
        provinsi_text = None
        kota_text = None
        full_paddle_text = None
        if paddle_ocr is not None:
            header_full_text, provinsi_text, kota_text = extract_header_from_crop(
                crop, paddle_ocr
            )

            # PaddleOCR fallback for agama/pekerjaan (only if enabled)
            if (
                enable_fallback
                and paddle_ocr is not None
                and (not ktp_fields.get("agama") or not ktp_fields.get("pekerjaan"))
            ):
                try:
                    # OPTIMIZED: Single-pass enhanced OCR (most effective, skip redundant passes)
                    # Use full image with enhancement for best accuracy
                    fh, fw = frame.shape[:2]
                    scale = (
                        1.3 if max(fh, fw) < 1200 else 1.15
                    )  # Reduced scale for speed
                    up = cv2.resize(
                        frame,
                        (int(fw * scale), int(fh * scale)),
                        interpolation=cv2.INTER_CUBIC,
                    )

                    # Light contrast enhancement (faster than CLAHE)
                    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
                    enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
                    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

                    full_res = paddle_ocr.ocr(enhanced)
                    if full_res and full_res[0]:
                        lines = full_res[0]
                        full_text = " ".join([ln[1][0] for ln in lines])
                        extra = parse_ktp_text(full_text)
                        if not ktp_fields.get("agama") and extra.get("agama"):
                            ktp_fields["agama"] = extra["agama"]
                        if not ktp_fields.get("pekerjaan") and extra.get("pekerjaan"):
                            ktp_fields["pekerjaan"] = extra["pekerjaan"]
                        # Fuzzy label-following (AGAMA/PEKERJAAN labels may be noisy)
                        import difflib

                        if not ktp_fields.get("agama"):
                            for i, ln in enumerate(lines):
                                t = str(ln[1][0]).upper()
                                ratio = difflib.SequenceMatcher(
                                    None, t.replace(" ", ""), "AGAMA"
                                ).ratio()
                                if ("AGAMA" in t) or ratio >= 0.72:
                                    if i + 1 < len(lines):
                                        cand = str(lines[i + 1][1][0])
                                        rel = parse_ktp_text(cand).get("agama")
                                        if rel:
                                            ktp_fields["agama"] = rel
                                            break
                        if not ktp_fields.get("pekerjaan"):
                            for i, ln in enumerate(lines):
                                t = str(ln[1][0]).upper()
                                ratio = difflib.SequenceMatcher(
                                    None, t.replace(" ", ""), "PEKERJAAN"
                                ).ratio()
                                if ("PEKERJAAN" in t) or ratio >= 0.65:
                                    if i + 1 < len(lines):
                                        cand = str(lines[i + 1][1][0])
                                        pek = parse_ktp_text(cand).get("pekerjaan")
                                        if pek:
                                            ktp_fields["pekerjaan"] = pek
                                            break
                except Exception:
                    pass

        # Merge header-derived fields when base is missing
        merged_sources = {}
        if provinsi_text:
            prov_fields = parse_ktp_text(provinsi_text)
            if (not ktp_fields.get("provinsi")) and prov_fields.get("provinsi"):
                ktp_fields["provinsi"] = prov_fields["provinsi"]
                merged_sources["provinsi"] = "header_ocr"
        if kota_text:
            kota_fields = parse_ktp_text(kota_text)
            if (not ktp_fields.get("kota")) and kota_fields.get("kota"):
                ktp_fields["kota"] = kota_fields["kota"]
                merged_sources["kota"] = "header_ocr"

        # Confidence and thresholds
        field_confidences = compute_field_confidence(ktp_fields)
        default_thresholds = {
            "nik": 0.90,
            "nama": 0.70,
            "provinsi": 0.80,
            "kota": 0.70,
            "tempat_lahir": 0.60,
            "tanggal_lahir": 0.70,
            "jenis_kelamin": 0.80,
            "alamat": (
                alamat_threshold_value if "alamat_threshold_value" in globals() else 0.5
            ),
            "rt_rw": 0.80,
            "kel_desa": 0.60,
            "kecamatan": 0.60,
            "agama": 0.70,
            "status_perkawinan": 0.70,
            "pekerjaan": 0.70,
            "kewarganegaraan": 0.90,
            "berlaku_hingga": 0.80,
        }

        def _apply_thresholds(fields: dict, conf: dict, thr: dict) -> dict:
            out = {}
            for k, v in (fields or {}).items():
                if v in (None, ""):
                    out[k] = None
                    continue
                c = float(conf.get(k, 0.0))
                m = float(thr.get(k, 0.60))
                out[k] = v if c >= m else None
            return out

        processed_fields = _apply_thresholds(
            ktp_fields, field_confidences, default_thresholds
        )

        # Metadata
        metadata = {
            "processing_time": f"{time.time() - t0:.2f}s",
            "image_size": f"{frame.shape[1]}x{frame.shape[0]}",
            "crop_size": f"{crop.shape[1]}x{crop.shape[0]}",
            "raw_ocr_text": decoded_text,
            "header_ocr_text": header_full_text,
            "header_provinsi_line": provinsi_text,
            "header_kota_line": kota_text,
            "yolo_box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
            "confidence": {
                "yolo_confidence": largest_conf,
                "ocr_avg_token_prob": None,
                "ocr_token_count": None,
                "overall": None,
            },
            "field_confidence": field_confidences,
            "field_thresholds": default_thresholds,
            "header_sources": merged_sources,
        }

        return create_response(True, fields=processed_fields, metadata=metadata)

    except Exception as e:
        return create_response(False, error=str(e))


def process_image_folder(
    folder_path, processor, model, yolo, device, paddle_ocr=None, enable_fallback=True
):
    results = []
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if any(f.lower().endswith(ext) for ext in image_extensions)
    ]
    if not image_files:
        return create_response(False, error=f"No image files found in {folder_path}")
    t0 = time.time()
    for image_path in image_files:
        print(f"\nProcessing {os.path.basename(image_path)}...")
        result = process_image(
            image_path, processor, model, yolo, device, paddle_ocr, enable_fallback
        )
        result["metadata"]["file_name"] = os.path.basename(image_path)
        results.append(result)
    total_time = time.time() - t0
    metadata = {
        "total_images": len(image_files),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "total_processing_time": f"{total_time:.2f}s",
    }
    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata,
        "results": results,
    }


def run_camera_mode(processor, model, yolo, device, paddle_ocr, enable_fallback=True):
    if CAMERA_INDEX is not None:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(DROIDCAM_URL)
    if not cap.isOpened():
        print(
            json.dumps(create_response(False, error="Failed to open camera"), indent=2)
        )
        return
    print(
        "Camera opened. Press 'c' to capture and OCR the largest detected card, 'q' to quit."
    )
    last_response = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print(
                json.dumps(
                    create_response(False, error="Failed to grab frame"), indent=2
                )
            )
            break
        disp = cv2.resize(frame, (960, 640))
        cv2.imshow("Preview (c=capture, q=quit)", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            print("\nCapturing frame and running detection/OCR...")
            t0 = time.time()
            last_response = process_image(
                frame, processor, model, yolo, device, paddle_ocr, enable_fallback
            )
            if last_response["status"] == "success":
                last_response["metadata"][
                    "processing_time"
                ] = f"{time.time() - t0:.2f}s"
            print(json.dumps(last_response, indent=2))
    cap.release()
    cv2.destroyAllWindows()


def main():
    # ------------------------------
    # CLI arguments (parse first so devices can be honored during model load)
    # ------------------------------
    parser = argparse.ArgumentParser(description="KTP OCR with JSON output")
    parser.add_argument(
        "--mode",
        choices=["camera", "file", "folder"],
        default="camera",
        help="Run mode: live camera, single file, or folder of images",
    )
    parser.add_argument(
        "--input", type=str, help="Input image path or folder (for file/folder mode)"
    )
    parser.add_argument("--output", type=str, help="Output JSON file path (optional)")
    parser.add_argument(
        "--yolo-device",
        type=str,
        default="auto",
        help="Device for YOLO: auto|cpu|cuda",
    )
    parser.add_argument(
        "--donut-device",
        type=str,
        default="auto",
        help="Device for Donut encoder-decoder: auto|cpu|cuda",
    )
    parser.add_argument(
        "--paddle-use-gpu",
        action="store_true",
        help="Use GPU for PaddleOCR (requires paddlepaddle-gpu installed)",
    )
    parser.add_argument(
        "--alamat-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for 'alamat' field (default: 0.5)",
    )
    parser.add_argument(
        "--enable-fallback",
        action="store_true",
        help="Enable PaddleOCR fallback for agama/pekerjaan fields (slower but more accurate)",
    )
    args = parser.parse_args()

    def pick_device(arg: str) -> str:
        a = (arg or "auto").lower()
        if a == "cpu":
            return "cpu"
        if a in ("cuda", "gpu"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        # auto
        return "cuda" if torch.cuda.is_available() else "cpu"

    donut_device = pick_device(args.donut_device)
    yolo_device = pick_device(args.yolo_device)

    # Planned devices from CLI
    print(
        f"Using devices -> Donut: {donut_device}, YOLO: {yolo_device}, Paddle GPU: {args.paddle_use_gpu}"
    )
    # Diagnostics: PyTorch CUDA availability
    try:
        print(
            "Torch CUDA available:",
            bool(torch.cuda.is_available()),
            "| torch.version.cuda:",
            getattr(torch.version, "cuda", None),
        )
        if torch.cuda.is_available():
            try:
                print("Torch GPU device:", torch.cuda.get_device_name(0))
            except Exception:
                pass
    except Exception:
        pass
    # expose alamat threshold to processing scope
    global alamat_threshold_value
    alamat_threshold_value = float(getattr(args, "alamat_threshold", 0.75))

    # ------------------------------
    # Load models
    # ------------------------------
    print("Loading Donut processor and model...")
    processor = DonutProcessor.from_pretrained(MODEL_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
    model.to(donut_device)
    try:
        print("Donut model actual device:", next(model.parameters()).device)
    except Exception:
        pass

    print("Loading YOLO model...")
    yolo = YOLO(YOLO_MODEL)
    # Try to move YOLO to the requested device (Ultralytics >=8 supports .to)
    try:
        yolo.to(yolo_device)
    except Exception:
        # Fallback: rely on Ultralytics internal device selection during inference
        pass
    try:
        # Best-effort: print YOLO internal device if exposed
        yolo_device_attr = getattr(getattr(yolo, "model", None), "device", None)
        if yolo_device_attr is not None:
            print("YOLO model actual device:", yolo_device_attr)
    except Exception:
        pass

    print("Loading PaddleOCR for header detection...")
    paddle_ocr = None
    try:
        # PaddleOCR 3.3+ doesn't support use_gpu parameter anymore
        # It automatically uses CPU by default
        paddle_ocr = PaddleOCR(use_textline_orientation=True, lang="id")
        # Diagnostics: Paddle device
        try:
            import paddle

            dev = None
            try:
                dev = paddle.device.get_device()
            except Exception:
                pass
            print("Paddle device:", dev)
        except Exception:
            pass
    except Exception as e:
        # PaddleOCR failed to load (import error, conflict, etc.)
        print(f"⚠️  PaddleOCR unavailable (header OCR disabled): {type(e).__name__}")
        print("   → PyTorch GPU will still work for Donut & YOLO")
        paddle_ocr = None

    if args.mode == "camera":
        # In camera mode we pass the Donut (encoder-decoder) device onward
        run_camera_mode(
            processor,
            model,
            yolo,
            donut_device,
            paddle_ocr,
            enable_fallback=bool(args.enable_fallback),
        )
        return

    if not args.input or not os.path.exists(args.input):
        print(
            json.dumps(
                create_response(False, error="Valid --input is required"), indent=2
            )
        )
        return

    t0 = time.time()
    if args.mode == "file":
        result = process_image(
            args.input,
            processor,
            model,
            yolo,
            donut_device,
            paddle_ocr,
            enable_fallback=bool(args.enable_fallback),
        )
        result["metadata"]["processing_time"] = f"{time.time() - t0:.2f}s"
        result["metadata"]["file_name"] = os.path.basename(args.input)
    else:
        result = process_image_folder(
            args.input,
            processor,
            model,
            yolo,
            donut_device,
            paddle_ocr,
            enable_fallback=bool(args.enable_fallback),
        )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {args.output}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
