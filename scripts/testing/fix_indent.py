# Temp fix snippet
# Lines 218-235 should be:

            # Also do full crop OCR to extract fields Donut missed (agama, pekerjaan)
            # Only if agama or pekerjaan is missing
            if not ktp_fields.get("agama") or not ktp_fields.get("pekerjaan"):
                try:
                    paddle_result = paddle_ocr.ocr(crop, cls=False)
                    if paddle_result and paddle_result[0]:
                        full_paddle_text = " ".join([line[1][0] for line in paddle_result[0]])
                        print(f"DEBUG: {full_paddle_text[:150]}")
                        extra_fields = parse_ktp_text(full_paddle_text)
                        print(f"DEBUG Agama={extra_fields.get('agama')}")
                        if not ktp_fields.get("agama") and extra_fields.get("agama"):
                            ktp_fields["agama"] = extra_fields["agama"]
                        if not ktp_fields.get("pekerjaan") and extra_fields.get("pekerjaan"):
                            ktp_fields["pekerjaan"] = extra_fields["pekerjaan"]
                except Exception as e:
                    print(f"DEBUG Error: {e}")
