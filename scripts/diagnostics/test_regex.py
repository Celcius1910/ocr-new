import re

# Test cases from actual OCR output
test_cases = [
    "PBOVINSHJAWABARAT KABUPATEN BEKASI NAUFALAZIZ MAUEANA",
    "PROVINS BANTEN KABUPATEN TANGERANG",
    "PROVINSI SUMATERA UTARA KOTA MEDAN NIK",
    "PROVINSI KEPULAUANRIAU KOTA BATAM NIK NAMA",
    "KOTA SURABAYA NIK",
    "PROVINSI JAWA TIMUF KOTA SURABAYA NIK",
    "PROVINSI SUMATERA UTARA KOTA MEDAN NIK",
]

for test in test_cases:
    # Bersihkan characters yang bukan huruf atau space
    up = re.sub(r"[^A-Z\s]", "", test.upper())
    up = re.sub(r"\s+", " ", up).strip()

    print(f"\nOriginal: {test}")
    print(f"Cleaned:  {up}")

    # Test provinsi pattern - stop before KOTA/KABUPATEN or other keywords
    m_p = re.search(
        r"(PROVINSI|PROPINSI|PROVINS|PBOVINS|PBOVINSH|PROV)\s*([A-Z]+(?:\s+[A-Z]+){0,3}?)(?=\s+(KOTA|KABUPATEN|KAB|NIK|NAMA)|$)",
        up,
    )
    if m_p:
        prov_keyword = m_p.group(1)
        name = m_p.group(2).strip()

        if prov_keyword in ["PBOVINS", "PBOVINSH"]:
            name = re.sub(r"^[HIJ]", "", name)
        name = re.sub(r"^(I|H|J)", "", name)

        print(f"  PROVINSI matched: {prov_keyword} -> {name}")
    else:
        print("  PROVINSI: NO MATCH")

    # Test kota pattern - stop before NIK/NAMA or other keywords
    m_k = re.search(
        r"(KOTA|KABUPATEN|KOTAKABUPATEN|KAB)\s*([A-Z]+(?:\s+[A-Z]+){0,2}?)(?=\s+(NIK|NAMA|TEMPAT|JENIS|GOL|DARAH|ALAMAT)|$)",
        up,
    )
    if m_k:
        kota_name = m_k.group(2).strip()
        kota_type = m_k.group(1).strip()
        print(f"  KOTA/KAB matched: {kota_type} -> {kota_name}")
    else:
        print("  KOTA/KAB: NO MATCH")
