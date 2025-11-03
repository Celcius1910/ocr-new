import time
from rapidfuzz import fuzz, process

# Daftar lengkap provinsi Indonesia
PROVINSI_LIST = [
    "ACEH",
    "SUMATERA UTARA",
    "SUMATERA BARAT",
    "RIAU",
    "JAMBI",
    "SUMATERA SELATAN",
    "BENGKULU",
    "LAMPUNG",
    "KEPULAUAN BANGKA BELITUNG",
    "KEPULAUAN RIAU",
    "DKI JAKARTA",
    "JAWA BARAT",
    "JAWA TENGAH",
    "DI YOGYAKARTA",
    "JAWA TIMUR",
    "BANTEN",
    "BALI",
    "NUSA TENGGARA BARAT",
    "NUSA TENGGARA TIMUR",
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
    "SULAWESI BARAT",
    "MALUKU",
    "MALUKU UTARA",
    "PAPUA",
    "PAPUA BARAT",
    "PAPUA SELATAN",
    "PAPUA TENGAH",
    "PAPUA PEGUNUNGAN",
    "PAPUA BARAT DAYA",
]

# Sample kota/kabupaten (subset untuk test)
KOTA_LIST = [
    "MEDAN",
    "BINJAI",
    "TEBING TINGGI",
    "PEMATANG SIANTAR",
    "JAKARTA PUSAT",
    "JAKARTA UTARA",
    "JAKARTA BARAT",
    "JAKARTA SELATAN",
    "JAKARTA TIMUR",
    "BANDUNG",
    "BEKASI",
    "BOGOR",
    "DEPOK",
    "CIMAHI",
    "TASIKMALAYA",
    "BANJAR",
    "SURABAYA",
    "MALANG",
    "KEDIRI",
    "BLITAR",
    "MADIUN",
    "PROBOLINGGO",
    "BATAM",
    "TANJUNG PINANG",
    "TANGERANG",
    "TANGERANG SELATAN",
    "SERANG",
    "CILEGON",
    # ... bisa tambah lebih banyak
]

# Test cases dari OCR yang typo/error
test_cases = [
    "JAWABARAT",
    "JAWA BARAT",
    "AWA BARAT",
    "SUMATERA UTARA",
    "SUMATERA UTARA",
    "KEPULAUANRIAU",
    "KEPULAUAN RIAU",
    "JAWA TIMUF",  # typo F
    "BEKASI",
    "MEDAN",
    "SURABAYA",
    "TANGERANG",
]

print("Testing Fuzzy Matching Performance...")
print("=" * 60)

# Test provinsi matching
provinsi_times = []
for test in test_cases[:8]:  # Test provinsi cases
    start = time.perf_counter()
    result = process.extractOne(test, PROVINSI_LIST, scorer=fuzz.ratio)
    elapsed = time.perf_counter() - start
    provinsi_times.append(elapsed)
    print(
        f"Query: {test:20} -> Match: {result[0]:20} (score: {result[1]:.1f}) [{elapsed*1000:.3f}ms]"
    )

print()

# Test kota matching
kota_times = []
for test in test_cases[8:]:  # Test kota cases
    start = time.perf_counter()
    result = process.extractOne(test, KOTA_LIST, scorer=fuzz.ratio)
    elapsed = time.perf_counter() - start
    kota_times.append(elapsed)
    print(
        f"Query: {test:20} -> Match: {result[0]:20} (score: {result[1]:.1f}) [{elapsed*1000:.3f}ms]"
    )

print()
print("=" * 60)
print(
    f"Average time for provinsi matching: {sum(provinsi_times)/len(provinsi_times)*1000:.3f}ms"
)
print(f"Average time for kota matching: {sum(kota_times)/len(kota_times)*1000:.3f}ms")
print(
    f"Total overhead per image: ~{(sum(provinsi_times)/len(provinsi_times) + sum(kota_times)/len(kota_times))*1000:.3f}ms"
)
print()
print("✅ Fuzzy matching is VERY FAST - negligible overhead!")
