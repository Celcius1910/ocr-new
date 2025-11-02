import json

with open("results_final_timing.json", "r", encoding="utf-8") as f:
    data = json.load(f)

metadata = data["metadata"]
results = data["results"]

print("=" * 90)
print("OCR PROCESSING TIME ANALYSIS")
print("=" * 90)

print(f"\n📊 OVERALL STATISTICS:")
print(f'   Total images processed: {metadata["total_images"]}')
print(
    f'   Successful: {metadata["successful"]} ({metadata["successful"]/metadata["total_images"]*100:.1f}%)'
)
print(
    f'   Failed: {metadata["failed"]} ({metadata["failed"]/metadata["total_images"]*100:.1f}%)'
)
print(f'   Total processing time: {metadata["total_processing_time"]}')
print(f'   Average time per image: {metadata["avg_time_per_image"]}')
print(f'   Fastest: {metadata["min_time"]}')
print(f'   Slowest: {metadata["max_time"]}')

print(f"\n⏱️  PROCESSING TIME PER FILE:")
print("=" * 90)
print(f'{"#":<4} {"File Name":<40} {"Time":<10} {"Status":<10}')
print("-" * 90)

for i, r in enumerate(results, 1):
    file_name = r["metadata"].get("file_name", "N/A")
    proc_time = r["metadata"].get("processing_time", "N/A")
    status = r["status"]

    status_icon = "✅" if status == "success" else "❌"
    print(f"{i:<4} {file_name:<40} {proc_time:<10} {status_icon} {status}")

print("=" * 90)

# Show slowest and fastest files
successful_results = [
    (
        r["metadata"].get("file_name", "N/A"),
        float(r["metadata"].get("processing_time", "0s").rstrip("s")),
    )
    for r in results
    if r["status"] == "success" and "processing_time" in r["metadata"]
]

if successful_results:
    successful_results.sort(key=lambda x: x[1])

    print(f"\n🐌 TOP 5 SLOWEST FILES:")
    for i, (fname, time) in enumerate(successful_results[-5:][::-1], 1):
        print(f"   {i}. {fname:<40} {time:.2f}s")

    print(f"\n🚀 TOP 5 FASTEST FILES:")
    for i, (fname, time) in enumerate(successful_results[:5], 1):
        print(f"   {i}. {fname:<40} {time:.2f}s")

print("\n" + "=" * 90)
