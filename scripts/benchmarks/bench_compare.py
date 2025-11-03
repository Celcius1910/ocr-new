import json
from statistics import mean


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summarize(run):
    results = run.get("results", [run])
    total_images = len(results)
    per_times = []
    prov_ok = 0
    kota_ok = 0

    for r in results:
        mt = r.get("metadata", {})
        pt = mt.get("processing_time")
        if pt and isinstance(pt, str) and pt.endswith("s"):
            try:
                per_times.append(float(pt[:-1]))
            except Exception:
                pass
        d = r.get("data", {})
        if d.get("provinsi"):
            prov_ok += 1
        if d.get("kota") or d.get("kota_kabupaten"):
            kota_ok += 1

    meta = run.get("metadata", {})
    total_time_s = None
    if meta.get("total_processing_time") and meta["total_processing_time"].endswith(
        "s"
    ):
        try:
            total_time_s = float(meta["total_processing_time"][:-1])
        except Exception:
            pass

    return {
        "total_images": total_images,
        "total_time_s": total_time_s,
        "avg_time_s": mean(per_times) if per_times else None,
        "min_time_s": min(per_times) if per_times else None,
        "max_time_s": max(per_times) if per_times else None,
        "fps": (
            (total_images / total_time_s)
            if (total_time_s and total_time_s > 0)
            else None
        ),
        "provinsi_hit": prov_ok,
        "kota_hit": kota_ok,
    }


def print_row(name, s):
    def fmt(x, unit="s"):
        if x is None:
            return "-"
        if unit == "s":
            return f"{x:.2f}s"
        if unit == "fps":
            return f"{x:.2f} img/s"
        return str(x)

    print(
        f"{name:10} | imgs: {s['total_images']:2d} | total: {fmt(s['total_time_s'])} | avg: {fmt(s['avg_time_s'])} | min: {fmt(s['min_time_s'])} | max: {fmt(s['max_time_s'])} | fps: {fmt(s['fps'], 'fps')} | prov: {s['provinsi_hit']:2d} | kota: {s['kota_hit']:2d}"
    )


def main():
    gpu = load("outputs/bench_gpu.json")
    cpu = load("outputs/bench_cpu.json")
    sg = summarize(gpu)
    sc = summarize(cpu)

    print("\nBenchmark comparison (sample_ocr_ktp_axa)")
    print("=" * 100)
    print(
        "Variant    | imgs | total    | avg      | min      | max      | fps         | prov | kota"
    )
    print("-" * 100)
    print_row("GPU", sg)
    print_row("CPU", sc)

    if sg["total_time_s"] and sc["total_time_s"]:
        speedup = (
            sc["total_time_s"] / sg["total_time_s"] if sg["total_time_s"] > 0 else None
        )
        print("-" * 100)
        if speedup:
            print(f"Speedup (CPU -> GPU): {speedup:.2f}x faster on GPU")


if __name__ == "__main__":
    main()
