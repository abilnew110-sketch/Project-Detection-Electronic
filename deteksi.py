import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import time

import cv2
from ultralytics import YOLO

ELEKTRONIK = {
    "tv": "TV",
    "laptop": "Laptop",
    "cell phone": "Handphone",
    "mouse": "Mouse",
    "keyboard": "Keyboard",
    "remote": "Remote",
    "refrigerator": "Kulkas",
    "microwave": "Microwave",
    "oven": "Oven",
    "toaster": "Toaster",
    "hair drier": "Pengering Rambut",
}

WATT_PERANGKAT = {
    # minimum, daya penggunaan umum, maksimum/nominal
    "TV": (40, 100, 300),
    "Laptop": (30, 65, 150),
    "Handphone": (5, 10, 25),
    "Mouse": (1, 2, 5),
    "Keyboard": (3, 5, 10),
    "Remote": (0.5, 1, 2),
    "Kulkas": (80, 150, 300),
    "Microwave": (700, 1200, 1500),
    "Oven": (1200, 2000, 3500),
    "Toaster": (700, 1000, 1500),
    "Pengering Rambut": (800, 1500, 2400),
}

# Faktor pemakaian rata-rata terhadap daya nominal/pasangan penggunaan.
FAKTOR_PEMAKAIAN = {
    "TV": 1.0,
    "Laptop": 1.0,
    "Handphone": 0.1,
    "Mouse": 1.0,
    "Keyboard": 1.0,
    "Remote": 1.0,
    "Kulkas": 0.35,
    "Microwave": 0.20,
    "Oven": 0.50,
    "Toaster": 0.25,
    "Pengering Rambut": 0.80,
}

HIJAU = (0, 200, 0)
PUTIH = (255, 255, 255)
MERAH = (0, 0, 255)

def hitung_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    luas_irisan = max(0, x2 - x1) * max(0, y2 - y1)
    luas_a = max(1, box_a[2] - box_a[0]) * max(1, box_a[3] - box_a[1])
    luas_b = max(1, box_b[2] - box_b[0]) * max(1, box_b[3] - box_b[1])
    return luas_irisan / (luas_a + luas_b - luas_irisan)


def deteksi_elektronik(model, frame, confidence):
    hasil = model(frame, conf=confidence, verbose=False)
    deteksi = []

    for kotak in hasil[0].boxes:
        class_id = int(kotak.cls.item())
        nama = model.names[class_id]
        if nama not in ELEKTRONIK:
            continue

        deteksi.append({
            "nama": ELEKTRONIK[nama],
            "confidence": float(kotak.conf.item()),
            "box": kotak.xyxy[0].int().tolist(),
        })

    # Beberapa kotak YOLO dari objek yang sama dapat menghasilkan duplikat.
    # Simpan kotak terbaik dari setiap objek yang sangat bertumpang tindih.
    deteksi = sorted(deteksi, key=lambda item: item["confidence"], reverse=True)
    deteksi_unik = []
    for item in deteksi:
        duplikat = any(
            item["nama"] == item_lain["nama"]
            and hitung_iou(item["box"], item_lain["box"]) >= 0.45
            for item_lain in deteksi_unik
        )
        if not duplikat:
            deteksi_unik.append(item)

    deteksi = deteksi_unik
    return deteksi

def hitung_daya(deteksi, jam):
    jumlah_per_nama = Counter(item["nama"] for item in deteksi)
    total_watt = sum(
        WATT_PERANGKAT[nama][1] * FAKTOR_PEMAKAIAN[nama] * jumlah
        for nama, jumlah in jumlah_per_nama.items()
    )
    return jumlah_per_nama, total_watt, total_watt * jam / 1000

def ringkasan_kamera(jumlah_maksimum, durasi_detik):
    durasi_jam = durasi_detik / 3600
    baris = []
    total_minimum = 0
    total_umum = 0
    total_maksimum = 0
    total_rata_rata = 0

    if not jumlah_maksimum:
        return (
            "Tidak ada perangkat elektronik yang dikenali selama kamera aktif.",
            {
                "durasi_detik": durasi_detik,
                "durasi_jam": durasi_jam,
                "total_minimum_watt": 0,
                "total_umum_watt": 0,
                "total_maksimum_watt": 0,
                "total_rata_rata_watt": 0,
                "kwh_rata_rata": 0,
            },
        )

    for nama, jumlah in sorted(jumlah_maksimum.items()):
        nilai_min, nilai_umum, nilai_max = WATT_PERANGKAT[nama]
        rata_rata = nilai_umum * FAKTOR_PEMAKAIAN[nama]
        minimum = nilai_min * jumlah
        umum = nilai_umum * jumlah
        maksimum = nilai_max * jumlah
        rata_satuan = rata_rata * jumlah
        total_minimum += minimum
        total_umum += umum
        total_maksimum += maksimum
        total_rata_rata += rata_satuan
        baris.append(
            f"- {nama}: {jumlah} perangkat, "
            f"rentang {minimum:.0f}-{maksimum:.0f} W, "
            f"rata-rata penggunaan ~{rata_satuan:.0f} W "
            f"(nilai umum {umum:.0f} W)"
        )

    kwh_rata_rata = total_rata_rata * durasi_jam / 1000
    ringkasan = "\n".join([
        "PERANGKAT TERDETEKSI SELAMA KAMERA HIDUP",
        *baris,
        "",
        f"Durasi kamera aktif: {durasi_detik:.1f} detik ({durasi_jam:.4f} jam)",
        f"Total minimum perkiraan: {total_minimum:.0f} W",
        f"Total nilai umum perkiraan: {total_umum:.0f} W",
        f"Total maksimum perkiraan: {total_maksimum:.0f} W",
        f"TOTAL WATT RATA-RATA PERKIRAAN: {total_rata_rata:.0f} W",
        f"Estimasi konsumsi selama sesi: {kwh_rata_rata:.4f} kWh",
    ])
    data = {
        "durasi_detik": durasi_detik,
        "durasi_jam": durasi_jam,
        "total_minimum_watt": total_minimum,
        "total_umum_watt": total_umum,
        "total_maksimum_watt": total_maksimum,
        "total_rata_rata_watt": total_rata_rata,
        "kwh_rata_rata": kwh_rata_rata,
    }
    return ringkasan, data

def gambarkan_hasil(frame, deteksi, jam):
    jumlah_per_nama, total_watt, total_kwh = hitung_daya(deteksi, jam)

    for item in deteksi:
        x1, y1, x2, y2 = item["box"]
        nama = item["nama"]
        nilai_min, watt, nilai_max = WATT_PERANGKAT[nama]
        cv2.rectangle(frame, (x1, y1), (x2, y2), HIJAU, 2)
        cv2.putText(
            frame,
            f"{nama} {item['confidence']:.2f} ({watt:.0f} W umum)",
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            HIJAU,
            2,
        )

    cv2.putText(
        frame,
        f"Elektronik terdeteksi: {len(deteksi)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        PUTIH,
        2,
    )

    posisi_y = 50
    for nama, jumlah in jumlah_per_nama.items():
        watt_rata_rata = (
            WATT_PERANGKAT[nama][1] * FAKTOR_PEMAKAIAN[nama] * jumlah
        )
        cv2.putText(
            frame,
            f"{nama}: {jumlah} (~{watt_rata_rata:.0f} W rata-rata)",
            (10, posisi_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            PUTIH,
            2,
        )
        posisi_y += 25

    cv2.putText(
        frame,
        f"Perkiraan: {total_watt:.0f} W rata-rata | {total_kwh:.2f} kWh / {jam:g} jam",
        (10, posisi_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        MERAH,
        2,
    )

    return jumlah_per_nama, total_watt, total_kwh

def main():
    parser = argparse.ArgumentParser(
        description="Deteksi perangkat elektronik dari kamera atau gambar."
    )
    parser.add_argument("--camera", type=int, default=0, help="Index kamera")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path satu gambar atau folder gambar.",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=1,
        help="Durasi penggunaan dalam jam untuk estimasi kWh.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Folder output untuk hasil deteksi gambar.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Ambang batas confidence deteksi",
    )
    args = parser.parse_args()

    if args.hours <= 0:
        parser.error("--hours harus lebih besar dari 0")

    base_dir = Path(__file__).resolve().parent
    model = YOLO(str(base_dir / "yolov8n.pt"))

    if args.image:
        sumber = Path(args.image)
        if not sumber.is_absolute():
            sumber = base_dir / sumber

        if sumber.is_dir():
            daftar_gambar = sorted(
                path for path in sumber.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            )
        elif sumber.is_file():
            daftar_gambar = [sumber]
        else:
            print(f"Gambar atau folder tidak ditemukan: {sumber}")
            return

        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = base_dir / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if not daftar_gambar:
            print(f"Tidak ada gambar yang ditemukan pada: {sumber}")
            return

        for gambar in daftar_gambar:
            frame = cv2.imread(str(gambar))
            if frame is None:
                print(f"Gagal membaca gambar: {gambar}")
                continue

            deteksi = deteksi_elektronik(model, frame, args.confidence)
            jumlah, watt, kwh = gambarkan_hasil(frame, deteksi, args.hours)
            output_path = output_dir / f"{gambar.stem}_deteksi.jpg"
            cv2.imwrite(str(output_path), frame)

            print(f"\n{gambar.name}:")
            if not jumlah:
                print("  Tidak ada perangkat elektronik yang dikenali.")
            for nama, count in jumlah.items():
                nilai_min, nilai_umum, nilai_max = WATT_PERANGKAT[nama]
                rata_rata = nilai_umum * FAKTOR_PEMAKAIAN[nama] * count
                print(
                    f"  {nama}: {count} "
                    f"(rentang {nilai_min * count:.0f}-{nilai_max * count:.0f} W; "
                    f"rata-rata pemakaian ~{rata_rata:.0f} W)"
                )
            print(
                f"  Total perkiraan: {watt:.0f} W rata-rata / "
                f"{kwh:.2f} kWh untuk {args.hours:g} jam"
            )
            print(f"  Hasil tersimpan: {output_path}")
        return

    kamera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)

    if not kamera.isOpened():
        print(
            f"Kamera index {args.camera} tidak dapat dibuka. "
            "Coba index lain atau tutup aplikasi lain yang menggunakan kamera."
        )
        return

    jumlah_maksimum = Counter()
    waktu_mulai = time.monotonic()
    frame_terakhir = None
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            berhasil, frame = kamera.read()
            if not berhasil:
                print("Gagal membaca frame dari kamera.")
                break

            deteksi = deteksi_elektronik(model, frame, args.confidence)
            frame_terakhir = frame.copy()
            jumlah_frame, _, _ = hitung_daya(deteksi, 1)
            for nama, jumlah in jumlah_frame.items():
                jumlah_maksimum[nama] = max(jumlah_maksimum[nama], jumlah)
            gambarkan_hasil(frame, deteksi, args.hours)

            cv2.imshow("Deteksi Elektronik", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        durasi_detik = time.monotonic() - waktu_mulai
        ringkasan, data = ringkasan_kamera(jumlah_maksimum, durasi_detik)
        waktu_selesai = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_ringkasan = output_dir / f"ringkasan_kamera_{waktu_selesai}.txt"
        file_ringkasan.write_text(
            "KESIMPULAN DETEKSI DAN PERKIRAAN DAYA\n"
            f"Dicatat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "========================================\n\n"
            f"{ringkasan}\n\n"
            "Catatan: Watt dan kWh merupakan perkiraan rata-rata karena model "
            "tidak mengetahui spesifikasi daya asli perangkat.\n",
            encoding="utf-8",
        )
        if frame_terakhir is not None:
            cv2.imwrite(
                str(output_dir / f"frame_terakhir_{waktu_selesai}.jpg"),
                frame_terakhir,
            )

        kamera.release()
        cv2.destroyAllWindows()
        print("\n========================================")
        print(ringkasan)
        print("========================================")
        print(f"Hasil ringkasan tersimpan: {file_ringkasan}")

if __name__ == "__main__":
    main()