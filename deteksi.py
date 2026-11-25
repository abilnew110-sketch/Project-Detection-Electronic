import argparse
from collections import Counter
from pathlib import Path

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

WATT_PERKIRAAN = {
    "TV": 100,
    "Laptop": 65,
    "Handphone": 10,
    "Mouse": 2,
    "Keyboard": 5,
    "Remote": 1,
    "Kulkas": 150,
    "Microwave": 1200,
    "Oven": 2000,
    "Toaster": 1000,
    "Pengering Rambut": 1500,
}

HIJAU = (0, 200, 0)
PUTIH = (255, 255, 255)
MERAH = (0, 0, 255)

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

    return deteksi

def hitung_daya(deteksi, jam):
    jumlah_per_nama = Counter(item["nama"] for item in deteksi)
    total_watt = sum(WATT_PERKIRAAN[nama] * jumlah for nama, jumlah in jumlah_per_nama.items())
    return jumlah_per_nama, total_watt, total_watt * jam / 1000

def gambarkan_hasil(frame, deteksi, jam):
    jumlah_per_nama, total_watt, total_kwh = hitung_daya(deteksi, jam)

    for item in deteksi:
        x1, y1, x2, y2 = item["box"]
        nama = item["nama"]
        watt = WATT_PERKIRAAN[nama]
        cv2.rectangle(frame, (x1, y1), (x2, y2), HIJAU, 2)
        cv2.putText(
            frame,
            f"{nama} {item['confidence']:.2f} (~{watt} W)",
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
        watt = WATT_PERKIRAAN[nama] * jumlah
        cv2.putText(
            frame,
            f"{nama}: {jumlah} ({watt} W)",
            (10, posisi_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            PUTIH,
            2,
        )
        posisi_y += 25

    cv2.putText(
        frame,
        f"Perkiraan: {total_watt} W | {total_kwh:.2f} kWh untuk {jam:g} jam",
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
                print(f"  {nama}: {count} (~{WATT_PERKIRAAN[nama] * count} W)")
            print(f"  Total perkiraan: {watt} W / {kwh:.2f} kWh untuk {args.hours:g} jam")
            print(f"  Hasil tersimpan: {output_path}")
        return

    kamera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)

    if not kamera.isOpened():
        print(
            f"Kamera index {args.camera} tidak dapat dibuka. "
            "Coba index lain atau tutup aplikasi lain yang menggunakan kamera."
        )
        return

    try:
        while True:
            berhasil, frame = kamera.read()
            if not berhasil:
                print("Gagal membaca frame dari kamera.")
                break

            deteksi = deteksi_elektronik(model, frame, args.confidence)
            gambarkan_hasil(frame, deteksi, args.hours)

            cv2.imshow("Deteksi Elektronik", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        kamera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()