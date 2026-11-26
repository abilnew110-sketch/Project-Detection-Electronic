import argparse
from collections import Counter
from datetime import datetime
import math
from pathlib import Path
import time

import cv2
from ultralytics import YOLO, YOLOWorld

PROMPT_KE_NAMA = {
    "television": "TV",
    "laptop computer": "Laptop",
    "smartphone": "Handphone",
    "computer keyboard": "Keyboard",
    "computer mouse": "Mouse",
    "remote control": "Remote",
    "headphones": "Headphone",
    "wireless earbuds": "Earbud Nirkabel",
    "tablet": "Tablet",
    "smartwatch": "Smartwatch",
    "digital camera": "Kamera Digital",
    "desktop computer": "Komputer Desktop",
    "computer monitor": "Monitor",
    "computer server": "Server",
    "printer": "Printer",
    "scanner": "Pemindai",
    "computer router": "Router",
    "computer speaker": "Speaker",
    "projector": "Proyektor",
    "hard drive": "Hard Disk",
    "circuit board": "Papan Sirkuit",
    "computer motherboard": "Motherboard",
    "refrigerator": "Kulkas",
    "washing machine": "Mesin Cuci",
    "clothes dryer": "Pengering Pakaian",
    "microwave oven": "Microwave",
    "conventional oven": "Oven",
    "electric stove": "Kompor Listrik",
    "induction cooktop": "Kompor Induksi",
    "pedestal fan": "Kipas Angin",
    "electric fan": "Kipas Angin",
    "vacuum cleaner": "Penyedot Debu",
    "blender": "Blender",
    "food processor": "Food Processor",
    "stand mixer": "Mixer Berdiri",
    "hand mixer": "Mixer Tangan",
    "immersion blender": "Blender Tangan",
    "electric kettle": "Ketel Listrik",
    "coffee maker": "Mesin Kopi",
    "espresso machine": "Mesin Espresso",
    "coffee grinder": "Penggiling Kopi",
    "meat grinder": "Penggiling Daging",
    "rice cooker": "Penanak Nasi",
    "electric rice cooker": "Penanak Nasi",
    "electric pressure cooker": "Panci Tekanan Listrik",
    "slow cooker": "Slow Cooker",
    "toaster": "Toaster",
    "electric iron": "Setrika",
    "hair dryer": "Pengering Rambut",
    "water dispenser": "Dispenser Air",
    "water cooler": "Dispenser Air",
    "portable air cooler": "Pendingin Udara Portabel",
    "air conditioner": "AC",
    "air purifier": "Pembersih Udara",
    "humidifier": "Humidifier",
    "electric heater": "Pemanas Listrik",
    "air fryer": "Air Fryer",
    "dishwasher": "Mesin Cuci Piring",
    "electric grill": "Pemanggang Listrik",
    "electric sandwich maker": "Pembuat Sandwich",
    "bread maker": "Pembuat Roti",
    "emergency light": "Lampu Darurat",
    "water heater": "Pemanas Air",
    "range hood": "Penghisap Asap Dapur",
}

KELAS_KUSTOM_KE_NAMA = {
    "blender": "Blender",
    "camera": "Kamera Digital",
    "fan": "Kipas Angin",
    "headphones": "Headphone",
    "keyboard": "Keyboard",
    "laptop": "Laptop",
    "microwave oven": "Microwave",
    "mixer": "Mixer",
    "mobile phone": "Handphone",
    "monitor": "Monitor",
    "mouse": "Mouse",
    "printer": "Printer",
    "refrigerator": "Kulkas",
    "tv": "TV",
    "tablet": "Tablet",
    "toaster": "Toaster",
    "washing machine": "Mesin Cuci",
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


def deteksi_elektronik(
    model, frame, confidence, imgsz, kelas_ke_nama=PROMPT_KE_NAMA, device=None
):
    hasil = model(
        frame, conf=confidence, imgsz=imgsz, device=device, verbose=False
    )
    deteksi = []

    for kotak in hasil[0].boxes:
        class_id = int(kotak.cls.item())
        prompt = str(model.names[class_id]).lower()
        nama = kelas_ke_nama.get(prompt)
        if nama is None:
            continue

        deteksi.append({
            "nama": nama,
            "confidence": float(kotak.conf.item()),
            "box": kotak.xyxy[0].int().tolist(),
        })

    deteksi = sorted(deteksi, key=lambda item: item["confidence"], reverse=True)
    deteksi_unik = []
    for item in deteksi:
        duplikat = any(
            hitung_iou(item["box"], item_lain["box"])
            >= (0.50 if item["nama"] == item_lain["nama"] else 0.80)
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
        if nama in WATT_PERANGKAT and nama in FAKTOR_PEMAKAIAN
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
        if nama not in WATT_PERANGKAT or nama not in FAKTOR_PEMAKAIAN:
            baris.append(f"- {nama}: {jumlah} perangkat, data daya belum tersedia")
            continue

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
        "Catatan: total daya hanya mencakup perangkat dengan data watt.",
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
    tinggi_frame, lebar_frame = frame.shape[:2]

    for item in deteksi:
        x1, y1, x2, y2 = item["box"]
        nama = item["nama"]
        label = f"{nama} {item['confidence']:.2f}"
        ukuran_teks, garis_dasar = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )[0], 3
        lebar_teks, tinggi_teks = ukuran_teks
        posisi_x = max(0, min(x1, lebar_frame - lebar_teks - 6))
        posisi_y = y1 - 5
        if posisi_y - tinggi_teks - garis_dasar < 0:
            posisi_y = min(y1 + tinggi_teks + garis_dasar + 5, tinggi_frame - 4)

        cv2.rectangle(frame, (x1, y1), (x2, y2), HIJAU, 2)
        cv2.rectangle(
            frame,
            (posisi_x, max(0, posisi_y - tinggi_teks - garis_dasar - 2)),
            (posisi_x + lebar_teks + 6, posisi_y + garis_dasar + 2),
            (0, 100, 0),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (posisi_x + 3, posisi_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            PUTIH,
            1,
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

    if jumlah_per_nama:
        perangkat_tanpa_daya = sum(
            jumlah for nama, jumlah in jumlah_per_nama.items()
            if nama not in WATT_PERANGKAT or nama not in FAKTOR_PEMAKAIAN
        )
        teks_daya = (
            f"{total_watt:.0f} W | {total_kwh:.2f} kWh/{jam:g} jam"
            if perangkat_tanpa_daya == 0
            else f"Data watt belum tersedia untuk {perangkat_tanpa_daya} perangkat"
        )
        cv2.putText(
            frame,
            teks_daya,
            (10, 50),
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
        default=0.30,
        help="Ambang batas confidence deteksi",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Ukuran input model; nilai lebih besar dapat membantu objek kecil",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path bobot model inferensi; default mengikuti --model-type.",
    )
    parser.add_argument(
        "--model-type",
        choices=("world", "custom"),
        default="world",
        help="Gunakan YOLO-World atau bobot YOLO kustom hasil pelatihan.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Latih model YOLO kustom dari dataset Open Images yang disiapkan.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/open_images_v6/yolo/data.yaml",
        help="Path data.yaml untuk mode pelatihan.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolov8n.pt",
        help="Bobot awal YOLO untuk pelatihan kustom.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Jumlah epoch pelatihan.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Ukuran batch pelatihan.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Jumlah worker pemuatan data saat pelatihan.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Perangkat pelatihan/inferensi, misalnya 0 atau cpu.",
    )
    args = parser.parse_args()

    if not math.isfinite(args.hours) or args.hours <= 0:
        parser.error("--hours harus berupa angka hingga yang lebih besar dari 0")
    if not math.isfinite(args.confidence) or not 0 < args.confidence <= 1:
        parser.error("--confidence harus berada di antara 0 dan 1")
    if args.imgsz <= 0:
        parser.error("--imgsz harus lebih besar dari 0")
    if args.epochs <= 0:
        parser.error("--epochs harus lebih besar dari 0")
    if args.batch <= 0:
        parser.error("--batch harus lebih besar dari 0")
    if args.workers < 0:
        parser.error("--workers tidak boleh negatif")

    base_dir = Path(__file__).resolve().parent
    if args.train:
        if args.image:
            parser.error("--image tidak dapat digunakan bersama --train")
        dataset_path = Path(args.data)
        if not dataset_path.is_absolute():
            dataset_path = base_dir / dataset_path
        if not dataset_path.is_file():
            parser.error(f"File data.yaml tidak ditemukan: {dataset_path}")

        base_model_path = Path(args.base_model)
        if not base_model_path.is_absolute():
            base_model_path = base_dir / base_model_path
        model = YOLO(str(base_model_path))
        model.train(
            data=str(dataset_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            project=str(base_dir / "runs"),
            name="electronics",
            exist_ok=True,
            device=args.device,
        )
        return

    model_path = Path(args.model) if args.model else None
    if model_path is not None and not model_path.is_absolute():
        model_path = base_dir / model_path

    if args.model_type == "custom":
        if model_path is None:
            model_path = base_dir / "runs" / "electronics" / "weights" / "best.pt"
        if not model_path.is_file():
            parser.error(
                f"Bobot model kustom tidak ditemukan: {model_path}. "
                "Latih dulu dengan --train."
            )
        model = YOLO(str(model_path))
        kelas_ke_nama = KELAS_KUSTOM_KE_NAMA
    else:
        if model_path is None:
            model_path = base_dir / "yolov8s-worldv2.pt"
        model = YOLOWorld(str(model_path))
        model.set_classes(list(PROMPT_KE_NAMA))
        kelas_ke_nama = PROMPT_KE_NAMA

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

            deteksi = deteksi_elektronik(
                model,
                frame,
                args.confidence,
                args.imgsz,
                kelas_ke_nama,
                args.device,
            )
            jumlah, watt, kwh = gambarkan_hasil(frame, deteksi, args.hours)
            output_path = output_dir / f"{gambar.stem}_deteksi.jpg"
            if not cv2.imwrite(str(output_path), frame):
                raise OSError(f"Gagal menyimpan hasil deteksi: {output_path}")

            print(f"\n{gambar.name}:")
            if not jumlah:
                print("  Tidak ada perangkat elektronik yang dikenali.")
            for nama, count in jumlah.items():
                if nama not in WATT_PERANGKAT:
                    print(f"  {nama}: {count} (data daya belum tersedia)")
                    continue
                nilai_min, nilai_umum, nilai_max = WATT_PERANGKAT[nama]
                rata_rata = nilai_umum * FAKTOR_PEMAKAIAN[nama] * count
                print(
                    f"  {nama}: {count} "
                    f"(rentang {nilai_min * count:.0f}-{nilai_max * count:.0f} W; "
                    f"rata-rata pemakaian ~{rata_rata:.0f} W)"
                )
            ada_data_watt = any(
                nama in WATT_PERANGKAT and nama in FAKTOR_PEMAKAIAN
                for nama in jumlah
            )
            if ada_data_watt:
                print(
                    f"  Total perkiraan perangkat dengan data watt: "
                    f"{watt:.0f} W rata-rata / {kwh:.2f} kWh "
                    f"untuk {args.hours:g} jam"
                )
            elif jumlah:
                print("  Total daya tidak dihitung karena data watt belum tersedia.")
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

            deteksi = deteksi_elektronik(
                model,
                frame,
                args.confidence,
                args.imgsz,
                kelas_ke_nama,
                args.device,
            )
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
        ringkasan, _ = ringkasan_kamera(jumlah_maksimum, durasi_detik)
        waktu_selesai = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_ringkasan = output_dir / f"ringkasan_kamera_{waktu_selesai}.txt"
        file_ringkasan.write_text(
            "KESIMPULAN DETEKSI DAN PERKIRAAN DAYA\n"
            f"Dicatat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "========================================\n\n"
            f"{ringkasan}\n\n"
            "Catatan: Watt dan kWh merupakan perkiraan rata-rata karena model "
            "tidak mengetahui spesifikasi daya asli perangkat. Perangkat tanpa "
            "data watt tidak disertakan dalam total daya.\n",
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