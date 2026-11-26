import csv
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "datasets" / "open_images_v6"
OUTPUT_DIR = SOURCE_DIR / "yolo"
ANNOTATIONS_PATH = SOURCE_DIR / "oidv6-train-annotations-bbox.csv"
CLASS_DESCRIPTIONS_PATH = SOURCE_DIR / "class-descriptions-boxable.csv"

ANNOTATIONS_URL = (
    "https://storage.googleapis.com/openimages/v6/"
    "oidv6-train-annotations-bbox.csv"
)
CLASS_DESCRIPTIONS_URL = (
    "https://storage.googleapis.com/openimages/v5/"
    "class-descriptions-boxable.csv"
)
IMAGE_METADATA_URL = (
    "https://storage.googleapis.com/openimages/2018_04/train/"
    "train-images-boxable-with-rotation.csv"
)
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg"

CLASSES = [
    "Blender",
    "Camera",
    "Fan",
    "Headphones",
    "Keyboard",
    "Laptop",
    "Microwave oven",
    "Mixer",
    "Mobile phone",
    "Monitor",
    "Mouse",
    "Printer",
    "Refrigerator",
    "TV",
    "Tablet",
    "Toaster",
    "Washing machine",
]

SOURCE_TO_CLASS = {
    "Washing machine": "Washing machine",
    "Headphones": "Headphones",
    "Laptop": "Laptop",
    "Toaster": "Toaster",
    "Computer keyboard": "Keyboard",
    "Printer": "Printer",
    "Computer mouse": "Mouse",
    "Computer monitor": "Monitor",
    "Blender": "Blender",
    "Mechanical fan": "Fan",
    "Ceiling fan": "Fan",
    "Refrigerator": "Refrigerator",
    "Mixer": "Mixer",
    "Television": "TV",
    "Tablet computer": "Tablet",
    "Camera": "Camera",
    "Microwave oven": "Microwave oven",
    "Mobile phone": "Mobile phone",
}

ATTRIBUTION_FIELDS = [
    "ImageID",
    "OriginalURL",
    "OriginalLandingURL",
    "License",
    "AuthorProfileURL",
    "Author",
    "Title",
    "Rotation",
]


def download_small_file(url: str, destination: Path) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)


def read_label_ids() -> dict[str, int]:
    label_ids = {}
    with CLASS_DESCRIPTIONS_PATH.open(
        newline="", encoding="utf-8"
    ) as class_file:
        for label_id, display_name in csv.reader(class_file):
            if display_name in SOURCE_TO_CLASS:
                target_name = SOURCE_TO_CLASS[display_name]
                label_ids[label_id] = CLASSES.index(target_name)
    if len(label_ids) != len(SOURCE_TO_CLASS):
        raise ValueError(
            f"Matched {len(label_ids)} of {len(SOURCE_TO_CLASS)} source labels"
        )
    return label_ids


def split_for_image(image_id: str) -> str:
    value = int(hashlib.sha1(image_id.encode("ascii")).hexdigest()[:8], 16)
    return "val" if value % 10 == 0 else "train"


def collect_annotations(label_ids: dict[str, int]) -> dict[str, list[str]]:
    annotations: dict[str, list[str]] = {}
    with ANNOTATIONS_PATH.open(
        newline="", encoding="utf-8"
    ) as annotation_file:
        for row in csv.DictReader(annotation_file):
            class_id = label_ids.get(row["LabelName"])
            if class_id is None or row["IsGroupOf"] != "0":
                continue

            x_min = float(row["XMin"])
            x_max = float(row["XMax"])
            y_min = float(row["YMin"])
            y_max = float(row["YMax"])
            if not (0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1):
                continue

            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            line = (
                f"{class_id} {center_x:.6f} {center_y:.6f} "
                f"{width:.6f} {height:.6f}"
            )
            annotations.setdefault(row["ImageID"], []).append(line)
    return annotations


def write_dataset_files(annotations: dict[str, list[str]]) -> dict[str, str]:
    image_splits = {}
    for image_id, labels in annotations.items():
        split = split_for_image(image_id)
        image_splits[image_id] = split
        label_path = OUTPUT_DIR / "labels" / split / f"{image_id}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASSES))
    data_yaml = (
        f'path: "{OUTPUT_DIR.resolve().as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names:\n{names}\n"
    )
    (OUTPUT_DIR / "data.yaml").write_text(data_yaml, encoding="utf-8")

    ids_path = SOURCE_DIR / "selected-train-image-ids.txt"
    ids_path.write_text(
        "".join(f"train/{image_id}\n" for image_id in sorted(image_splits)),
        encoding="utf-8",
    )
    return image_splits


def download_image(image_id: str, split: str) -> tuple[str, str | None]:
    image_path = OUTPUT_DIR / "images" / split / f"{image_id}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if image_path.is_file() and image_path.stat().st_size >= 1024:
        with image_path.open("rb") as image_file:
            if image_file.read(2) == b"\xff\xd8":
                return image_id, None

    url = IMAGE_URL.format(image_id=image_id)

    for attempt in range(4):
        try:
            response = requests.get(url, timeout=(15, 60))
            response.raise_for_status()
            content = response.content
            if len(content) < 1024 or not content.startswith(b"\xff\xd8"):
                raise ValueError("Response is not a valid JPEG image")
            temporary_path = image_path.with_suffix(".jpg.part")
            temporary_path.write_bytes(content)
            temporary_path.replace(image_path)
            return image_id, None
        except (requests.RequestException, OSError, ValueError) as error:
            if attempt == 3:
                return image_id, str(error)
            time.sleep(2**attempt)
    return image_id, "Download failed after retries"


def download_images(image_splits: dict[str, str]) -> list[tuple[str, str]]:
    failures = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(download_image, image_id, split): image_id
            for image_id, split in image_splits.items()
        }
        for index, future in enumerate(as_completed(futures), start=1):
            image_id, error = future.result()
            if error is not None:
                failures.append((image_id, error))
                label_path = (
                    OUTPUT_DIR
                    / "labels"
                    / image_splits[image_id]
                    / f"{image_id}.txt"
                )
                label_path.unlink(missing_ok=True)
            if index % 500 == 0:
                print(f"Downloaded {index:,}/{len(futures):,} images")
    return failures


def write_attributions(image_ids: set[str]) -> tuple[dict[str, int], int]:
    attribution_path = OUTPUT_DIR / "image_attributions.csv"
    license_counts: dict[str, int] = {}
    response = requests.get(IMAGE_METADATA_URL, stream=True, timeout=(15, 120))
    response.raise_for_status()
    response.encoding = "utf-8"
    metadata_rows = 0

    with attribution_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=ATTRIBUTION_FIELDS)
        writer.writeheader()
        try:
            for row in csv.DictReader(response.iter_lines(decode_unicode=True)):
                if row["ImageID"] not in image_ids:
                    continue
                writer.writerow(
                    {field: row.get(field, "") for field in ATTRIBUTION_FIELDS}
                )
                license_name = row.get("License", "unknown")
                license_counts[license_name] = (
                    license_counts.get(license_name, 0) + 1
                )
                metadata_rows += 1
        finally:
            response.close()
    return license_counts, metadata_rows


def write_summary(
    annotations: dict[str, list[str]],
    image_splits: dict[str, str],
    failures: list[tuple[str, str]],
    license_counts: dict[str, int],
    metadata_rows: int,
) -> None:
    valid_ids = set(image_splits) - {image_id for image_id, _ in failures}
    counts = {name: {"train": 0, "val": 0} for name in CLASSES}
    for image_id in valid_ids:
        split = image_splits[image_id]
        for line in annotations[image_id]:
            class_id = int(line.split(" ", 1)[0])
            counts[CLASSES[class_id]][split] += 1

    with (OUTPUT_DIR / "class_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["class", "train_boxes", "val_boxes"])
        for name in CLASSES:
            writer.writerow([name, counts[name]["train"], counts[name]["val"]])

    if failures:
        with (SOURCE_DIR / "failed-image-downloads.csv").open(
            "w", newline="", encoding="utf-8"
        ) as failed_file:
            writer = csv.writer(failed_file)
            writer.writerow(["ImageID", "Error"])
            writer.writerows(failures)
    else:
        (SOURCE_DIR / "failed-image-downloads.csv").unlink(missing_ok=True)

    readme = [
        "# Open Images V6 appliance/electronics subset",
        "",
        "This YOLO-format subset was generated from Open Images V6 train bounding boxes.",
        "Images are downloaded from the official Open Images CVDF S3 bucket.",
        "The original image license and attribution fields are recorded in",
        "`image_attributions.csv`; check those per-image terms before redistribution",
        "or commercial use.",
        "",
        "Classes included: " + ", ".join(CLASSES) + ".",
        "Open Images has no air-conditioner box class, and this subset does not",
        "cover all appliance categories in the project.",
        "",
        "Source: https://storage.googleapis.com/openimages/web/download_v6.html",
        "Annotations: " + ANNOTATIONS_URL,
        "Image source: https://github.com/cvdfoundation/open-images-dataset",
        "",
        f"Train images: {sum(split == 'train' for image_id, split in image_splits.items() if image_id in valid_ids)}",
        f"Validation images: {sum(split == 'val' for image_id, split in image_splits.items() if image_id in valid_ids)}",
        f"Failed image downloads: {len(failures)}",
        f"Images with attribution metadata: {metadata_rows}",
        "",
        "Per-class box counts are in `class_counts.csv`.",
        "The train/validation split is a deterministic 90/10 split by image ID.",
        "",
        "License counts for downloaded images:",
    ]
    readme.extend(f"- {license_name}: {count}" for license_name, count in sorted(license_counts.items()))
    (OUTPUT_DIR / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print("Dataset summary:")
    print(f"  images: {len(valid_ids):,}")
    print(f"  train: {sum(split == 'train' for image_id, split in image_splits.items() if image_id in valid_ids):,}")
    print(f"  val: {sum(split == 'val' for image_id, split in image_splits.items() if image_id in valid_ids):,}")
    print(f"  failed downloads: {len(failures):,}")
    print(f"  attribution rows: {metadata_rows:,}")
    print("  class box counts:")
    for name in CLASSES:
        print(f"    {name}: train={counts[name]['train']:,}, val={counts[name]['val']:,}")
    print("  license counts:")
    for license_name, count in sorted(license_counts.items()):
        print(f"    {license_name}: {count:,}")


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CLASS_DESCRIPTIONS_PATH.exists():
        download_small_file(CLASS_DESCRIPTIONS_URL, CLASS_DESCRIPTIONS_PATH)
    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(
            "Download the Open Images V6 train bounding-box CSV first: "
            f"{ANNOTATIONS_PATH}"
        )

    label_ids = read_label_ids()
    print(f"Filtering {len(label_ids)} Open Images source labels")
    annotations = collect_annotations(label_ids)
    if not annotations:
        raise ValueError("No matching bounding boxes were found")

    image_splits = write_dataset_files(annotations)
    print(f"Downloading {len(image_splits):,} matching images")
    failures = download_images(image_splits)
    successful_ids = set(image_splits) - {image_id for image_id, _ in failures}
    license_counts, metadata_rows = write_attributions(successful_ids)
    if metadata_rows != len(successful_ids):
        raise ValueError(
            f"Found attribution metadata for {metadata_rows} of "
            f"{len(successful_ids)} downloaded images"
        )
    write_summary(
        annotations, image_splits, failures, license_counts, metadata_rows
    )


if __name__ == "__main__":
    main()
