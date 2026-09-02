import os
import supervision as sv
import requests, zipfile, os, shutil, random
import cv2
from PIL import Image
from tqdm import tqdm
import time
from dotenv import load_dotenv

load_dotenv()

# ------------------ CONFIG ------------------ #
classes = ["car", "truck", "bike", "accident"]

output_dir = "auto_dataset"
images_dir = f"{output_dir}/images"
labels_dir = f"{output_dir}/labels"

os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Hugging Face access token
HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set. Add it to your environment or .env file.")

datasets = {
    "car":      "https://huggingface.co/datasets/anujgarg09/car-small-yolo/resolve/main/car_small.zip",
    "truck":    "https://huggingface.co/datasets/anujgarg09/truck-small-yolo/resolve/main/truck_small.zip",
    "bike":     "https://huggingface.co/datasets/anujgarg09/bike-small-yolo/resolve/main/bike_small.zip",
    "accident": "https://huggingface.co/datasets/anujgarg09/accident-yolo/resolve/main/accident_small.zip"
}

# ------------------ DOWNLOAD ZIP WITH TOKEN ------------------ #
def download_zip(url, out_file, retries=3):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    for attempt in range(retries):
        try:
            print(f"📥 Downloading: {out_file} (Attempt {attempt+1})")
            r = requests.get(url, headers=headers, stream=True)
            if r.status_code != 200:
                raise Exception(f"Status code {r.status_code}")
            with open(out_file, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            if zipfile.is_zipfile(out_file):
                print("✔ Download and verification complete.")
                return
            else:
                print("❌ Invalid ZIP file, retrying...")
        except Exception as e:
            print("❌ Error:", e)
        time.sleep(2)
    raise Exception(f"Failed to download a valid zip for {out_file}")

# ------------------ IMAGE COMPRESSION ------------------ #
def compress_image(in_path, out_path, size_kb=120):
    img = Image.open(in_path)
    quality = 85
    while True:
        img.save(out_path, optimize=True, quality=quality)
        if os.path.getsize(out_path)/1024 <= size_kb or quality <= 20:
            break
        quality -= 5

# ------------------ PROCESS DATASETS ------------------ #
counter = 0

for cname, url in datasets.items():
    zip_name = f"{cname}.zip"
    extract_dir = f"tmp_{cname}"

    # Download if not exists or invalid
    if not os.path.exists(zip_name) or not zipfile.is_zipfile(zip_name):
        download_zip(url, zip_name)

    # Extract ZIP
    print(f"📦 Extracting {zip_name} ...")
    with zipfile.ZipFile(zip_name, 'r') as z:
        z.extractall(extract_dir)

    img_dir = f"{extract_dir}/images"
    label_dir = f"{extract_dir}/labels"

    image_files = os.listdir(img_dir)
    print(f"🔧 Preparing {cname} dataset...")

    for img_file in tqdm(image_files):
        src_img = f"{img_dir}/{img_file}"
        out_img = f"{images_dir}/{counter}.jpg"
        out_label = f"{labels_dir}/{counter}.txt"

        # Save compressed image
        img = cv2.imread(src_img)
        cv2.imwrite(out_img, img)
        compress_image(out_img, out_img, size_kb=120)

        # Copy label
        src_label = f"{label_dir}/{img_file.replace('.jpg', '.txt')}"
        if os.path.exists(src_label):
            new_lines = []
            with open(src_label) as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    parts[0] = str(classes.index(cname))
                    new_lines.append(" ".join(parts))
            with open(out_label, "w") as f:
                f.write("\n".join(new_lines))

        counter += 1

    # Cleanup extracted directory
    shutil.rmtree(extract_dir)

# ------------------ SPLIT TRAIN/VAL ------------------ #
print("\n📚 Splitting dataset...")
paths = list(range(counter))
random.shuffle(paths)
split = int(0.8 * counter)
train_idx = paths[:split]
val_idx = paths[split:]

for mode, idx_list in [("train", train_idx), ("val", val_idx)]:
    os.makedirs(f"{output_dir}/{mode}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/{mode}/labels", exist_ok=True)
    for i in idx_list:
        shutil.copy(f"{images_dir}/{i}.jpg", f"{output_dir}/{mode}/images/")
        shutil.copy(f"{labels_dir}/{i}.txt", f"{output_dir}/{mode}/labels/")

# ------------------ CREATE data.yaml ------------------ #
yaml = f"""
train: {output_dir}/train/images
val: {output_dir}/val/images

nc: {len(classes)}
names: {classes}
"""

open(f"{output_dir}/data.yaml", "w").write(yaml)

print("\n\n🎉 ALL DONE!")
print(f"Total final images: {counter}")
print("Dataset path:", output_dir)
