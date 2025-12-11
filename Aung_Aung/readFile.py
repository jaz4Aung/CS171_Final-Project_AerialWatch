import os
import csv

def build_csv(main_dir, output_csv="dataset_index.csv"):

    images_dir = os.path.join(main_dir, "images")
    labels_dir = os.path.join(main_dir, "labels")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    rows = []

    for img_name in sorted(os.listdir(images_dir)):
        root, ext = os.path.splitext(img_name)
        if ext.lower() not in image_exts:
            continue  # skip non-image files

        img_path = os.path.join("images", img_name)  # relative path

        label_name = root + ".txt"
        label_full_path = os.path.join(labels_dir, label_name)

        label_text = ""
        num_label_lines = 0
        has_person = 0

        if os.path.exists(label_full_path):
            with open(label_full_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            num_label_lines = len(lines)
            has_person = 1 if num_label_lines > 0 else 0
            # keep the raw text in one string (optional)
            label_text = " | ".join(lines)

        rows.append({
            "image_filename": img_name,
            "image_path": img_path,
            "label_filename": label_name if os.path.exists(label_full_path) else "",
            "num_label_lines": num_label_lines,   # how many boxes / persons
            "has_person": has_person,             # 1 = people present, 0 = none
            "label_text": label_text              # raw YOLO lines like: '0 xc yc w h'
        })

    # write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_filename",
            "image_path",
            "label_filename",
            "num_label_lines",
            "has_person",
            "label_text",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    # change this to the path of your "main files" folder
    build_csv(r"/Users/aungaung/Desktop/CS_171/Project/raw/valid")
