from pathlib import Path

root = Path(r"data\processed\keyframes")

for img in root.rglob("*.jpg"):
    stem = img.stem

    if stem.isdigit() and len(stem) < 6:
        new_path = img.with_name(f"{int(stem):06d}{img.suffix}")
        img.rename(new_path)
        print(img, "->", new_path)