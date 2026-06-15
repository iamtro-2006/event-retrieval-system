from pathlib import Path

root = Path(r"data\processed\keyframes")

for img in root.rglob("*.jpg"):
    stem = img.stem

    if stem.isdigit() and len(stem) < 6:
        new_path = img.with_name(f"{int(stem):06d}{img.suffix}")
        
        # Check if the target filename is already taken
        if new_path.exists():
            print(f"[SKIPPED] Target already exists: {new_path} (Source: {img})")
            continue
            
        img.rename(new_path)
        print(img, "->", new_path)