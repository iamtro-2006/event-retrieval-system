from __future__ import annotations

import json
import random
from pathlib import Path

import yaml


VOCABULARY = [

    # =========================
    # Traffic Signs
    # =========================
    "STOP",
    "GO",
    "ONE WAY",
    "NO ENTRY",
    "NO PARKING",
    "PARKING",
    "SPEED LIMIT 30",
    "SPEED LIMIT 40",
    "SPEED LIMIT 50",
    "SPEED LIMIT 60",
    "SPEED LIMIT 80",
    "SLOW",
    "YIELD",
    "PEDESTRIAN CROSSING",
    "SCHOOL ZONE",
    "BUS STOP",
    "TAXI",
    "EXIT",
    "ENTRANCE",
    "ROAD CLOSED",

    # =========================
    # Store Names
    # =========================
    "Circle K",
    "FamilyMart",
    "7-Eleven",
    "Highlands Coffee",
    "Phuc Long",
    "Starbucks",
    "Pizza Hut",
    "Lotteria",
    "KFC",
    "McDonald's",
    "The Coffee House",
    "WinMart",
    "Co.opmart",
    "Bach Hoa Xanh",
    "AEON Mall",

    # =========================
    # Restaurants
    # =========================
    "Pho 24",
    "Bun Bo Hue",
    "Com Tam",
    "Hu Tieu",
    "Mi Quang",
    "Banh Mi",
    "Coffee",
    "Restaurant",
    "Milk Tea",
    "Fast Food",

    # =========================
    # Building Labels
    # =========================
    "Hospital",
    "School",
    "University",
    "Police",
    "Airport",
    "Station",
    "Hotel",
    "Library",
    "Museum",
    "Bank",
    "ATM",

    # =========================
    # Common OCR
    # =========================
    "OPEN",
    "CLOSED",
    "SALE",
    "DISCOUNT",
    "WELCOME",
    "THANK YOU",
    "FREE WIFI",
    "CASHIER",
    "TOILET",
    "EMERGENCY EXIT",
    "PUSH",
    "PULL",

    # =========================
    # Brands
    # =========================
    "Samsung",
    "Apple",
    "Sony",
    "Canon",
    "Nikon",
    "Toyota",
    "Honda",
    "Yamaha",
    "Ford",
    "BMW",
    "Mercedes-Benz",
    "VinFast",

    # =========================
    # Vehicle Text
    # =========================
    "51A-12345",
    "59H-88888",
    "30F-56789",
    "29A-12345",
    "TAXI 38",
    "GRAB",
    "BE",
    "VINASUN",
    "MAI LINH",

    # =========================
    # Addresses
    # =========================
    "123 Nguyen Hue",
    "456 Le Loi",
    "789 Tran Hung Dao",
    "District 1",
    "District 3",
    "Thu Duc",
    "Ho Chi Minh City",
    "Ha Noi",
    "Da Nang",

    # =========================
    # Time
    # =========================
    "08:00",
    "09:30",
    "12:00",
    "15:45",
    "18:20",
    "21:00",

    # =========================
    # Dates
    # =========================
    "2025-01-01",
    "2025-12-31",
    "01/05/2025",
    "15/08/2025",
    "31/12/2025",

    # =========================
    # Warnings
    # =========================
    "DANGER",
    "WARNING",
    "CAUTION",
    "HIGH VOLTAGE",
    "NO SMOKING",
    "KEEP OUT",
    "AUTHORIZED PERSONNEL ONLY",

    # =========================
    # Public Information
    # =========================
    "CHECK IN",
    "CHECK OUT",
    "INFORMATION",
    "TICKET",
    "PLATFORM 1",
    "PLATFORM 2",
    "GATE A",
    "GATE B",
    "BOARDING",
    "ARRIVAL",
    "DEPARTURE",

    # =========================
    # Numbers
    # =========================
    "Room 101",
    "Room 202",
    "Floor 3",
    "Floor 10",
    "Table 15",
    "Counter 2",

    # =========================
    # Random OCR Noise
    # =========================
    "ABC123",
    "XYZ789",
    "INV-2025001",
    "Order #12345",
    "Receipt No. 8888",
    "Serial Number",
    "Model X100",
    "Version 2.0",
]

def load_config():

    with open("configs/ocr.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def random_sentence():

    length = random.randint(2, 6)

    return " ".join(random.sample(VOCABULARY, length))


def generate_video_json(
    output_file: Path,
    num_keyframes: int,
    min_texts: int,
    max_texts: int,
):

    data = {}

    for idx in range(num_keyframes):

        keyframe_id = f"{idx:06d}"

        num_texts = random.randint(min_texts, max_texts)

        texts = [
            random_sentence()
            for _ in range(num_texts)
        ]

        data[keyframe_id] = texts

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4,
        )


def main():

    cfg = load_config()

    random.seed(cfg["mock"]["seed"])

    root = Path(cfg["dataset"]["root"])

    datasets = cfg["mock"]["datasets"]

    videos_per_dataset = cfg["mock"]["videos_per_dataset"]

    keyframes = cfg["mock"]["keyframes_per_video"]

    min_texts = cfg["mock"]["min_texts_per_frame"]

    max_texts = cfg["mock"]["max_texts_per_frame"]

    for dataset in datasets:

        dataset_dir = root / dataset

        for i in range(1, videos_per_dataset + 1):

            video_id = f"{dataset}_V{i:03d}"

            output_file = dataset_dir / f"{video_id}.json"

            generate_video_json(
                output_file=output_file,
                num_keyframes=keyframes,
                min_texts=min_texts,
                max_texts=max_texts,
            )

            print(f"Generated {output_file}")


if __name__ == "__main__":
    main()