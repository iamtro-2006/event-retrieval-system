from __future__ import annotations

import json
import random
from pathlib import Path

import yaml


SUBJECTS = [
    "the reporter",
    "the driver",
    "the police officer",
    "the shop owner",
    "the tourist",
    "the student",
    "the doctor",
    "the mayor",
    "the athlete",
    "the chef",
    "the pilot",
    "the engineer",
]

VERBS = [
    "explains that",
    "says that",
    "announces that",
    "confirms that",
    "warns that",
    "reports that",
    "mentions that",
    "believes that",
    "reveals that",
    "adds that",
]

CLAUSES = [
    "the road will be closed for repairs",
    "the weather will get worse this afternoon",
    "the new store opens next week",
    "traffic is heavy near the city center",
    "the match will start in ten minutes",
    "the flight has been delayed",
    "the price of vegetables has increased",
    "the festival attracted a large crowd",
    "the bridge construction is almost finished",
    "the water level is rising after the rain",
    "the exhibition will run until next month",
    "the team is preparing for the final round",
    "the hospital is expanding its emergency ward",
    "the government announced a new policy today",
    "residents are asked to stay indoors",
    "the fire has been brought under control",
    "the market is busy this morning",
    "tickets are now available online",
    "the ceremony will begin shortly",
    "the situation is being closely monitored",
]

FILLERS = [
    "um",
    "so",
    "well",
    "you know",
    "actually",
    "basically",
    "in fact",
    "as we can see",
]


def load_config():

    with open("configs/asr.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def random_sentence() -> str:

    parts = []

    if random.random() < 0.3:
        parts.append(random.choice(FILLERS) + ",")

    parts.append(random.choice(SUBJECTS))
    parts.append(random.choice(VERBS))
    parts.append(random.choice(CLAUSES))

    sentence = " ".join(parts)

    return sentence[0].upper() + sentence[1:] + "."


def generate_video_json(
    output_file: Path,
    video_duration_sec: float,
    min_segment_duration_sec: float,
    max_segment_duration_sec: float,
):

    segments = []

    cursor = 0.0
    idx = 0

    while cursor < video_duration_sec:

        duration = random.uniform(min_segment_duration_sec, max_segment_duration_sec)

        start_time = round(cursor, 2)
        end_time = round(min(cursor + duration, video_duration_sec), 2)

        if end_time <= start_time:
            break

        num_sentences = random.randint(1, 2)

        text = " ".join(random_sentence() for _ in range(num_sentences))

        segments.append(
            {
                "segment_id": f"{idx:06d}",
                "start_time": start_time,
                "end_time": end_time,
                "text": text,
            }
        )

        cursor = end_time
        idx += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            segments,
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

    video_duration_sec = cfg["mock"]["video_duration_sec"]

    min_segment_duration_sec = cfg["mock"]["min_segment_duration_sec"]

    max_segment_duration_sec = cfg["mock"]["max_segment_duration_sec"]

    for dataset in datasets:

        dataset_dir = root / dataset

        for i in range(1, videos_per_dataset + 1):

            video_id = f"{dataset}_V{i:03d}"

            output_file = dataset_dir / f"{video_id}.json"

            generate_video_json(
                output_file=output_file,
                video_duration_sec=video_duration_sec,
                min_segment_duration_sec=min_segment_duration_sec,
                max_segment_duration_sec=max_segment_duration_sec,
            )

            print(f"Generated {output_file}")


if __name__ == "__main__":
    main()
