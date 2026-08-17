"""Diagnostic script cho bug 'timestamp_sec toan 0.0 + duration_limit khong
an hien' trong temporal search.

Chay o thu muc backend/ (cung cap voi main.py), vi du:

    python debug_temporal.py
    python debug_temporal.py --query "nguoi dan ong mac ao do, sau do mo cua xe" --duration 15
    python debug_temporal.py --config configs/app.yaml --video-id L21_V001

Script nay dung DUNG `build_system()` / `Orchestrator` / `FaissIndex` that
(khong mock gi ca) de tai hien chinh xac nhung gi API dang chay, roi in log
chi tiet qua tung tang:

  [1] METADATA GOC   - metadata.csv sau khi build index: bao nhieu dong
                        timestamp_sec la null/0/hop le, vai dong mau, va
                        cot `map_path` (duoc bake san luc build) de biet
                        CHINH XAC index_builder da doc map CSV nao cho tung
                        dong - so sanh voi map CSV that tren dia.
  [2] CANDIDATE BUILD - candidate_df tra ve tu build_temporal_candidates:
                        co con timestamp_sec khong, co bao nhieu candidate.
  [3] DP ALIGNMENT     - ket qua tho tu SearchPipeline.search(): cac cot
                        temporal_start_time/temporal_end_time/temporal_duration_sec.
  [4] DURATION FILTER  - chay lai voi 3 gia tri duration_limit khac nhau
                        (-1, nho, lon) de xem no co thuc su loc bot ket qua
                        khong (neu khong doi gi ca giua 3 lan chay -> dung
                        bug 'duration khong duoc nhan').
  [5] API RESPONSE     - dict_to_result_FAST() that su tra ve cho frontend,
                        in nguyen field "temporal" + "timestamp" cua top-1.

Gui toan bo output (stdout) lai cho minh phan tich.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def line(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_config(config_path: Path) -> dict:
    from src.api.legacy.paths import load_yaml
    return load_yaml(config_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--query", default="dogs; cats")
    parser.add_argument("--duration", type=float, default=10, help="duration_limit (giay) de test")
    parser.add_argument("--video-id", default=None, help="video_id cu the de soi metadata (vd: L21_V001)")
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    line(f"[0] LOAD CONFIG: {config_path}")
    try:
        config = load_config(config_path)
    except Exception:
        print("KHONG DOC DUOC CONFIG:")
        traceback.print_exc()
        return

    from src.retrieval.system import build_system

    try:
        system = build_system(config)
    except Exception:
        print("build_system() FAILED:")
        traceback.print_exc()
        return

    orchestrator = system.orchestrator
    index = orchestrator.index  # FaissIndex (model mac dinh, hoac model duy nhat)
    metadata: pd.DataFrame = index.metadata

    # ------------------------------------------------------------------
    # [1] METADATA GOC
    # ------------------------------------------------------------------
    line("[1] METADATA GOC (metadata.csv sau khi build index)")
    print(f"metadata_path = {index.metadata_path}")
    print(f"So dong       = {len(metadata)}")
    print(f"Cac cot       = {list(metadata.columns)}")

    if "timestamp_sec" not in metadata.columns:
        print("!!! metadata.csv KHONG CO cot 'timestamp_sec' luon.")
    else:
        ts = pd.to_numeric(metadata["timestamp_sec"], errors="coerce")
        n_null = int(ts.isna().sum())
        n_zero = int((ts.fillna(-1) == 0).sum())
        n_ok = int(((ts.notna()) & (ts != 0)).sum())
        print(f"timestamp_sec: null={n_null}  ==0={n_zero}  >0(hop_le)={n_ok}  / total={len(ts)}")
        if n_null + n_zero == len(ts):
            print(">>> XAC NHAN BUG: 100% timestamp_sec la null hoac 0. "
                  "Van de nam o BUILD-TIME (map CSV khong duoc doc dung), "
                  "khong phai o thuat toan DP.")

    if "fps" in metadata.columns:
        fps_null = int(pd.to_numeric(metadata["fps"], errors="coerce").isna().sum())
        print(f"fps: null={fps_null} / total={len(metadata)}")

    if "map_path" in metadata.columns:
        print("\n-- Doi chieu map_path (duong dan map CSV da duoc dung luc build) --")
        sample_paths = metadata["map_path"].dropna().unique()[:5]
        for p in sample_paths:
            p_path = Path(str(p))
            exists = p_path.exists()
            print(f"  {p}  -> exists={exists}")
            if exists:
                try:
                    df_map = pd.read_csv(p_path, nrows=3)
                    print(f"      cols={list(df_map.columns)}  sample_row0={df_map.iloc[0].to_dict() if len(df_map) else None}")
                except Exception as e:
                    print(f"      (khong doc duoc: {e})")
    else:
        print("!!! metadata.csv khong co cot 'map_path' -> ban dang dung metadata.csv CU "
              "(build truoc khi co cot nay), hoac index_builder.py da bi sua khac ban minh vua xem.")

    print("\n-- Vai dong metadata mau (dataset/video_id/keyframe_id/frame_idx/timestamp_sec/fps) --")
    cols_show = [c for c in ["dataset", "video_id", "keyframe_id", "frame_idx", "timestamp_sec", "fps", "map_path"] if c in metadata.columns]
    if args.video_id:
        sub = metadata[metadata["video_id"].astype(str) == str(args.video_id)]
        print(f"(loc theo video_id={args.video_id}, tim thay {len(sub)} dong)")
        print(sub[cols_show].head(10).to_string())
    else:
        print(metadata[cols_show].head(10).to_string())

    # ------------------------------------------------------------------
    # [2]+[3] CHAY THAT: build_query_plan -> temporal_search.search
    # ------------------------------------------------------------------
    line(f"[2]+[3] CHAY TEMPORAL SEARCH THAT: query={args.query!r} duration_limit={args.duration}")
    try:
        plan = orchestrator.build_query_plan(args.query, mode="temporal", use_split=True)
        print(f"events = {plan.events}")

        df = orchestrator.temporal_search.search(
            plan.events, top_k=args.top_k, candidate_k=max(200, args.top_k * 10),
            duration_limit=args.duration, model_key=args.model_key,
        )
        print(f"So ket qua tra ve: {len(df)}")
        if df.empty:
            print("!!! DataFrame RONG - khong co video nao du frame lien tuc cho tat ca sub-query.")
        else:
            show_cols = [c for c in [
                "video_id", "rank", "video_score", "avg_score",
                "temporal_start_time", "temporal_end_time", "temporal_duration_sec",
                "timestamp_sec",
            ] if c in df.columns]
            print(df[show_cols].to_string())
    except Exception:
        print("temporal_search.search() FAILED:")
        traceback.print_exc()
        df = pd.DataFrame()

    # ------------------------------------------------------------------
    # [4] DURATION FILTER co thuc su loc khong?
    # ------------------------------------------------------------------
    line("[4] SO SANH duration_limit = -1 vs nho vs lon (co thay doi ket qua khong?)")
    for dlim in (-1, max(0.5, args.duration / 5), args.duration * 5):
        try:
            df_d = orchestrator.temporal_search.search(
                plan.events, top_k=args.top_k, candidate_k=max(200, args.top_k * 10),
                duration_limit=dlim, model_key=args.model_key,
            )
            durations = df_d["temporal_duration_sec"].tolist() if "temporal_duration_sec" in df_d.columns else []
            print(f"duration_limit={dlim!r:>10}  -> {len(df_d)} ket qua, "
                  f"temporal_duration_sec cua tung video = {durations}")
        except Exception as e:
            print(f"duration_limit={dlim!r:>10}  -> LOI: {e}")
    print(
        "\n(Neu ca 3 dong tren giong het nhau du duration_limit rat nho hay rat lon "
        "-> duration_limit dang bi vo hieu hoa, thuong la vi timestamp_sec = 0 het "
        "nhu o muc [1].)"
    )

    # ------------------------------------------------------------------
    # [5] API RESPONSE that su tra ve frontend (dict_to_result_FAST)
    # ------------------------------------------------------------------
    line("[5] API RESPONSE (dict_to_result_FAST) - dung field frontend dang doc")
    if not df.empty:
        try:
            from src.api.legacy.serializers import dict_to_result_FAST
            from src.api.legacy.paths import LegacyPaths

            paths = LegacyPaths(REPO_ROOT, config_path, config)
            top1 = df.iloc[0].to_dict()
            result = dict_to_result_FAST(top1, paths.keyframes_root, REPO_ROOT)
            print("Top-1 result['timestamp']    =", result.get("timestamp"))
            print("Top-1 result['temporal']     =", json.dumps(result.get("temporal"), indent=2, ensure_ascii=False))
            if result.get("matched_sequence"):
                print("\nmatched_sequence (frame trong chuoi) timestamp_sec tung frame:")
                for fr in result["matched_sequence"]:
                    print(f"  sub_query_idx={fr.get('sub_query_idx')}  video_id={fr.get('video_id')}  "
                          f"keyframe_id={fr.get('keyframe_id')}  timestamp_sec={fr.get('timestamp_sec')}")
        except Exception:
            print("dict_to_result_FAST() FAILED:")
            traceback.print_exc()
    else:
        print("(bo qua vi [2]/[3] khong co ket qua)")

    line("XONG - gui toan bo output nay lai de phan tich tiep")


if __name__ == "__main__":
    main()
