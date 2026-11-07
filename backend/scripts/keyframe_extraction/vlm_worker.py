"""Notebook entrypoint. Imports the embedded canonical VLM pipeline, not a copy."""
import argparse
import json
from pathlib import Path
import sys
import traceback
import yaml


def process_job(config_path, models):
    config_path = Path(config_path)
    job = json.loads(config_path.read_text(encoding='utf-8'))
    if (job.get('pipeline_id') != 'vlm-p3-quality-v2'
            or 'mmr_max_similarity' in job
            or job.get('final_similarity_threshold') is None):
        raise ValueError('Notebook/source mismatch: use the latest vlm_keyframes.ipynb and rerun cell 4')
    sys.path.insert(0, job['source_root'])
    from src.utils.config import load_config
    from src.keyframe_extraction.pipeline.extract_keyframes import KeyframeExtractionPipeline
    import torch
    from src.utils.gpu_video import CudaVideoReader
    def probe_nvdec(video):
        # Check actual decode and GPU JPEG support before loading large models.
        with CudaVideoReader(video, 1) as probe:
            _, first_frame = next(iter(probe))
            from torchvision.io import encode_jpeg
            encoded = encode_jpeg([first_frame[0].contiguous()], quality=95)[0]
            if encoded.numel() == 0:
                raise RuntimeError('CUDA JPEG smoke test returned no data')
            print(f'NVDEC verified: {tuple(first_frame.shape)} on {first_frame.device}', flush=True)
            print('CUDA JPEG verified.', flush=True)
        del first_frame, encoded
    from src.keyframe_extraction.nvdec_input import prepare_nvdec_input
    job['video'] = str(prepare_nvdec_input(
        Path(job['video']), config_path.parent, probe_nvdec,
        compute_major=torch.cuda.get_device_capability(0)[0]))
    source = Path(job['source_root'])
    raw = yaml.safe_load((source/'configs/kf_extraction.yaml').read_text(encoding='utf-8'))
    raw['paths'] = dict(input_dir=str(Path(job['video']).parent), output_dir=job['output_dir'])
    raw['logging'].update(log_to_file=False)
    raw['omni'].update(weights_path=job.get('omni_weights'), cache_dir=job['weights_cache'],
                       decode_batch_size=job.get('decode_batch_size',8))
    raw['embedding'].update(batch_size=job.get('batch_size',16),sample_fps=job.get('sample_fps',3.))
    raw['keyframe'].update(final_similarity_threshold=job.get('final_similarity_threshold',.96),
                           min_hist_bins=job.get('min_hist_bins',5),
                           max_scene_seconds=job.get('max_scene_seconds',30))
    for section in ['omni','embedding','keyframe']:
        raw[section]['skip_existing'] = not job.get('overwrite',False)
    # Separate VLM workspace is intentional: never reuse former SurgVLP cache.
    config = config_path.with_suffix('.yaml')
    config.write_text(yaml.safe_dump(raw),encoding='utf-8')
    cfg = load_config(config)
    KeyframeExtractionPipeline(cfg, models=models).run(
        videos=[Path(job['video'])], group=Path(job['group']))


def serve(models):
    """One job at a time; acknowledge only after all extraction output is saved."""
    for line in sys.stdin:
        config_path = None
        try:
            config_path = json.loads(line)['config']
            process_job(config_path, models)
        except Exception as exc:
            traceback.print_exc()
            print('__VLM_RESULT__=' + json.dumps(dict(config=config_path, ok=False, error=str(exc))),
                  flush=True)
            return 1  # Never reuse a worker after a failed CUDA/model operation.
        print('__VLM_RESULT__=' + json.dumps(dict(config=config_path, ok=True)), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--config')
    mode.add_argument('--serve', action='store_true')
    parser.add_argument('--source-root')
    args = parser.parse_args()
    if args.config:
        source_root = json.loads(Path(args.config).read_text(encoding='utf-8'))['source_root']
    elif args.source_root:
        source_root = args.source_root
    else:
        parser.error('--serve requires --source-root')
    sys.path.insert(0, str(source_root))
    import torch
    from src.keyframe_extraction.model_cache import KeyframeModels
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    models = KeyframeModels()
    try:
        if args.serve:
            return serve(models)
        process_job(args.config, models)
        return 0
    finally:
        models.close()


if __name__ == '__main__':
    raise SystemExit(main())
