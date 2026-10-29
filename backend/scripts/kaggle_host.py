"""Foreground notebook supervisor: owned ES/model children, OCR seeding, FastAPI logs."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

log = logging.getLogger('ers.host')


async def serve(config):
    import httpx
    import uvicorn
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s', force=True)
    children = []
    backend = Path(config['backend']).resolve()
    os.chdir(backend)
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    os.environ.update(ES_HOST='127.0.0.1', ES_PORT='9200', ES_SCHEME='http',
                      CONFIG_PATH=str(config['app_config']), PYTHONUNBUFFERED='1')
    if config.get('build_segment_indexes'):
        os.environ['QWEN_EMBEDDING_ENGINE'] = 'vllm'

    def launch(label, command, env=None):
        process = subprocess.Popen(command, cwd=backend, env=env or os.environ.copy(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True)
        children.append((label, process))
        def forward():
            for line in process.stdout:
                log.info('[%s pid=%s] %s', label, process.pid, line.rstrip())
        threading.Thread(target=forward, daemon=True).start()
        return process

    async def ready(url, process, timeout=240):
        async with httpx.AsyncClient() as client:
            for _ in range(timeout):
                if process.poll() is not None:
                    raise RuntimeError(f'Service exited before readiness: {url}, code={process.returncode}')
                try:
                    response = await client.get(url, timeout=2)
                    if response.is_success:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1)
        raise TimeoutError(f'Service not ready: {url}')

    tunnel = None
    try:
        es_home = Path(config['es_home'])
        command = [str(es_home / 'bin/elasticsearch')]
        environment = os.environ.copy()
        environment['ES_JAVA_OPTS'] = '-Xms512m -Xmx512m'
        if os.getuid() == 0:
            import pwd
            try:
                pwd.getpwnam('erselastic')
            except KeyError:
                subprocess.run(['useradd', '--system', '--no-create-home', 'erselastic'], check=True)
            subprocess.run(['chown', '-R', 'erselastic', str(es_home)], check=True)
            command = ['runuser', '-u', 'erselastic', '--', 'env', 'ES_JAVA_OPTS=-Xms512m -Xmx512m', *command]
        es = launch('elasticsearch', command, environment)
        await ready('http://127.0.0.1:9200', es)
        if config.get('vllm_command'):
            model = launch('visual-encoder', config['vllm_command'])
            await ready(config.get('vllm_ready_url', 'http://127.0.0.1:8002/v1/models'), model, 600)
        if config.get('build_segment_indexes'):
            for dataset in config['datasets']:
                command = [
                    sys.executable, '-u', '-m', 'scripts.build_segment_data',
                    '--dataset', dataset,
                    '--data-root', str(config['data_root']),
                    '--model', config.get('segment_model', 'Qwen/Qwen3-VL-Embedding-2B'),
                    '--batch-size', str(config.get('segment_batch_size', 8)),
                    '--sources', *config.get('segment_sources', ['asr', 'caption']),
                ]
                if config.get('segment_embeddings_root'):
                    command += ['--embeddings-root', str(config['segment_embeddings_root'])]
                builder = launch(f'segment-index-{dataset}', command)
                while builder.poll() is None:
                    await asyncio.sleep(1)
                children.remove((f'segment-index-{dataset}', builder))
                if builder.returncode:
                    raise RuntimeError(f'Segment index build failed for {dataset}, code={builder.returncode}')
        # This is a full local OCR seed, explicitly recreate to avoid duplicated documents.
        for dataset in config['datasets']:
            seed = launch(f'ocr-seed-{dataset}', [sys.executable, '-u', '-m', 'scripts.ocr.index.run', '--dataset', dataset, '--recreate'])
            while seed.poll() is None:
                await asyncio.sleep(1)
            if seed.returncode:
                raise RuntimeError(f'OCR seed failed for {dataset}, code={seed.returncode}')
            children.remove((f'ocr-seed-{dataset}', seed))
        if config.get('ngrok_token'):
            from pyngrok import ngrok
            ngrok.set_auth_token(config['ngrok_token'])
            tunnel = ngrok.connect(config.get('port', 8000), 'http')
            log.info('API URL: %s (do not share beyond your trusted client)', tunnel.public_url)
        server = uvicorn.Server(uvicorn.Config('src.api.main:app', host='0.0.0.0',
            port=config.get('port', 8000), log_level='info', access_log=True))
        task = asyncio.create_task(server.serve())
        checked = False
        while not task.done():
            for label, process in children:
                if process.poll() is not None:
                    server.should_exit = True
                    await task
                    raise RuntimeError(f'{label} exited unexpectedly, code={process.returncode}; see logs above')
            # A visible heartbeat is diagnostics, not a way to bypass platform session limits.
            log.info('heartbeat API=%s children=%s', 'ready' if server.started else 'starting',
                     {label: process.pid for label, process in children})
            if server.started and not checked:
                checked = True
                async with httpx.AsyncClient() as client:
                    for dataset in config['datasets']:
                        response = await client.get(f"http://127.0.0.1:{config.get('port',8000)}/api/search/models",
                            params={'dataset': dataset}, timeout=30)
                        if not response.is_success:
                            log.error('Dataset %s NOT READY: %s', dataset, response.text)
                        else:
                            log.info('Dataset %s models: %s', dataset, response.text)
            await asyncio.sleep(15)
        await task
    finally:
        if 'server' in locals():
            server.should_exit = True
        if 'task' in locals() and not task.done():
            try:
                await asyncio.wait_for(task, 20)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()
        if tunnel:
            from pyngrok import ngrok
            ngrok.disconnect(tunnel.public_url)
        for label, process in reversed(children):
            if process.poll() is None:
                log.info('Stopping owned process %s pid=%s', label, process.pid)
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.to_thread(process.wait, 15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
