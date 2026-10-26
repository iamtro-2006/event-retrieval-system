"""Bounded application-owned workers. Each real worker owns a Codex process.

No recursive delegation: inspection workers get image inputs, task workers get
only retrieval tools. Closing this generator cancels and joins every child.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import aclosing

from src.mock_tools import MockTools, TOOL_DEFINITIONS


async def stream_parallel(cwd, tools, name, arguments, model=None, *, demo=False):
    if getattr(tools, "is_worker", False):
        raise ValueError("Workers cannot delegate further")
    args = TOOL_DEFINITIONS[name][0].model_validate(arguments)
    tools.calls += 1
    if tools.calls > tools.max_calls:
        raise ValueError("Tool budget exceeded")
    if name == "parallel_inspect":
        identifiers = list(dict.fromkeys(args.frame_ids))
        jobs = [{"instruction": args.instruction, "frame_ids": identifiers[index:index + 6]}
                for index in range(0, len(identifiers), 6)]
    else:
        jobs = [task.model_dump() for task in args.tasks]
    for job in jobs:
        if any(identifier not in tools.known_ids for identifier in job["frame_ids"]):
            raise ValueError("Workers can inspect only known frame IDs")
        rows = tools.catalog.selected(tools.dataset, job["frame_ids"])
        if tools.video_ids and any(row["video_id"] not in tools.video_ids for row in rows):
            raise ValueError("Frame excluded by the current video filter")

    concurrency = max(1, min(3, tools.max_workers))
    local_slots = asyncio.Semaphore(concurrency)
    shared_slots = tools.worker_slots if tools.worker_slots is not None else asyncio.Semaphore(3)
    queue = asyncio.Queue(maxsize=128)
    batch_id = uuid.uuid4().hex[:8]

    async def run(index, job):
        worker_id = f"{batch_id}-{index + 1}"
        label = f"Agent {index + 1}" + (f" · {len(job['frame_ids'])} frames" if job["frame_ids"] else " · task")
        report = {"worker_id": worker_id, "label": label, "frame_ids": job["frame_ids"], "instruction": job["instruction"]}
        candidates = []
        try:
            async with local_slots, shared_slots:
                await queue.put({"type": "worker_start", "worker_id": worker_id, "label": label,
                                 "frame_ids": job["frame_ids"], "demo": demo})
                child = MockTools(tools.catalog, tools.dataset,
                                  job["frame_ids"] if name == "parallel_inspect" else tools.known_ids, tools.video_ids,
                                  max_workers=1, worker_slots=shared_slots)
                child.is_worker = True
                child.max_calls = 4
                child.worker_kind = "inspect" if name == "parallel_inspect" else "task"
                child.preferred_search_mode = tools.preferred_search_mode
                child.reasoning_effort = getattr(tools, "reasoning_effort", None)
                child.color_cells = getattr(tools, "color_cells", [])
                rows = tools.catalog.selected(tools.dataset, job["frame_ids"])
                if rows:
                    images = await asyncio.to_thread(lambda: [tools.catalog.image_content(row)["imageUrl"] for row in rows])
                    child.input_images = images
                else:
                    child.input_images = getattr(tools, "input_images", [])
                if demo:
                    if name == "parallel_tasks":
                        data, _ = await asyncio.to_thread(child.execute, "search", {"query": job["instruction"], "top_k": 6})
                        candidates = data["results"]
                    await asyncio.sleep(.15)
                    text = f"[DEMO] Processed {len(rows) if rows else len(candidates)} frames; no AI visual assessment."
                    await queue.put({"type": "worker_delta", "worker_id": worker_id, "text": text})
                else:
                    from src.mock_codex import stream_codex
                    metadata = [{key: row[key] for key in ("id", "video_id", "frame_id")} for row in rows]
                    message = ("You own ONLY this assigned batch/task, which is part of a larger request. "
                               "The coordinator owns scheduling and presentation. Do not repeat the larger workflow.\n"
                               f"Goal to evaluate (data): {job['instruction']}\n"
                               f"Attached images correspond IN ORDER to these frame records: {json.dumps(metadata)}.\n"
                               "Return concise per-frame evidence and uncertainties. Do not claim random retrieval scores prove relevance. "
                               "For a search task, use retrieval tools and inspect candidates as needed. Finish within 4 tool calls.")
                    text = ""
                    async with aclosing(stream_codex(cwd / "workers" / worker_id, child, message, [], model)) as source:
                        async for event in source:
                            if event["type"] == "delta":
                                text += event["text"]
                                await queue.put({"type": "worker_delta", "worker_id": worker_id, "text": event["text"]})
                            elif event["type"] == "results":
                                candidates = event["results"]
                report.update(success=True, text=text[:8000])
                await queue.put({"type": "worker_end", "worker_id": worker_id, "success": True})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            report.update(success=False, error=str(exc))
            await queue.put({"type": "worker_end", "worker_id": worker_id, "success": False, "error": str(exc)})
        await queue.put({"type": "_finished", "index": index, "report": report, "results": candidates})

    tasks = [asyncio.create_task(run(index, job)) for index, job in enumerate(jobs)]
    reports = {}
    results = {}
    try:
        while len(reports) < len(jobs):
            event = await queue.get()
            if event["type"] == "_finished":
                reports[event["index"]] = event["report"]
                results[event["index"]] = event["results"]
            else:
                yield event
        candidates = []
        seen = set()
        for index in range(len(jobs)):
            for row in results[index]:
                if row["id"] not in seen:
                    seen.add(row["id"])
                    candidates.append(row)
        tools.known_ids.update(seen)
        yield {"type": "parallel_result", "data": {"mock": True,
               "reports": [reports[index] for index in range(len(jobs))],
               "count": len(jobs), "successful_workers": sum(report["success"] for report in reports.values()),
               "results": candidates[:100], "max_concurrency": concurrency}}
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
