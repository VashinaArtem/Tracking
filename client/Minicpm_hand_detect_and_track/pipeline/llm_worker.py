"""
LLM detection worker.

Runs ServerClient.detect() in a dedicated thread.
Enforces a single in-flight request via a semaphore slot.
"""

from __future__ import annotations

import threading
import logging
from queue import Queue, Empty, Full
from typing import Optional

from network.server_client import ServerClient
from pipeline.messages import LLMTask, LLMResult

logger = logging.getLogger(__name__)


def _put_overwrite(q: Queue, item) -> None:
    """Put item into a size-1 queue, overwriting old value if full."""
    try:
        q.put(item, block=False)
        return
    except Full:
        try:
            q.get(block=False)
        except Empty:
            pass
        q.put(item, block=False)


class LLMWorker(threading.Thread):
    def __init__(
        self,
        *,
        stop_event: threading.Event,
        server_client: ServerClient,
        task_queue: Queue,
        result_queue: Queue,
        llm_slot: threading.BoundedSemaphore,
    ):
        super().__init__(daemon=True)
        self._stop_event = stop_event
        self._server_client = server_client
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._llm_slot = llm_slot

    def run(self) -> None:
        while not self._stop_event.is_set():
            task: Optional[LLMTask] = None
            try:
                task = self._task_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                bbox, inference_time, network_time = self._server_client.detect(task.frame, task.prompt)
                ok = bbox is not None
                res = LLMResult(
                    ok=ok,
                    prompt=task.prompt,
                    ts=task.ts,
                    frame_id=task.frame_id,
                    bbox=bbox,
                    frame=task.frame,
                    inference_time=inference_time,
                    network_time=network_time,
                    error="",
                )
                _put_overwrite(self._result_queue, res)
            except Exception as e:
                logger.exception("LLMWorker detect failed: %s", e)
                res = LLMResult(
                    ok=False,
                    prompt=task.prompt if task else "",
                    ts=task.ts if task else 0.0,
                    frame_id=task.frame_id if task else -1,
                    bbox=None,
                    inference_time=0.0,
                    network_time=0.0,
                    error=str(e),
                )
                _put_overwrite(self._result_queue, res)
            finally:
                # Release the single-inflight slot after the request finishes.
                try:
                    self._llm_slot.release()
                except ValueError:
                    # Shouldn't happen, but keep worker alive.
                    pass

