from __future__ import annotations

import asyncio
import queue
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd

from voice_assistant.tts.queue import AudioChunk


@dataclass(slots=True)
class PlaybackState:
    interrupted: bool = False


class AudioPlayer:
    def __init__(self, sample_rate: int = 22_050, blocksize: int = 128) -> None:
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._pending = np.array([], dtype=np.float32)
        self.state = PlaybackState()
        self._active = False

    def _callback(self, outdata: Any, frames: int, _time: Any, _status: sd.CallbackFlags) -> None:
        if self.state.interrupted:
            outdata.fill(0)
            return

        if len(self._pending) < frames:
            parts = [self._pending]
            needed = frames - len(self._pending)
            while needed > 0:
                try:
                    nxt = self._queue.get_nowait()
                    parts.append(nxt)
                    needed -= len(nxt)
                except queue.Empty:
                    break
            if parts:
                self._pending = np.concatenate(parts)

        if len(self._pending) == 0:
            outdata.fill(0)
            return

        out = np.zeros((frames,), dtype=np.float32)
        take = min(frames, len(self._pending))
        out[:take] = self._pending[:take]
        self._pending = self._pending[take:]
        outdata[:, 0] = out

    async def start(self) -> None:
        if self._active:
            return
        self._active = True
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            blocksize=self.blocksize,
        )
        self._stream.start()

    async def play(self, chunk: AudioChunk) -> None:
        audio = np.frombuffer(chunk.pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        while True:
            try:
                self._queue.put_nowait(audio)
                break
            except queue.Full:
                await asyncio.sleep(0.005)

    async def stop(self) -> None:
        self._active = False
        if hasattr(self, "_stream"):
            self._stream.stop()
            self._stream.close()

    def interrupt(self) -> None:
        self.state.interrupted = True
        self._pending = np.array([], dtype=np.float32)
        while not self._queue.empty():
            self._queue.get_nowait()

    def resume(self) -> None:
        self.state.interrupted = False
