from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TurnMetrics:
    speech_end_ts: float | None = None
    final_text_ts: float | None = None
    prompt_sent_ts: float | None = None
    first_token_ts: float | None = None
    first_audio_ts: float | None = None
    tts_start_ts: float | None = None
    tts_end_ts: float | None = None
    synthesized_audio_sec: float = 0.0

    def as_dict(self) -> dict[str, float | None]:
        asr = self._delta(self.speech_end_ts, self.final_text_ts)
        ttft = self._delta(self.prompt_sent_ts, self.first_token_ts)
        tts_first = self._delta(self.first_token_ts, self.first_audio_ts)
        e2e = self._delta(self.speech_end_ts, self.first_audio_ts)
        tts_wall = self._delta(self.tts_start_ts, self.tts_end_ts)
        rtf = (tts_wall / self.synthesized_audio_sec) if tts_wall and self.synthesized_audio_sec else None
        return {
            "ASR_latency_ms": self._to_ms(asr),
            "TTFT_ms": self._to_ms(ttft),
            "TTS_first_chunk_ms": self._to_ms(tts_first),
            "E2E_ms": self._to_ms(e2e),
            "RTF": rtf,
        }

    @staticmethod
    def _delta(start: float | None, end: float | None) -> float | None:
        if start is None or end is None:
            return None
        return max(0.0, end - start)

    @staticmethod
    def _to_ms(value: float | None) -> float | None:
        if value is None:
            return None
        return round(value * 1000.0, 2)


@dataclass(slots=True)
class BenchmarkTracker:
    current: TurnMetrics = field(default_factory=TurnMetrics)

    @staticmethod
    def now() -> float:
        return time.perf_counter()

    def mark(self, name: str) -> None:
        setattr(self.current, name, self.now())

    def add_synthesized_audio(self, seconds: float) -> None:
        self.current.synthesized_audio_sec += max(0.0, seconds)

    def snapshot(self) -> dict[str, Any]:
        return self.current.as_dict()

    def reset(self) -> None:
        self.current = TurnMetrics()
