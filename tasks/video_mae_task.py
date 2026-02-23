"""
Video MAE captioning task for the WIZ Intelligence Pipeline.

Generates short natural-language descriptions for each ~5 s window of a video
and stores them as VideoCaption objects in the pipeline context.
"""

import gc
import cv2
import numpy as np
from typing import List

try:
    from ..core.base_task import BaseTask
    from ..core.context import PipelineContext, VideoCaption
    from ..models.video_mae import VideoMAE
except ImportError:
    from core.base_task import BaseTask
    from core.context import PipelineContext, VideoCaption
    from models.video_mae import VideoMAE


# Frames per second to sample from the video (keeps CPU load predictable)
_SAMPLE_FPS: float = 4.0


class VideoMAETask(BaseTask):
    """
    Task for generating frame-level video captions using a VideoMAE model.

    Slides a fixed window across the video, samples frames at _SAMPLE_FPS,
    and calls the configured VideoMAE backend to generate a short caption
    for each window.  Results are appended to context.video_captions.
    """

    def __init__(
        self,
        model: VideoMAE,
        window_seconds: float = 5.0,
    ) -> None:
        """
        Args:
            model:          Configured VideoMAE instance (mock or CoreML)
            window_seconds: Duration of each captioning window in seconds
        """
        super().__init__("VideoMAE")
        self.model = model
        self.window_seconds = window_seconds

    # ── BaseTask interface ────────────────────────────────────────────────────

    def _run(self, context: PipelineContext) -> None:
        """
        Slide a window across the video and generate captions.

        Args:
            context: Pipeline context with video_metadata set
        """
        logger = context.logger
        if context.video_metadata is None:
            raise ValueError("video_metadata not available in context")

        video_path    = context.video_metadata.path
        total_seconds = context.video_metadata.duration_seconds
        source_fps    = context.video_metadata.fps

        logger.log_info(
            f"VideoMAE: captioning {total_seconds:.1f}s video "
            f"in {self.window_seconds}s windows "
            f"(sampling at {_SAMPLE_FPS} fps)"
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video for VideoMAE: {video_path}")

        captions: List[VideoCaption] = []

        try:
            window_start = 0.0
            window_id    = 0

            while window_start < total_seconds:
                window_end  = min(window_start + self.window_seconds, total_seconds)
                frames      = self._read_window_frames(cap, window_start, window_end, source_fps)

                caption_text = self.model.generate_caption(
                    frames, source_fps, window_start
                )
                logger.log_info(
                    f"  window {window_id} [{window_start:.1f}-{window_end:.1f}s]: {caption_text}"
                )

                captions.append(VideoCaption(
                    window_id  = window_id,
                    start_time = window_start,
                    end_time   = window_end,
                    caption    = caption_text,
                    confidence = 1.0,
                ))

                window_start = window_end
                window_id   += 1

        finally:
            cap.release()
            gc.collect()

        context.video_captions.extend(captions)

        context.processing_metadata["video_mae"] = {
            "model_info":         self.model.get_model_info(),
            "window_seconds":     self.window_seconds,
            "sample_fps":         _SAMPLE_FPS,
            "total_windows":      len(captions),
            "total_captions":     len(captions),
        }

        logger.log_info(f"VideoMAE: {len(captions)} captions generated")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _read_window_frames(
        self,
        cap: cv2.VideoCapture,
        start_s: float,
        end_s: float,
        source_fps: float,
    ) -> List[np.ndarray]:
        """
        Read frames for the window [start_s, end_s), sampled at _SAMPLE_FPS.

        Seeks the capture to start_s, then reads every N-th frame so that
        the effective frame rate matches _SAMPLE_FPS.
        """
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)

        # How many source frames between each sampled frame
        step = max(1, int(round(source_fps / _SAMPLE_FPS)))
        total_source_frames = int((end_s - start_s) * source_fps)

        frames: List[np.ndarray] = []
        read_count = 0

        while read_count < total_source_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if read_count % step == 0:
                frames.append(frame)
            read_count += 1

        return frames

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create_default(cls) -> "VideoMAETask":
        """
        Create a VideoMAETask with the default mock backend.

        Returns:
            Configured VideoMAETask using MockVideoMAE
        """
        return cls(model=VideoMAE(backend="mock"), window_seconds=5.0)