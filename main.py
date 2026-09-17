#!/usr/bin/env python3
"""Talk to the published Voice Agent using the default audio devices."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import traceback
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

try:
    import sounddevice
except ImportError as error:
    raise SystemExit(
        "Missing dependency 'sounddevice'. Run: "
        "python -m pip install -r requirements.txt"
    ) from error

try:
    from websockets.asyncio.client import connect
    from websockets.exceptions import WebSocketException
except ImportError as error:
    raise SystemExit(
        "Missing dependency 'websockets'. Run: "
        "python -m pip install -r requirements.txt"
    ) from error

from debug_agent import (
    DEFAULT_ENV_FILE,
    VOICE_AGENT_FEATURE,
    EventPrinter,
    azure_access_token,
    load_agent_name,
    load_env_file,
    realtime_url,
    validate_endpoint,
    wire_event,
)


SAMPLE_RATE = 24_000
CHANNELS = 1
CHUNK_FRAMES = 2_400
AUDIO_DELTA_EVENTS = {
    "response.audio.delta",
    "response.output_audio.delta",
}


def discard_pending_audio(audio_queue: asyncio.Queue[bytes]) -> None:
    while True:
        try:
            audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        else:
            audio_queue.task_done()


async def stream_microphone(
    websocket: Any,
    microphone: Any,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        audio_data, overflowed = await asyncio.to_thread(
            microphone.read, CHUNK_FRAMES
        )
        if stop_event.is_set():
            return
        if overflowed:
            print("audio> microphone input overflow", file=sys.stderr)
        await websocket.send(
            wire_event(
                "input_audio_buffer.append",
                audio=base64.b64encode(bytes(audio_data)).decode("ascii"),
            )
        )


async def play_audio(
    speaker: Any,
    audio_queue: asyncio.Queue[bytes],
) -> None:
    while True:
        audio_chunk = await audio_queue.get()
        try:
            await asyncio.to_thread(speaker.write, audio_chunk)
        finally:
            audio_queue.task_done()


async def receive_events(
    websocket: Any,
    printer: EventPrinter,
    audio_queue: asyncio.Queue[bytes],
    session_ready: asyncio.Event,
) -> None:
    async for raw in websocket:
        try:
            event = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("Voice Agent returned invalid JSON") from error
        if not isinstance(event, dict):
            raise RuntimeError("Voice Agent event must be a JSON object")

        kind = str(event.get("type") or "")
        if kind == "session.updated":
            session_ready.set()
        elif kind in AUDIO_DELTA_EVENTS:
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                try:
                    audio_queue.put_nowait(base64.b64decode(delta, validate=True))
                except ValueError as error:
                    raise RuntimeError(
                        "Voice Agent returned invalid base64 audio"
                    ) from error
        elif kind == "input_audio_buffer.speech_started":
            discard_pending_audio(audio_queue)
        elif kind == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript")
            if isinstance(transcript, str) and transcript:
                printer.finish_line()
                print(f"you> {transcript}")

        printer.show(event)

    raise RuntimeError("Voice Agent closed the connection")


async def wait_for_session(
    session_ready: asyncio.Event,
    receiver_task: asyncio.Task[None],
    timeout_seconds: float,
) -> None:
    ready_task = asyncio.create_task(session_ready.wait())
    try:
        done, _ = await asyncio.wait(
            {ready_task, receiver_task},
            timeout=timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if receiver_task in done:
            await receiver_task
        if ready_task not in done:
            raise RuntimeError(
                f"timed out after {timeout_seconds:g}s waiting for session readiness"
            )
    finally:
        if not ready_task.done():
            ready_task.cancel()
            with suppress(asyncio.CancelledError):
                await ready_task


async def run(args: argparse.Namespace) -> None:
    file_values = load_env_file(args.env_file)
    endpoint = validate_endpoint(file_values.get("AZURE_VOICE_AGENTS_ENDPOINT", ""))
    agent_name = load_agent_name()
    session_id = f"local-audio-{uuid.uuid4().hex}"
    url = realtime_url(endpoint, agent_name, session_id)
    headers = {
        "Authorization": f"Bearer {azure_access_token()}",
        "Foundry-Features": VOICE_AGENT_FEATURE,
    }

    microphone = None
    speaker = None
    try:
        input_device = sounddevice.query_devices(kind="input")
        output_device = sounddevice.query_devices(kind="output")
        microphone = sounddevice.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_FRAMES,
            dtype="int16",
            channels=CHANNELS,
        )
        speaker = sounddevice.RawOutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_FRAMES,
            dtype="int16",
            channels=CHANNELS,
        )
        speaker.start()

        print(f"connecting agent={agent_name} agent_session_id={session_id}")
        async with connect(
            url,
            additional_headers=headers,
            max_size=None,
            ping_timeout=None,
            open_timeout=args.timeout,
        ) as websocket:
            stop_event = asyncio.Event()
            session_ready = asyncio.Event()
            audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
            printer = EventPrinter(args.verbose)
            playback_task = asyncio.create_task(play_audio(speaker, audio_queue))
            receiver_task = asyncio.create_task(
                receive_events(websocket, printer, audio_queue, session_ready)
            )
            microphone_task: asyncio.Task[None] | None = None

            try:
                await wait_for_session(session_ready, receiver_task, args.timeout)
                print("session=ready")
                print(
                    f"Listening on {input_device['name']}. "
                    "Speak now (Ctrl+C to stop)."
                )
                print(f"Playing responses on {output_device['name']}.")
                microphone.start()
                microphone_task = asyncio.create_task(
                    stream_microphone(websocket, microphone, stop_event)
                )

                done, _ = await asyncio.wait(
                    {microphone_task, playback_task, receiver_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if receiver_task in done:
                    await receiver_task
                if microphone_task in done:
                    await microphone_task
                    raise RuntimeError("Microphone stream ended unexpectedly")
                await playback_task
                raise RuntimeError("Speaker stream ended unexpectedly")
            finally:
                stop_event.set()
                if microphone_task is not None:
                    await asyncio.gather(microphone_task, return_exceptions=True)
                for task in (playback_task, receiver_task):
                    if task.done():
                        continue
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                printer.finish_line()
    finally:
        if microphone is not None:
            microphone.stop()
            microphone.close()
        if speaker is not None:
            speaker.stop()
            speaker.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Default-microphone client for the published Voice Agent."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="customer Foundry project settings (default: config/deploy.env)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="seconds to wait for connection and session readiness (default: 60)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print additional protocol event types (audio payloads stay hidden)",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nclosed by user", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError, WebSocketException) as error:
        print(traceback.format_exc())
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())