import asyncio
import json
import os
from pathlib import Path
import shutil

import decky

PLUGIN_DIR = Path(__file__).resolve().parent
USER_HOME = Path(decky.DECKY_USER_HOME)
DATA_DIR = USER_HOME / ".local" / "share" / "ally-sound-npu"
PYTHON = DATA_DIR / ".venv" / "bin" / "python"
SAMPLE_DIR = USER_HOME / "Downloads" / "AllySoundNPU"
DEMO = {
    "original": "01-stereo.wav",
    "hrtf": "02-hrtf.wav",
    "stemgen": "04-stemgen-hrtf.wav",
}
MODELS = ("vocals.onnx", "ku100.sofa")
SETTINGS = DATA_DIR / "settings.json"
WORKER_LOG = DATA_DIR / "worker.log"
LIVE_MODES = {"off", "hrtf", "stemgen"}
RESTART_LIMIT = 5        # unexpected live-worker exits tolerated...
RESTART_WINDOW = 120     # ...within this many seconds


class Plugin:
    async def _main(self):
        self.lock = asyncio.Lock()
        self.live = None
        self.player = None
        self.workers = set()
        self.live_mode = "off"
        self.live_status = {}
        # Leftovers from the removed TIGER-DnR file renderer.
        for stale in DATA_DIR.glob("render-*"):
            shutil.rmtree(stale, ignore_errors=True)
        self.wanted = self._load_mode()
        self.restarts = []
        decky.logger.info("Ally Sound Lab initialized, saved mode %s", self.wanted)
        if self.wanted != "off":
            self._watch(self._autostart())

    def _load_mode(self):
        try:
            mode = json.loads(SETTINGS.read_text()).get("mode", "off")
        except (OSError, ValueError, AttributeError):
            return "off"
        return mode if mode in LIVE_MODES else "off"

    def _save_mode(self, mode):
        self.wanted = mode
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            SETTINGS.write_text(json.dumps({"mode": mode}))
        except OSError as exc:
            decky.logger.warning("Cannot save mode: %s", exc)

    async def _autostart(self):
        # At boot Decky can start before the user's PipeWire session.
        pulse = Path(f"/run/user/{os.getuid()}/pulse/native")
        for _ in range(60):
            if pulse.exists():
                break
            await asyncio.sleep(2)
        await asyncio.sleep(3)
        async with self.lock:
            if self.live is None and self.wanted != "off":
                try:
                    await self._start_live(self.wanted)
                except Exception as exc:
                    self.live_status = {"error": f"Автозапуск не удался: {exc}"}
                    decky.logger.error("Autostart failed: %s", exc)

    async def get_status(self):
        return {
            "installed": PYTHON.is_file() and all((DATA_DIR / "models" / name).is_file() for name in MODELS),
            "mode": self.live_mode,
            "wanted": self.wanted,
            "live": self.live_status,
            "playing": self.player is not None and self.player.returncode is None,
        }

    async def _spawn(self, *args, capture=False):
        if os.geteuid() == 0:
            raise RuntimeError("Ally Sound Lab должен запускаться от пользователя Steam.")
        env = os.environ.copy()
        env["HOME"] = str(USER_HOME)
        env["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
        env["PULSE_SERVER"] = f"unix:{env['XDG_RUNTIME_DIR']}/pulse/native"
        env["OPENBLAS_NUM_THREADS"] = "1"
        # Decky is a PyInstaller bundle: its LD_LIBRARY_PATH points at a bundled
        # libstdc++ that is older than the system one, which breaks mpv and
        # anything else linked against current CachyOS libraries.
        for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH"):
            env.pop(name, None)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if WORKER_LOG.stat().st_size > 1_000_000:
                WORKER_LOG.unlink()
        except FileNotFoundError:
            pass
        with WORKER_LOG.open("ab") as log:
            return await asyncio.create_subprocess_exec(
                *args,
                env=env,
                stdout=asyncio.subprocess.PIPE if capture else asyncio.subprocess.DEVNULL,
                stderr=log,
            )

    def _watch(self, coro):
        task = asyncio.create_task(coro)
        self.workers.add(task)
        task.add_done_callback(self.workers.discard)

    async def _stop(self, process):
        if process is None or process.returncode is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), 8)
        except asyncio.TimeoutError:
            if process.returncode is None:
                process.kill()
            await process.wait()

    async def _watch_live(self, process, mode):
        async for line in process.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if self.live is process:
                self.live_status = {**self.live_status, **event}
        result = await process.wait()
        if self.live is not process:
            return
        self.live = None
        self.live_mode = "off"
        error = self.live_status.get("error") or f"Аудиопроцесс завершился (код {result})."
        decky.logger.warning("Live worker exited (%s): %s", result, error)
        now = asyncio.get_running_loop().time()
        self.restarts = [moment for moment in self.restarts if now - moment < RESTART_WINDOW]
        if self.wanted != mode or len(self.restarts) >= RESTART_LIMIT:
            self.live_status = {"error": error}
            return
        self.restarts.append(now)
        self.live_status = {"phase": "starting", "error": f"Перезапуск после сбоя: {error}"}
        await asyncio.sleep(2)
        async with self.lock:
            if self.live is None and self.wanted == mode:
                try:
                    await self._start_live(mode)
                except Exception as exc:
                    self.live_status = {"error": str(exc)}

    async def _start_live(self, mode):
        """Start the live worker. Caller holds self.lock."""
        if not PYTHON.is_file():
            raise RuntimeError("Модели ещё не установлены.")
        await self._stop(self.player)
        self.player = None
        self.live_status = {"phase": "starting"}
        self.live = await self._spawn(PYTHON, PLUGIN_DIR / "audio_worker.py", "live", mode, capture=True)
        self.live_mode = mode
        self._watch(self._watch_live(self.live, mode))

    async def set_live(self, mode):
        if mode not in LIVE_MODES:
            raise ValueError("Неизвестный режим.")
        async with self.lock:
            self._save_mode(mode)
            self.restarts = []
            process, self.live = self.live, None
            await self._stop(process)
            self.live_mode = "off"
            self.live_status = {}
            if mode != "off":
                await self._start_live(mode)
        return await self.get_status()

    async def play_audio(self, collection, variant):
        if collection == "demo" and variant in DEMO:
            file = SAMPLE_DIR / DEMO[variant]
        else:
            raise ValueError("Тестовый файл недоступен.")
        async with self.lock:
            if self.live is not None:
                raise RuntimeError("Выключи живую обработку, чтобы не применять HRTF дважды.")
            if not file.is_file():
                raise FileNotFoundError(file.name)
            await self._stop(self.player)
            self.player = await self._spawn(
                "/usr/bin/env", "DISABLE_NPU_SR=1", "/usr/bin/mpv", "--no-config",
                "--no-video", "--force-window=no", "--ao=pipewire", "--", file,
            )
        return await self.get_status()

    async def stop_audio(self):
        async with self.lock:
            await self._stop(self.player)
            self.player = None
        return await self.get_status()

    async def _unload(self):
        async with self.lock:
            processes = (self.live, self.player)
            self.live = self.player = None
            for process in processes:
                await self._stop(process)
            # Watchers may be waiting on self.lock (auto-restart); cancel them
            # instead of awaiting, or unload would deadlock.
            for task in list(self.workers):
                task.cancel()
            if self.workers:
                await asyncio.gather(*self.workers, return_exceptions=True)
