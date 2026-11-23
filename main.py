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
MODELS = ("vocals.onnx", "ku100.sofa")
EQ_MODELS = ("yamnet/yamnet.onnx", "yamnet/yamnet.data", "yamnet/discogs-effnet-bsdynamic-1.onnx")
SETTINGS = DATA_DIR / "settings.json"
WORKER_LOG = DATA_DIR / "worker.log"
PROFILES = {"game", "movie", "music"}
DEVICES = {"headphones", "speakers", "glasses"}
LIVE_MODES = {"off"} | PROFILES
DEFAULT = {"mode": "off", "neural": True, "reverb": True, "upmix": False, "auto_eq": False}
RESTART_LIMIT = 5        # unexpected live-worker exits tolerated...
RESTART_WINDOW = 120     # ...within this many seconds


def clean_entry(entry, fallback):
    entry = entry if isinstance(entry, dict) else {}
    mode = entry.get("mode", fallback["mode"])
    # Modes from the first version ("hrtf", "stemgen") map onto profiles.
    mode = {"hrtf": "game", "stemgen": "movie"}.get(mode, mode)
    values = {"mode": mode if mode in LIVE_MODES else fallback["mode"]}
    for key in ("neural", "reverb", "upmix", "auto_eq"):
        value = entry.get(key)
        values[key] = value if isinstance(value, bool) else fallback[key]
    return values


def has_models(names):
    return all((DATA_DIR / "models" / name).is_file() for name in names)


class Plugin:
    async def _main(self):
        self.lock = asyncio.Lock()
        self.live = None
        self.workers = set()
        self.live_mode = "off"
        self.live_neural = True
        self.live_reverb = True
        self.live_status = {}
        self.bypass = False
        self.installer = None
        self.install_status = {}
        self.eq_install_status = {}
        self.app = {"id": 0, "name": ""}
        self.restarts = []
        # Leftovers from the removed TIGER-DnR file renderer.
        for stale in DATA_DIR.glob("render-*"):
            shutil.rmtree(stale, ignore_errors=True)
        self.settings = self._load_settings()
        decky.logger.info("Ally Sound Lab initialized, default %s", self.settings["default"])
        self._watch(self._autostart())

    # -- settings ---------------------------------------------------------

    def _load_settings(self):
        try:
            raw = json.loads(SETTINGS.read_text())
        except (OSError, ValueError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        if "default" not in raw and "mode" in raw:  # first-version file
            raw = {"default": {"mode": raw["mode"]}}
        default = clean_entry(raw.get("default", {}), DEFAULT)
        apps = {}
        for appid, entry in (raw.get("apps") or {}).items():
            if str(appid).isdigit() and isinstance(entry, dict):
                apps[str(appid)] = {**clean_entry(entry, default), "name": str(entry.get("name", ""))[:120]}
        devices = {
            str(name)[:200]: kind for name, kind in (raw.get("devices") or {}).items() if kind in DEVICES
        }
        return {"default": default, "apps": apps, "devices": devices}

    def _save_settings(self):
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            temporary = SETTINGS.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.settings, ensure_ascii=False, indent=1))
            temporary.replace(SETTINGS)
        except OSError as exc:
            decky.logger.warning("Cannot save settings: %s", exc)

    def _app_entry(self):
        return self.settings["apps"].get(str(self.app["id"])) if self.app["id"] else None

    def _effective(self):
        entry = self._app_entry()
        source = entry or self.settings["default"]
        return {key: source[key] for key in DEFAULT}

    def _edit_context(self, **changes):
        """Apply changes to the running app's own entry, or to the default."""
        if self.app["id"]:
            key = str(self.app["id"])
            entry = self.settings["apps"].get(key) or dict(self.settings["default"])
            entry.update(changes, name=self.app["name"])
            self.settings["apps"][key] = entry
        else:
            self.settings["default"].update(changes)
        self._save_settings()

    # -- status -----------------------------------------------------------

    async def get_status(self):
        effective = self._effective()
        return {
            "installed": PYTHON.is_file() and has_models(MODELS),
            "mode": self.live_mode,
            "wanted": effective["mode"],
            "neural": effective["neural"],
            "reverb": effective["reverb"],
            "upmix": effective["upmix"],
            "auto_eq": effective["auto_eq"],
            "eq_ready": has_models(EQ_MODELS),
            "bypass": self.bypass,
            "app": self.app,
            "custom": self._app_entry() is not None,
            "default": self.settings["default"],
            "devices": self.settings["devices"],
            "live": self.live_status,
            "install": self.install_status,
            "eq_install": self.eq_install_status,
        }

    # -- processes --------------------------------------------------------

    async def _spawn(self, *args):
        if os.geteuid() == 0:
            raise RuntimeError("Ally Sound Lab должен запускаться от пользователя Steam.")
        env = os.environ.copy()
        env["HOME"] = str(USER_HOME)
        env["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
        env["PULSE_SERVER"] = f"unix:{env['XDG_RUNTIME_DIR']}/pulse/native"
        env["OPENBLAS_NUM_THREADS"] = "1"
        # Decky is a PyInstaller bundle: its LD_LIBRARY_PATH points at a bundled
        # libstdc++ that is older than the system one, which breaks programs
        # linked against current CachyOS libraries.
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
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
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

    async def _command(self, **fields):
        """Send a control message to the running worker (no audio restart)."""
        process = self.live
        if process is None or process.returncode is not None or process.stdin is None:
            return False
        try:
            process.stdin.write((json.dumps(fields) + "\n").encode())
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            return False
        return True

    async def _start_live(self, mode, neural, reverb, upmix, auto_eq):
        """Start the live worker. Caller holds self.lock."""
        if not PYTHON.is_file():
            raise RuntimeError("Модели ещё не установлены.")
        args = [PYTHON, PLUGIN_DIR / "audio_worker.py", "live", mode]
        if not neural:
            args.append("--no-neural")
        if not reverb:
            args.append("--no-reverb")
        if upmix:
            args.append("--upmix")
        if auto_eq:
            args.append("--auto-eq")
        self.live_status = {"phase": "starting"}
        self.live = await self._spawn(*args)
        self.live_mode = mode
        self.live_neural = neural
        self.live_reverb = reverb
        await self._command(devices=self.settings["devices"])
        if self.bypass:
            await self._command(bypass=True)
        self._watch(self._watch_live(self.live))

    async def _apply(self):
        """Bring the worker in line with the effective settings. Caller holds self.lock."""
        target = self._effective()
        target["auto_eq"] = target["auto_eq"] and has_models(EQ_MODELS)
        if target["mode"] == "off":
            process, self.live = self.live, None
            await self._stop(process)
            self.live_mode = "off"
            self.live_status = {}
            self.bypass = False
            return
        if self.live is not None and self.live_mode in PROFILES:
            if await self._command(
                profile=target["mode"], neural=target["neural"], reverb=target["reverb"],
                upmix=target["upmix"], auto_eq=target["auto_eq"],
            ):
                self.live_mode = target["mode"]
                self.live_neural = target["neural"]
                self.live_reverb = target["reverb"]
                self.live_status = {
                    **self.live_status,
                    "profile": target["mode"],
                    "neural": target["neural"],
                    "reverb": target["reverb"],
                    "upmix": target["upmix"],
                    "auto_eq": target["auto_eq"],
                }
                return
        process, self.live = self.live, None
        await self._stop(process)
        await self._start_live(**target)

    async def _watch_live(self, process):
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
        if self._effective()["mode"] == "off" or len(self.restarts) >= RESTART_LIMIT:
            self.live_status = {"error": error}
            return
        self.restarts.append(now)
        self.live_status = {"phase": "starting", "error": f"Перезапуск после сбоя: {error}"}
        await asyncio.sleep(2)
        async with self.lock:
            if self.live is None:
                try:
                    await self._apply()
                except Exception as exc:
                    self.live_status = {"error": str(exc)}

    async def _autostart(self):
        # At boot Decky can start before the user's PipeWire session.
        pulse = Path(f"/run/user/{os.getuid()}/pulse/native")
        for _ in range(60):
            if pulse.exists():
                break
            await asyncio.sleep(2)
        await asyncio.sleep(3)
        async with self.lock:
            if self.live is None and self._effective()["mode"] != "off":
                try:
                    await self._apply()
                except Exception as exc:
                    self.live_status = {"error": f"Автозапуск не удался: {exc}"}
                    decky.logger.error("Autostart failed: %s", exc)

    # -- frontend API -----------------------------------------------------

    async def app_changed(self, appid, name):
        """Called by the frontend whenever the foreground Steam app changes."""
        appid = int(appid) if isinstance(appid, (int, float)) and appid > 0 else 0
        name = str(name or "")[:120] if appid else ""
        async with self.lock:
            if appid == self.app["id"]:
                return await self.get_status()
            self.app = {"id": appid, "name": name}
            entry = self._app_entry()
            if entry is not None and name and entry.get("name") != name:
                entry["name"] = name
                self._save_settings()
            self.restarts = []
            decky.logger.info("Foreground app %s (%s): %s", appid, name, self._effective())
            await self._apply()
        return await self.get_status()

    async def set_live(self, mode):
        if mode not in LIVE_MODES:
            raise ValueError("Неизвестный режим.")
        async with self.lock:
            self._edit_context(mode=mode)
            self.restarts = []
            await self._apply()
        return await self.get_status()

    async def set_neural(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("Ожидалось true/false.")
        async with self.lock:
            self._edit_context(neural=enabled)
            await self._apply()
        return await self.get_status()

    async def set_reverb(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("Ожидалось true/false.")
        async with self.lock:
            self._edit_context(reverb=enabled)
            await self._apply()
        return await self.get_status()

    async def set_upmix(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("Ожидалось true/false.")
        async with self.lock:
            self._edit_context(upmix=enabled)
            await self._apply()
        return await self.get_status()

    async def set_auto_eq(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("Ожидалось true/false.")
        async with self.lock:
            if enabled and not has_models(EQ_MODELS):
                if not PYTHON.is_file():
                    raise RuntimeError("Сначала установите основные компоненты.")
                if self.installer is None or self.installer.returncode is not None:
                    await self._start_installer(auto_eq=True)
                elif self.eq_install_status.get("phase") != "running":
                    raise RuntimeError("Дождитесь установки основных компонентов.")
            self._edit_context(auto_eq=enabled)
            await self._apply()
        return await self.get_status()

    async def reset_app(self):
        """Forget the running app's own settings; it follows the default again."""
        async with self.lock:
            if self.settings["apps"].pop(str(self.app["id"]), None) is not None:
                self._save_settings()
            await self._apply()
        return await self.get_status()

    async def set_device_kind(self, kind):
        """Override the output type of the current device; "auto" forgets it."""
        if kind != "auto" and kind not in DEVICES:
            raise ValueError("Неизвестный тип выхода.")
        async with self.lock:
            sink = self.live_status.get("sink")
            if not sink:
                raise RuntimeError("Обработка не запущена: тип выхода не к чему применить.")
            if kind == "auto":
                self.settings["devices"].pop(sink, None)
            else:
                self.settings["devices"][sink] = kind
            self._save_settings()
            await self._command(devices=self.settings["devices"])
        return await self.get_status()

    async def install_runtime(self):
        """Download uv (if missing), Python 3.13, libraries and models."""
        async with self.lock:
            await self._start_installer()
        return await self.get_status()

    async def _start_installer(self, auto_eq=False):
        if self.installer is not None and self.installer.returncode is None:
            raise RuntimeError("Установка уже идёт.")
        script = PLUGIN_DIR / "install-runtime.sh"
        if not script.is_file():
            raise RuntimeError("В плагине нет install-runtime.sh.")
        args = ("--auto-eq",) if auto_eq else ()
        self.installer = await self._spawn("/bin/bash", script, *args)
        status = self.eq_install_status if auto_eq else self.install_status
        status.clear()
        status.update(phase="running", step="Запуск")
        self._watch(self._watch_installer(self.installer, auto_eq, status))

    async def _watch_installer(self, process, auto_eq, status):
        last = ""
        async for raw in process.stdout:
            line = raw.decode(errors="replace").strip()
            if line.startswith("## "):
                last = line[3:]
                status.update(phase="running", step=last)
        code = await process.wait()
        async with self.lock:
            if self.installer is not process:
                return
            self.installer = None
            status.clear()
            if code == 0 and has_models(EQ_MODELS if auto_eq else MODELS) and PYTHON.is_file():
                status.update(phase="done")
                decky.logger.info("%s installed", "YAMNet" if auto_eq else "Runtime")
                if (auto_eq or self.live is None) and self._effective()["mode"] != "off":
                    try:
                        await self._apply()
                    except Exception as exc:
                        self.live_status = {"error": str(exc)}
            else:
                reason = last if last.startswith("Ошибка") else f"код {code}, шаг «{last}»"
                status.update(phase="error", error=f"Установка не удалась ({reason}). Подробности в worker.log.")
                decky.logger.error("Installation failed: %s", reason)

    async def set_bypass(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("Ожидалось true/false.")
        async with self.lock:
            self.bypass = enabled
            await self._command(bypass=enabled)
            self.live_status = {**self.live_status, "bypass": enabled}
        return await self.get_status()

    async def _unload(self):
        async with self.lock:
            process, self.live = self.live, None
            await self._stop(process)
            installer, self.installer = self.installer, None
            await self._stop(installer)
            # Watchers may be waiting on self.lock (auto-restart); cancel them
            # instead of awaiting, or unload would deadlock.
            for task in list(self.workers):
                task.cancel()
            if self.workers:
                await asyncio.gather(*self.workers, return_exceptions=True)
