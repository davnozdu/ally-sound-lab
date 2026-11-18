# Ally Sound Lab

[Decky Loader](https://github.com/SteamDeckHomebrew/decky-loader) plugin for the ROG Ally / Xbox Ally X on CachyOS. Processes all game and video audio for headphones in real time, in Game Mode.

Плагин Decky для ROG Ally на CachyOS. Перехватывает весь звук игр и фильмов в игровом режиме, обрабатывает его для наушников и отдаёт на текущий аудиовыход.

## Режимы

- **Обычный HRTF.** Стерео раскладывается на четыре виртуальных динамика (спереди ±30°, сзади ±110°) и сводится в наушники по HRIR манекена KU100. Около 0,3 мс CPU на блок 128 сэмплов.
- **StemgenRT + HRTF.** Потоковая нейросеть [StemgenRT](https://github.com/sweetspotsoundsystem/stemgen-rt) отделяет голос. Всё, кроме голоса, уходит в тыловые виртуальные динамики, диалоги остаются впереди. Около 1,6 мс CPU на блок.

Обработка рассчитана на наушники.

## Как работает

1. `audio_worker.py live <mode>` создаёт null sink `ally_sound_lab`, делает его выходом по умолчанию и переносит на него уже играющие приложения.
2. `parec` читает монитор этого выхода блоками по 128 сэмплов (44,1 кГц, float32), воркер обрабатывает их и отдаёт через `pacat` на реальное устройство.
3. Воркер следит за устройствами. Выбор другого выхода или подключение новых наушников (Bluetooth, USB, HDMI) переключают выход на лету. Если устройство пропало, звук уходит на устройство с наивысшим приоритетом.
4. При выключении прежний выход по умолчанию восстанавливается, а виртуальный удаляется. Если остался мусор от аварийного завершения, он убирается при следующем запуске.

`main.py` (бэкенд Decky):
- запоминает выбранный режим в `~/.local/share/ally-sound-npu/settings.json` и поднимает его после перезагрузки;
- перезапускает упавший воркер (до 5 раз за 2 минуты);
- пишет stderr воркера в `worker.log`.

## Установка

На приставке, от обычного пользователя:

```bash
./install-runtime.sh          # venv (Python 3.13) + модели в ~/.local/share/ally-sound-npu
```

Сборка фронтенда:

```bash
npm ci
npm run build                 # dist/index.js
```

Скопируй `plugin.json`, `package.json`, `main.py`, `audio_worker.py` и `dist/index.js` в `~/homebrew/plugins/Ally Sound Lab/`, затем выполни `sudo systemctl restart plugin_loader`.

Кнопки сравнения в меню проигрывают `~/Downloads/AllySoundNPU/01-stereo.wav`, `02-hrtf.wav` и `04-stemgen-hrtf.wav` через `mpv`.

## Заметки

- Decky собран PyInstaller-ом и выставляет `LD_LIBRARY_PATH` на свою `libstdc++`, с которой системный mpv падает. Поэтому `main.py` очищает `LD_*` у дочерних процессов.
- `pactl` в PipeWire не отдаёт `index` в `--format=json list modules`, поэтому модуль ищется через `pactl list short modules`.
- mpv запускается с `DISABLE_NPU_SR=1`: Vulkan-слой NPU_SR роняет его на этой системе.
