import { callable, definePlugin } from "@decky/api";
import { ButtonItem, DropdownItem, PanelSection, PanelSectionRow, ToggleField, staticClasses } from "@decky/ui";
import { useEffect, useState } from "react";
import { FaHeadphones } from "react-icons/fa";

type Mode = "off" | "game" | "movie";
type Device = "headphones" | "speakers" | "glasses";

interface Status {
  installed: boolean;
  mode: Mode;
  wanted: Mode;
  neural: boolean;
  bypass: boolean;
  app: { id: number; name: string };
  custom: boolean;
  default: { mode: Mode; neural: boolean };
  devices: Record<string, Device>;
  install: { phase?: "running" | "done" | "error"; step?: string; error?: string };
  live: {
    phase?: string;
    output?: string;
    error?: string;
    avg_ms?: number;
    slow_blocks?: number;
    source?: "stereo" | "surround";
    sink?: string;
    device?: Device;
    device_auto?: Device;
    correction?: boolean;
  };
}

const getStatus = callable<[], Status>("get_status");
const setLive = callable<[Mode], Status>("set_live");
const setBypass = callable<[boolean], Status>("set_bypass");
const setNeural = callable<[boolean], Status>("set_neural");
const resetApp = callable<[], Status>("reset_app");
const setDeviceKind = callable<[Device | "auto"], Status>("set_device_kind");
const installRuntime = callable<[], Status>("install_runtime");

const deviceLabels: Record<Device, string> = { headphones: "Наушники", speakers: "Динамики", glasses: "Очки (XREAL)" };
const deviceHints: Record<Device, string> = {
  headphones: "Полный 3D через HRTF: звук вокруг головы.",
  speakers: "Голос в центре, сцена шире корпуса, мягкая компрессия.",
  glasses: "3D для открытых динамиков у ушей и подъём басов.",
};
const appChanged = callable<[number, string], Status>("app_changed");

// Follow the foreground app for the whole Steam session, not only while the
// quick-access menu is open, so per-app profiles switch automatically.
let reportedApp = -1;
function syncForegroundApp() {
  const app = SteamUIStore?.MainRunningApp;
  const id = app?.appid ?? 0;
  if (id === reportedApp) return;
  reportedApp = id;
  appChanged(id, app?.display_name ?? "").catch(() => {
    reportedApp = -1; // retry on the next tick
  });
}

const labels: Record<Mode, string> = { off: "Выключено", game: "Игра", movie: "Кино" };

const BUDGET_MS = (128 / 44100) * 1000;

const descriptions: Record<Mode, string> = {
  off: "Звук идёт без изменений.",
  game: "Точное направление звуков, маленькая комната, динамика не трогается.",
  movie: "Диалоги громче и в центре, громкие сцены сглажены, комната больше.",
};

const sources: Record<string, string> = {
  surround: "Источник 5.1/7.1: каждый канал на своём месте",
  stereo: "Источник стерео: объём строится из двух каналов",
};

// Decky rejects with name "Python Exception"; the real message is the last
// line of pythonTraceback, e.g. "RuntimeError: Модели ещё не установлены."
function errorText(reason: unknown): string {
  const traceback = (reason as { pythonTraceback?: unknown } | null)?.pythonTraceback;
  if (typeof traceback === "string") {
    const last = traceback.trim().split("\n").pop() ?? "";
    const text = last.replace(/^[\w.]*(Error|Exception):\s*/, "").trim();
    if (text) return text;
  }
  return String(reason);
}

function Content() {
  const [status, setStatus] = useState<Status>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const current = await getStatus();
        if (active) setStatus(current);
      } catch (reason) {
        if (active) setError(errorText(reason));
      }
    };
    void refresh();
    const timer = setInterval(() => void refresh(), 1500);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  const run = async (call: () => Promise<Status>) => {
    setBusy(true);
    setError("");
    try {
      setStatus(await call());
    } catch (reason) {
      setError(errorText(reason));
    } finally {
      setBusy(false);
    }
  };

  const live = status?.live;
  const selected: Mode = status?.wanted ?? status?.mode ?? "off";
  const running = !!status && status.mode !== "off" && live?.phase === "live";
  const message = error || live?.error;
  const inApp = !!status?.app.id;
  const profileValue = inApp && !status?.custom ? "default" : selected;
  const profileOptions = [
    ...(inApp ? [{ data: "default", label: `Как по умолчанию (${labels[status!.default.mode]})` }] : []),
    { data: "off", label: labels.off },
    { data: "game", label: labels.game },
    { data: "movie", label: labels.movie },
  ];
  const load = live?.avg_ms !== undefined ? Math.round((live.avg_ms / BUDGET_MS) * 100) : undefined;

  return (
    <PanelSection title="Объёмный звук">
      <PanelSectionRow>
        <div style={{ opacity: 0.85, fontSize: "0.9em" }}>
          {inApp ? (
            <>
              Сейчас: <b>{status?.app.name || `приложение ${status?.app.id}`}</b>
              {status?.custom ? " · свой профиль" : " · профиль по умолчанию"}
            </>
          ) : (
            "Нет запущенной игры: настраивается профиль по умолчанию"
          )}
        </div>
      </PanelSectionRow>
      <PanelSectionRow>
        <DropdownItem
          label={inApp ? "Профиль для этого приложения" : "Профиль по умолчанию"}
          description={descriptions[selected]}
          disabled={busy || !status?.installed}
          selectedOption={profileValue}
          rgOptions={profileOptions}
          onChange={(option) =>
            void run(() => (option.data === "default" ? resetApp() : setLive(option.data as Mode)))
          }
        />
      </PanelSectionRow>
      {selected !== "off" && (
        <PanelSectionRow>
          <ToggleField
            label="Нейросеть для стерео"
            description={
              status?.neural
                ? "StemgenRT отделяет голос и ставит его в центр. Лучше для дубляжа, но больше нагрузка на CPU."
                : "Простая матрица без нейросети: меньше нагрузка, голос отделяется грубее."
            }
            checked={!!status?.neural}
            disabled={busy}
            onChange={(enabled) => void run(() => setNeural(enabled))}
          />
        </PanelSectionRow>
      )}
      {selected !== "off" && running && live?.sink && (
        <PanelSectionRow>
          <DropdownItem
            label={`Тип выхода: ${live.output ?? ""}`}
            description={
              (live.device ? deviceHints[live.device] : "") +
              (live.correction ? " Коррекция динамиков Ally включена." : "")
            }
            disabled={busy}
            selectedOption={status?.devices[live.sink] ?? "auto"}
            rgOptions={[
              { data: "auto", label: `Авто (${live.device_auto ? deviceLabels[live.device_auto] : "—"})` },
              { data: "headphones", label: deviceLabels.headphones },
              { data: "speakers", label: deviceLabels.speakers },
              { data: "glasses", label: deviceLabels.glasses },
            ]}
            onChange={(option) => void run(() => setDeviceKind(option.data as Device | "auto"))}
          />
        </PanelSectionRow>
      )}
      {selected !== "off" && (
        <PanelSectionRow>
          <ToggleField
            label="Сравнить с оригиналом"
            description="Временно отключает обработку, громкость выровнена."
            checked={!!status?.bypass}
            disabled={busy || !running}
            onChange={(enabled) => void run(() => setBypass(enabled))}
          />
        </PanelSectionRow>
      )}
      {selected !== "off" && (
        <PanelSectionRow>
          <div style={{ lineHeight: 1.45, opacity: 0.85, fontSize: "0.9em" }}>
            {!running ? (
              "Подключаю обработку…"
            ) : (
              <>
                <div>
                  Выход: {live?.output}
                  {live?.device ? ` · ${deviceLabels[live.device]}` : ""}
                </div>
                {live?.source && <div>{sources[live.source]}</div>}
                {load !== undefined && (
                  <div>
                    Нагрузка {load}% бюджета · пропусков {live?.slow_blocks ?? 0}
                  </div>
                )}
              </>
            )}
          </div>
        </PanelSectionRow>
      )}
      {selected !== "off" && running && (
        <PanelSectionRow>
          <div style={{ lineHeight: 1.4, opacity: 0.65, fontSize: "0.85em" }}>
            Игры выбирают число каналов при запуске: включай обработку до запуска игры или фильма.
          </div>
        </PanelSectionRow>
      )}
      {status && !status.installed && (
        <>
          <PanelSectionRow>
            <div style={{ lineHeight: 1.4, fontSize: "0.9em" }}>
              {status.install.phase === "running"
                ? `Устанавливаю: ${status.install.step ?? ""}`
                : "Нужны компоненты: Python, библиотеки и модели (около 300 МБ, нужен интернет)."}
            </div>
          </PanelSectionRow>
          <PanelSectionRow>
            <ButtonItem
              layout="below"
              disabled={busy || status.install.phase === "running"}
              onClick={() => void run(installRuntime)}
            >
              {status.install.phase === "running" ? "Идёт установка…" : "Установить компоненты"}
            </ButtonItem>
          </PanelSectionRow>
          {status.install.error && (
            <PanelSectionRow>
              <div role="alert" style={{ color: "#f4b7ab" }}>{status.install.error}</div>
            </PanelSectionRow>
          )}
        </>
      )}
      {status?.installed && status.install.phase === "done" && (
        <PanelSectionRow>
          <div style={{ opacity: 0.8, fontSize: "0.9em" }}>Компоненты установлены.</div>
        </PanelSectionRow>
      )}
      {message && (
        <PanelSectionRow>
          <div role="alert" style={{ color: "#f4b7ab" }}>{message}</div>
        </PanelSectionRow>
      )}
    </PanelSection>
  );
}

export default definePlugin(() => {
  const lifetime = SteamClient.GameSessions.RegisterForAppLifetimeNotifications(() => {
    // The store updates slightly after the notification.
    setTimeout(syncForegroundApp, 1500);
  });
  const timer = setInterval(syncForegroundApp, 5000);
  syncForegroundApp();
  return {
    name: "Ally Sound Lab",
    titleView: <div className={staticClasses.Title}>Ally Sound Lab</div>,
    content: <Content />,
    icon: <FaHeadphones />,
    onDismount() {
      lifetime.unregister();
      clearInterval(timer);
    },
  };
});
