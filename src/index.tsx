import { callable, definePlugin } from "@decky/api";
import { ButtonItem, DropdownItem, PanelSection, PanelSectionRow, staticClasses } from "@decky/ui";
import { useEffect, useState } from "react";
import { FaHeadphones } from "react-icons/fa";

type Mode = "off" | "hrtf" | "stemgen";
type Variant = "original" | "hrtf" | "stemgen";

interface Status {
  installed: boolean;
  mode: Mode;
  wanted: Mode;
  live: { phase?: string; output?: string; error?: string; avg_ms?: number; slow_blocks?: number };
  playing: boolean;
}

const getStatus = callable<[], Status>("get_status");
const setLive = callable<[Mode], Status>("set_live");
const playAudio = callable<["demo", Variant], Status>("play_audio");
const stopAudio = callable<[], Status>("stop_audio");

const variants: Record<Variant, string> = {
  original: "Исходное стерео",
  hrtf: "Обычный HRTF",
  stemgen: "StemgenRT + HRTF",
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
  const active = !!status && status.mode !== "off";
  const message = error || live?.error;
  const demoButtons = (["original", "hrtf", "stemgen"] as const).map((variant) => (
    <PanelSectionRow key={variant}>
      <ButtonItem disabled={busy || active} onClick={() => void run(() => playAudio("demo", variant))}>
        {variants[variant]}
      </ButtonItem>
    </PanelSectionRow>
  ));

  return (
    <>
      <PanelSection title="Звук игры">
        <PanelSectionRow>
          <DropdownItem
            label="Обработка"
            description="Весь звук игр и фильмов идёт через обработку. Режим сохраняется после перезагрузки."
            disabled={busy || !status?.installed}
            selectedOption={status?.wanted ?? status?.mode ?? "off"}
            rgOptions={[
              { data: "off", label: "Выключена" },
              { data: "hrtf", label: "Обычный HRTF" },
              { data: "stemgen", label: "StemgenRT + HRTF" },
            ]}
            onChange={(option) => void run(() => setLive(option.data as Mode))}
          />
        </PanelSectionRow>
        {(live?.phase === "starting" || active) && (
          <PanelSectionRow>
            <div style={{ lineHeight: 1.45, opacity: 0.85 }}>
              {live?.phase === "starting" ? "Подключаю обработку…" : live?.output}
              {live?.avg_ms !== undefined && (
                <div>{live.avg_ms} мс на блок · превышений бюджета {live.slow_blocks}</div>
              )}
            </div>
          </PanelSectionRow>
        )}
        {status && !status.installed && <PanelSectionRow>Модели не установлены.</PanelSectionRow>}
        {message && <PanelSectionRow><div role="alert" style={{ color: "#f4b7ab" }}>{message}</div></PanelSectionRow>}
      </PanelSection>
      <PanelSection title="Сравнение на готовом отрывке">
        {demoButtons}
        {status?.playing && (
          <PanelSectionRow>
            <ButtonItem disabled={busy} onClick={() => void run(stopAudio)}>Остановить</ButtonItem>
          </PanelSectionRow>
        )}
      </PanelSection>
    </>
  );
}

export default definePlugin(() => ({
  name: "Ally Sound Lab",
  titleView: <div className={staticClasses.Title}>Ally Sound Lab</div>,
  content: <Content />,
  icon: <FaHeadphones />,
}));
