import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, ChevronLeft, ChevronRight, RotateCcw } from "lucide-react";
import "./AgentReasoningPicker.css";

const effortLabel = (value) => ({ none: "None", minimal: "Minimal", low: "Low", medium: "Medium", high: "High", xhigh: "Extra high", max: "Max", ultra: "Ultra" }[value] || value);

export default function AgentReasoningPicker({ config, disabled }) {
  const [open, setOpen] = useState(false);
  const [view, setView] = useState("effort");
  const host = useRef(null);
  const model = config.models?.find((item) => (item.model || item.id) === config.model);
  const levels = model?.supported_reasoning_efforts || [];
  const index = Math.max(0, levels.findIndex((item) => item.reasoningEffort === config.reasoningEffort));
  const progress = levels.length > 1 ? index / (levels.length - 1) : 0;
  const level = levels[index];
  const label = effortLabel(level?.reasoningEffort || config.reasoningEffort);

  useEffect(() => {
    if (!open) return;
    const outside = (event) => { if (!host.current?.contains(event.target)) setOpen(false); };
    const escape = (event) => {
      if (event.key === "Escape") { setOpen(false); host.current?.querySelector(".agent-reasoning-trigger")?.focus(); }
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);

  return <div className="agent-reasoning-picker" ref={host}>
    <button type="button" className="search-select agent-reasoning-trigger" disabled={disabled || !model}
      aria-label={`Model and reasoning: ${model?.name || "Loading models"}${label ? `, ${label}` : ""}`}
      aria-expanded={open} aria-haspopup="dialog" onClick={() => { setOpen(!open); setView("effort"); }}>
      <span>{model?.name || "Loading models…"}</span>{label && <span className="agent-effort-value">{label}</span>}<ChevronDown size={13} />
    </button>
    {open && <div className="agent-reasoning-popover" role="dialog" aria-label="Model and reasoning settings"
      style={{ "--effort-color": `hsl(${215 + progress * 165} 85% 65%)` }}>
      {view === "effort" ? <>
        <button type="button" className="agent-effort-reset" title="Reset reasoning effort" aria-label="Reset reasoning effort"
          disabled={disabled || !model?.default_reasoning_effort}
          onClick={() => config.onEffortChange(model.default_reasoning_effort)}><RotateCcw size={16} /></button>
        <div className="agent-effort-heading">{label || "Model default"}</div>
        <button type="button" className="agent-effort-model" onClick={() => setView("models")}>
          {model.name}<ChevronRight size={13} />
        </button>
        {levels.length > 1 ? <>
          <div className="agent-effort-slider" style={{ "--effort-fill": `calc(${progress * 100}% + ${16 - progress * 32}px)` }}>
            <div className="agent-effort-dots" aria-hidden="true">{levels.map((item) => <i key={item.reasoningEffort} />)}</div>
            <input type="range" min="0" max={levels.length - 1} step="1" value={index}
              aria-label="Reasoning effort" aria-valuetext={label} disabled={disabled}
              onChange={(event) => config.onEffortChange(levels[Number(event.target.value)].reasoningEffort)} />
          </div>
          <div className="agent-effort-limits"><span>{effortLabel(levels[0].reasoningEffort)}</span><span>{effortLabel(levels.at(-1).reasoningEffort)}</span></div>
        </> : null}
        <p className="agent-effort-description">{level?.description || "Uses the model’s default reasoning effort."}</p>
      </> : <>
        <button type="button" className="agent-model-back" onClick={() => setView("effort")}><ChevronLeft size={15} />Select model</button>
        <div role="listbox" aria-label="Assistant model" className="agent-model-list">
          {config.models.map((item) => <button type="button" role="option" key={item.model || item.id}
            aria-selected={(item.model || item.id) === config.model} disabled={disabled}
            onClick={() => { config.onModelChange(item.model || item.id); setView("effort"); }}>
            <span>{item.name}</span>{(item.model || item.id) === config.model && <Check size={16} />}
          </button>)}
        </div>
      </>}
    </div>}
  </div>;
}
