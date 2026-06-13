import { X, SlidersHorizontal, ShieldCheck, LogIn, LogOut } from "lucide-react";

export default function SettingsPanel({
  open,
  settings,
  dres,
  onChange,
  onClose,
  onDresLogin,
  onDresLogout,
}) {
  function updateField(key, value) {
    onChange({
      ...settings,
      [key]: value,
    });
  }

  return (
    <aside className={open ? "settings-panel open" : "settings-panel"}>
      <div className="settings-header">
        <h3>
          <SlidersHorizontal size={17} />
          Retrieval Settings
        </h3>

        <button type="button" onClick={onClose}>
          <X size={17} />
        </button>
      </div>

      <div className="settings-section">
        <h4>Search Pipeline</h4>

        <SwitchRow
          label="Split query"
          checked={settings.useSplit}
          onChange={(value) => updateField("useSplit", value)}
        />

        <SwitchRow
          label="Translate VI → EN"
          checked={settings.useTranslate}
          onChange={(value) => updateField("useTranslate", value)}
        />

        <label className="setting-slider">
          <div className="setting-slider-header">
            <span>Top K</span>
            <strong>{settings.topK}</strong>
          </div>

          <input
            type="range"
            min="1"
            max="200"
            value={settings.topK}
            onChange={(e) => updateField("topK", Number(e.target.value))}
          />
        </label>
      </div>

      <div className="settings-section">
        <h4>DRES Submission</h4>

        <label className="setting-field">
          <span>DRES URL</span>
          <input
            value={settings.submitUrl}
            placeholder="https://dres-or-ngrok-url"
            onChange={(e) => updateField("submitUrl", e.target.value)}
          />
        </label>

        <label className="setting-field">
          <span>Evaluation ID</span>
          <input
            value={settings.evaluationId}
            placeholder="Optional. Empty = legacy v1 submit"
            onChange={(e) => updateField("evaluationId", e.target.value)}
          />
        </label>

        <label className="setting-field">
          <span>Team ID</span>
          <input
            value={settings.username}
            placeholder="team"
            onChange={(e) => updateField("username", e.target.value)}
          />
        </label>

        <label className="setting-field">
          <span>Password</span>
          <input
            type="password"
            value={settings.password}
            placeholder="password"
            onChange={(e) => updateField("password", e.target.value)}
          />
        </label>

        <div className={dres?.sessionId ? "dres-status connected" : "dres-status"}>
          <span />
          {dres?.sessionId
            ? `DRES logged in · ${dres.sessionId.slice(0, 10)}...`
            : "DRES not logged in"}
        </div>

        <div className="settings-actions">
          <button type="button" onClick={onDresLogin} disabled={dres?.loading}>
            <LogIn size={15} />
            {dres?.loading ? "Logging in..." : "Login DRES"}
          </button>

          <button type="button" onClick={onDresLogout}>
            <LogOut size={15} />
            Clear
          </button>
        </div>

        <p className="settings-warning">
          <ShieldCheck size={14} />
          Frontend chỉ gọi backend proxy. Backend mới gọi DRES qua /api/dres/login và /api/dres/submit.
        </p>
      </div>
    </aside>
  );
}

function SwitchRow({ label, checked, onChange }) {
  return (
    <label className="setting-switch-row">
      <span>{label}</span>

      <button
        type="button"
        className={checked ? "switch-control checked" : "switch-control"}
        onClick={() => onChange(!checked)}
      >
        <span />
      </button>
    </label>
  );
}
