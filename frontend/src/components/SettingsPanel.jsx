import { X, SlidersHorizontal, ShieldCheck } from "lucide-react";

export default function SettingsPanel({ open, settings, onChange, onClose }) {
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
        <h4>Submission Config</h4>

        <label className="setting-field">
          <span>Submit URL</span>
          <input
            value={settings.submitUrl}
            placeholder="https://example.com/submit"
            onChange={(e) => updateField("submitUrl", e.target.value)}
          />
        </label>

        <label className="setting-field">
          <span>Username</span>
          <input
            value={settings.username}
            placeholder="username"
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

        <p className="settings-warning">
          <ShieldCheck size={14} />
          Hiện tại các giá trị này chỉ giữ trong frontend state. Khi làm thật,
          không nên lưu mật khẩu plaintext ở browser.
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