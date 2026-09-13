import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { X, GitMerge, Sparkles, Waves, ScanText, AudioLines } from "lucide-react";

/**
 * FusionSettingsModal — pop-up cấu hình cho search mode "fusion" (advanced
 * search).
 *
 * Cho phép tick nhiều semantic model, bật/tắt temporal (kèm duration
 * limit), OCR, ASR và điều chỉnh weight theo nhóm nguồn. Backend
 * Khi temporal bật, nó chạy multimodal PER EVENT:
 * mỗi event tự fuse các semantic model bằng weighted similarity, rồi kết hợp
 * đúng các method đã tick ở trên (semantic +
 * OCR/ASR nếu bật) TRƯỚC khi DP alignment ghép chuỗi — dùng chung danh sách
 * model đã tick ở "Semantic models", không có checklist model riêng cho
 * temporal.
 *
 * Bấm "Lưu cấu hình" → đóng modal, config được áp dụng ở lần bấm Search
 * tiếp theo (không tự động search khi save).
 *
 * Render qua createPortal vào document.body: modal KHÔNG được lồng bên
 * trong `.bottom-search-zone` (ancestor có `transform`, tạo containing
 * block riêng cho `position: fixed`) — nếu không portal, backdrop
 * `position: fixed` của modal sẽ bị tính toán lồng trong khung nhỏ đó
 * thay vì theo viewport, gây hiện tượng modal bị cắt/che sau khi search.
 */
export default function FusionSettingsModal({
  open,
  theme = "dark",
  models = [],
  modelRoles = {},
  value,
  onClose,
  onSave,
}) {
  const [draft, setDraft] = useState(value);
  const radarRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = prevOverflow; };
  }, [open]);

  if (!open) return null;

  function toggleModel(modelKey) {
    setDraft((prev) => {
      const checked = prev.semanticModels.some((m) => m.key === modelKey);
      return {
        ...prev,
        semanticModels: checked
          ? prev.semanticModels.filter((m) => m.key !== modelKey)
          : [...prev.semanticModels, { key: modelKey }],
      };
    });
  }

  function updateField(key, val) {
    setDraft((prev) => ({ ...prev, [key]: val }));
  }

  function updateWeight(key, value) {
    const enabled = { semantic: draft.semanticModels.length > 0, ocr: draft.useOcr, asr: draft.useAsr };
    if (!enabled[key]) return;
    const nextValue = Math.max(0, Math.min(1, Number(value) || 0));
    const current = { semantic: 0, ocr: 0, asr: 0, ...(draft.weights || {}) };
    const others = Object.keys(enabled).filter((name) => name !== key && enabled[name]);
    const remainder = Math.max(0, 1 - nextValue);
    const otherTotal = others.reduce((sum, name) => sum + Number(current[name] || 0), 0);
    others.forEach((name) => { current[name] = otherTotal > 0 ? remainder * current[name] / otherTotal : remainder / Math.max(1, others.length); });
    current[key] = nextValue;
    Object.keys(enabled).forEach((name) => { if (!enabled[name]) current[name] = 0; });
    setDraft((prev) => ({ ...prev, weights: current }));
  }

  function handleSave() {
    onSave?.(draft);
    onClose?.();
  }

  const hasAnyMethod =
    draft.semanticModels.length > 0 || draft.useOcr || draft.useAsr;
  const radarWeights = draft.weights || { semantic: 0, ocr: 0, asr: 0 };
  const radarAxes = [["semantic", 0], ["ocr", 120], ["asr", 240]];
  const radarPoint = (key, angle) => {
    const value = Math.max(0, Math.min(1, Number(radarWeights[key] || 0)));
    const radius = 34 * value;
    const radians = (angle - 90) * Math.PI / 180;
    return `${50 + Math.cos(radians) * radius},${50 + Math.sin(radians) * radius}`;
  };
  const radarHandle = (key, angle) => radarPoint(key, angle).split(",").map(Number);

  function dragRadarWeight(key, angle, event) {
    if (!radarRef.current) return;
    const enabled = key === "semantic" ? draft.semanticModels.length > 0 : key === "ocr" ? draft.useOcr : draft.useAsr;
    if (!enabled) return;
    const rect = radarRef.current.getBoundingClientRect();
    const x = (event.clientX - rect.left) * 100 / rect.width - 50;
    const y = (event.clientY - rect.top) * 100 / rect.height - 50;
    const radians = (angle - 90) * Math.PI / 180;
    updateWeight(key, Math.max(0, Math.min(1, (x * Math.cos(radians) + y * Math.sin(radians)) / 34)));
  }

  const selectedKeys = draft.semanticModels.map((item) => item.key);
  const localModel = modelRoles?.local || models[0] || "Local";
  const globalModel = modelRoles?.global || models[1] || "Global";
  const canBlendSemantic = selectedKeys.length === 2 && selectedKeys.includes(localModel) && selectedKeys.includes(globalModel);

  return createPortal(
    <div
      className={`fusion-modal-backdrop fusion-modal-backdrop--${theme}`}
      onClick={onClose}
    >
      <div className="fusion-modal" onClick={(e) => e.stopPropagation()}>
        <div className="fusion-modal-header">
          <div className="fusion-modal-header-icon">
            <GitMerge size={18} />
          </div>

          <div className="fusion-modal-header-text">
            <h2>Fusion search — cấu hình</h2>
            <p>Semantic được trộn theo similarity; các modality được điều chỉnh trực tiếp trên radar.</p>
          </div>

          <button className="modal-close-btn" type="button" onClick={onClose} aria-label="Đóng">
            <X size={18} />
          </button>
        </div>

        <div className="fusion-modal-body">
          {/* ── Semantic models checklist ─────────────────────────── */}
          <div className="fusion-card">
            <div className="fusion-card-title">
              <Sparkles size={13} />
              <span>Semantic models</span>
            </div>

            {models.length === 0 && (
              <p className="fusion-empty-hint">Không có model semantic nào khả dụng.</p>
            )}

            <div className="fusion-row-list">
              {models.map((modelKey) => {
                const checked = draft.semanticModels.some((m) => m.key === modelKey);
                return (
                  <MethodRow
                    key={modelKey}
                    label={modelKey}
                    checked={checked}
                    onToggle={() => toggleModel(modelKey)}
                  />
                );
              })}
            </div>
          </div>

          <div className="fusion-card">
            <div className="fusion-card-title"><GitMerge size={13} /><span>Fusion weights</span></div>
            <p className="fusion-weight-total">Kéo trực tiếp các núm trên radar; tổng nguồn đang bật được chuẩn hoá = 1.0.</p>
            <div className="fusion-radar-wrap" aria-label="Biểu đồ trọng số fusion">
              <svg ref={radarRef} viewBox="0 0 100 100" className="fusion-radar">
                <g className="fusion-radar-grid" aria-hidden="true">
                  {[1, 0.75, 0.5, 0.25].map((level) => (
                    <polygon
                      key={level}
                      points={radarAxes.map(([, angle]) => {
                        const radians = (angle - 90) * Math.PI / 180;
                        return `${50 + Math.cos(radians) * 34 * level},${50 + Math.sin(radians) * 34 * level}`;
                      }).join(" ")}
                      className={level === 1 ? "fusion-radar-grid-ring is-outer" : "fusion-radar-grid-ring"}
                    />
                  ))}
                  {radarAxes.map(([key, angle]) => {
                    const radians = (angle - 90) * Math.PI / 180;
                    return <line key={key} x1="50" y1="50" x2={50 + Math.cos(radians) * 34} y2={50 + Math.sin(radians) * 34} className="fusion-radar-grid-axis" />;
                  })}
                </g>
                <polygon points={`${radarPoint("semantic", 0)} ${radarPoint("ocr", 120)} ${radarPoint("asr", 240)}`} className="fusion-radar-value" />
                {radarAxes.map(([key, angle]) => {
                  const [cx, cy] = radarHandle(key, angle);
                  return <circle key={key} cx={cx} cy={cy} r="3" className="fusion-radar-handle" onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); dragRadarWeight(key, angle, event); }} onPointerMove={(event) => { if (event.currentTarget.hasPointerCapture(event.pointerId)) dragRadarWeight(key, angle, event); }} />;
                })}
                <text x="50" y="7"><tspan x="50">Semantic</tspan><tspan x="50" dy="5">{Number(radarWeights.semantic || 0).toFixed(2)}</tspan></text>
                <text x="91" y="72"><tspan x="91">OCR</tspan><tspan x="91" dy="5">{Number(radarWeights.ocr || 0).toFixed(2)}</tspan></text>
                <text x="9" y="72"><tspan x="9">ASR</tspan><tspan x="9" dy="5">{Number(radarWeights.asr || 0).toFixed(2)}</tspan></text>
              </svg>
            </div>
            <label className="semantic-blend-row">
              <span>Local · {localModel}</span>
              <input type="range" min="0" max="1" step="0.01" disabled={!canBlendSemantic} value={Number(draft.semanticLambda ?? 0.5)} onChange={(e) => updateField("semanticLambda", Number(e.target.value))} />
              <span>Global · {globalModel}</span>
              <output>{canBlendSemantic ? Number(draft.semanticLambda ?? 0.5).toFixed(2) : "1 model"}</output>
            </label>
          </div>

          {/* ── Method khác ────────────────────────────────────────── */}
          <div className="fusion-card">
            <div className="fusion-card-title">
              <Waves size={13} />
              <span>Method khác</span>
            </div>

            <div className="fusion-row-list">
              <MethodRow
                icon={<Waves size={14} />}
                label="Temporal"
                sublabel="Chuỗi sự kiện — mỗi event trộn semantic theo similarity rồi ghép chuỗi"
                checked={draft.temporal}
                onToggle={() => updateField("temporal", !draft.temporal)}
              />

              {draft.temporal && (
                <label className="fusion-duration-field">
                  <span>Duration limit (giây, -1 = không giới hạn)</span>
                  <input
                    type="number"
                    min={-1}
                    step={1}
                    value={draft.durationLimit}
                    onChange={(e) => updateField("durationLimit", Number(e.target.value))}
                  />
                </label>
              )}

              <MethodRow
                icon={<ScanText size={14} />}
                label="OCR"
                sublabel="Chữ xuất hiện trên màn hình"
                checked={draft.useOcr}
                onToggle={() => updateField("useOcr", !draft.useOcr)}
              />

              <MethodRow
                icon={<AudioLines size={14} />}
                label="ASR"
                sublabel="Lời thoại / giọng nói"
                checked={draft.useAsr}
                onToggle={() => updateField("useAsr", !draft.useAsr)}
              />
            </div>
          </div>

          {!hasAnyMethod && (
            <p className="fusion-empty-hint fusion-empty-hint--warn">
              Cần tick ít nhất 1 model semantic, hoặc bật OCR/ASR để fusion search chạy được.
            </p>
          )}
        </div>

        <div className="fusion-modal-footer">
          <button type="button" className="fusion-cancel-btn" onClick={onClose}>
            Huỷ
          </button>
          <button
            type="button"
            className="fusion-save-btn"
            onClick={handleSave}
            disabled={!hasAnyMethod}
          >
            Lưu cấu hình
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
}

function MethodRow({ icon, label, sublabel, checked, onToggle }) {
  return (
    <div className={["fusion-method-row", checked ? "is-checked" : ""].filter(Boolean).join(" ")}>
      <button
        type="button"
        className="fusion-method-toggle"
        onClick={onToggle}
        aria-pressed={checked}
      >
        <span className={["switch-control", checked ? "checked" : ""].filter(Boolean).join(" ")}>
          <span />
        </span>

        <span className="fusion-method-label">
          {icon && <span className="fusion-method-icon">{icon}</span>}
          <span className="fusion-method-text">
            <span className="fusion-method-name">{label}</span>
            {sublabel && <span className="fusion-method-sub">{sublabel}</span>}
          </span>
        </span>
      </button>
    </div>
  );
}
