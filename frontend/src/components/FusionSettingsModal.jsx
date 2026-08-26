import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { X, GitMerge, Sparkles, Waves, ScanText, AudioLines } from "lucide-react";

/**
 * FusionSettingsModal — pop-up cấu hình cho search mode "fusion" (advanced
 * search).
 *
 * Cho phép tick nhiều semantic model, bật/tắt temporal (kèm duration
 * limit), OCR, ASR và điều chỉnh weight theo nhóm nguồn. Backend
 * Khi temporal bật, nó chạy multimodal PER EVENT:
 * mỗi event tự fuse (bằng RRF) đúng các method đã tick ở trên (model(s) +
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
  models = [],
  value,
  onClose,
  onSave,
}) {
  const [draft, setDraft] = useState(value);

  useEffect(() => {
    if (open) setDraft(value);
  }, [open, value]);

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
  const radarPoint = (key, angle) => {
    const value = Math.max(0, Math.min(1, Number(radarWeights[key] || 0)));
    const radius = 34 * value;
    const radians = (angle - 90) * Math.PI / 180;
    return `${50 + Math.cos(radians) * radius},${50 + Math.sin(radians) * radius}`;
  };

  return createPortal(
    <div className="fusion-modal-backdrop" onClick={onClose}>
      <div className="fusion-modal" onClick={(e) => e.stopPropagation()}>
        <div className="fusion-modal-header">
          <div className="fusion-modal-header-icon">
            <GitMerge size={18} />
          </div>

          <div className="fusion-modal-header-text">
            <h2>Fusion search — cấu hình</h2>
            <p>Chọn model và method muốn kết hợp — kết quả được fuse theo rank (RRF), không cần chỉnh trọng số.</p>
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
            <p className="fusion-weight-total">Tổng nguồn đang bật luôn được chuẩn hoá = 1.0</p>
            <div className="fusion-radar-wrap" aria-label="Biểu đồ trọng số fusion">
              <svg viewBox="0 0 100 100" className="fusion-radar">
                <polygon points="50,16 84,68 16,68" className="fusion-radar-grid" />
                <polygon points={`${radarPoint("semantic", 0)} ${radarPoint("ocr", 120)} ${radarPoint("asr", 240)}`} className="fusion-radar-value" />
                <text x="50" y="10">Semantic</text><text x="87" y="73">OCR</text><text x="3" y="73">ASR</text>
              </svg>
            </div>
            {[["semantic", "Semantic", draft.semanticModels.length > 0], ["ocr", "OCR", draft.useOcr], ["asr", "ASR", draft.useAsr]].map(([key, label, enabled]) => (
              <label className="fusion-weight-row" key={key}>
                <span>{label}</span>
                <input type="range" min="0" max="1" step="0.01" disabled={!enabled} value={Number(draft.weights?.[key] ?? 0)} onChange={(e) => updateWeight(key, e.target.value)} />
                <input className="fusion-weight-number" type="number" min="0" max="1" step="0.01" disabled={!enabled} value={Number(draft.weights?.[key] ?? 0).toFixed(2)} onChange={(e) => updateWeight(key, e.target.value)} />
              </label>
            ))}
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
                sublabel="Chuỗi sự kiện — mỗi event tự fuse (RRF) các method đã tick ở trên rồi ghép chuỗi"
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
