import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { X, GitMerge, Sparkles, Waves, ScanText, AudioLines } from "lucide-react";

/**
 * FusionSettingsModal — pop-up cấu hình cho search mode "fusion".
 *
 * Cho phép tick nhiều semantic model (mỗi model 1 weight riêng), bật/tắt
 * temporal (kèm duration limit + weight), OCR (weight), ASR (weight).
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
          : [...prev.semanticModels, { key: modelKey, weight: 1 }],
      };
    });
  }

  function setModelWeight(modelKey, weight) {
    setDraft((prev) => ({
      ...prev,
      semanticModels: prev.semanticModels.map((m) =>
        m.key === modelKey ? { ...m, weight } : m
      ),
    }));
  }

  function updateField(key, val) {
    setDraft((prev) => ({ ...prev, [key]: val }));
  }

  function handleSave() {
    onSave?.(draft);
    onClose?.();
  }

  const hasAnyMethod =
    draft.semanticModels.length > 0 || draft.useOcr || draft.useAsr;

  const totalWeight =
    draft.semanticModels.reduce((sum, m) => sum + (Number(m.weight) || 0), 0) +
    (draft.temporal ? Number(draft.temporalWeight) || 0 : 0) +
    (draft.useOcr ? Number(draft.ocrWeight) || 0 : 0) +
    (draft.useAsr ? Number(draft.asrWeight) || 0 : 0);

  return createPortal(
    <div className="fusion-modal-backdrop" onClick={onClose}>
      <div className="fusion-modal" onClick={(e) => e.stopPropagation()}>
        <div className="fusion-modal-header">
          <div className="fusion-modal-header-icon">
            <GitMerge size={18} />
          </div>

          <div className="fusion-modal-header-text">
            <h2>Fusion search — cấu hình</h2>
            <p>Chọn model, method, và trọng số (weight) cho từng nguồn.</p>
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
                const entry = draft.semanticModels.find((m) => m.key === modelKey);
                const checked = Boolean(entry);
                return (
                  <MethodRow
                    key={modelKey}
                    label={modelKey}
                    checked={checked}
                    onToggle={() => toggleModel(modelKey)}
                    weight={entry?.weight}
                    onWeightChange={(w) => setModelWeight(modelKey, w)}
                  />
                );
              })}
            </div>
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
                sublabel="Chuỗi sự kiện, chạy trên các model đã tick ở trên"
                checked={draft.temporal}
                onToggle={() => updateField("temporal", !draft.temporal)}
                weight={draft.temporal ? draft.temporalWeight : undefined}
                onWeightChange={(w) => updateField("temporalWeight", w)}
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
                weight={draft.useOcr ? draft.ocrWeight : undefined}
                onWeightChange={(w) => updateField("ocrWeight", w)}
              />

              <MethodRow
                icon={<AudioLines size={14} />}
                label="ASR"
                sublabel="Lời thoại / giọng nói"
                checked={draft.useAsr}
                onToggle={() => updateField("useAsr", !draft.useAsr)}
                weight={draft.useAsr ? draft.asrWeight : undefined}
                onWeightChange={(w) => updateField("asrWeight", w)}
              />
            </div>
          </div>

          {!hasAnyMethod ? (
            <p className="fusion-empty-hint fusion-empty-hint--warn">
              Cần tick ít nhất 1 model semantic, hoặc bật OCR/ASR để fusion search chạy được.
            </p>
          ) : (
            <p className="fusion-total-hint">
              Tổng weight hiện tại: <strong>{totalWeight.toFixed(2)}</strong> — không bắt buộc phải bằng 1,
              backend sẽ tự chuẩn hoá khi fuse.
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

function MethodRow({ icon, label, sublabel, checked, onToggle, weight, onWeightChange }) {
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

      {checked && (
        <div className="fusion-weight-control">
          <span>Weight</span>
          <input
            type="number"
            min={0}
            step={0.1}
            value={weight ?? 1}
            onChange={(e) => onWeightChange(Number(e.target.value))}
          />
        </div>
      )}
    </div>
  );
}
