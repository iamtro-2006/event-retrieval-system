import {
  AudioLines, Blend, ChevronLeft, ChevronRight, Clock3, Flame, Gem, Menu, Moon,
  ImagePlus, Orbit, Plus, ScanText, Search, Settings, Sparkles, Sun, Trash2, Video, X,
  WandSparkles,
} from "lucide-react";
import { useState } from "react";

const SEARCH_MODES = [
  { key: "text", label: "Semantic", hint: "Visual meaning", icon: Search },
  { key: "temporal", label: "Temporal", hint: "Ordered events", icon: Clock3 },
  { key: "auto", label: "Auto", hint: "Detect query intent", icon: WandSparkles },
  { key: "ocr", label: "OCR", hint: "On-screen text", icon: ScanText },
  { key: "asr", label: "ASR", hint: "Spoken content", icon: AudioLines },
  { key: "fusion", label: "Fusion", hint: "Combine sources", icon: Blend },
];

const BRAND_VARIANTS = [
  { icon: Sparkles, className: "is-blue", label: "Spark" },
  { icon: Orbit, className: "is-violet", label: "Orbit" },
  { icon: Flame, className: "is-coral", label: "Flame" },
  { icon: Gem, className: "is-emerald", label: "Gem" },
];

const getModeLabel = (mode) => SEARCH_MODES.find((item) => item.key === mode)?.label ?? "Semantic";
const getConnector = (mode) => mode === "temporal" || mode === "auto" ? "THEN" : "AND";

export default function Sidebar({
  theme, mode, queryClauses, clauseImages = [[]], useSplit = true, expanded, loading, disabled, onToggleExpanded,
  onModeChange, onClausesChange, onAddClauseImages, onRemoveClauseImage, onSearch, onToggleTheme, onReset,
  onOpenSettings, onOpenPreview,
}) {
  const clauses = queryClauses?.length ? queryClauses : [""];
  const [brandVariant, setBrandVariant] = useState(0);
  const activeBrand = BRAND_VARIANTS[brandVariant];
  const BrandIcon = activeBrand.icon;

  function handleBrandClick() {
    setBrandVariant((current) => (current + 1) % BRAND_VARIANTS.length);
    onReset?.();
  }

  const updateClause = (index, value) => onClausesChange?.(
    clauses.map((clause, clauseIndex) => clauseIndex === index ? value : clause)
  );
  const addClause = () => onClausesChange?.([...clauses, ""]);
  const removeClause = (index) => onClausesChange?.(
    clauses.length === 1 ? [""] : clauses.filter((_, clauseIndex) => clauseIndex !== index),
    { removedIndex: index }
  );
  const imageQueryPosition = (clauseIndex, imageIndex) => {
    if (!useSplit) return 1;
    let position = 1;
    for (let index = 0; index < clauseIndex; index += 1) {
      position += (clauses[index]?.trim() ? 1 : 0) + (clauseImages[index]?.length ?? 0);
    }
    return position + (clauses[clauseIndex]?.trim() ? 1 : 0) + imageIndex;
  };

  return (
    <aside className={`sidebar ${expanded ? "is-expanded" : ""}`}>
      <div className="sidebar-top">
        <div className="sidebar-brand-row">
          <button
            className={`logo-button brand-variant ${activeBrand.className}`}
            onClick={handleBrandClick}
            title={`New search · ${activeBrand.label}`}
            aria-label="New search and change brand icon"
          >
            <BrandIcon size={24} />
          </button>
          {expanded && <div className="sidebar-brand-copy"><strong>VIRETA</strong><span>Video Retrieval</span></div>}
        </div>

        <div className="mode-menu-host">
          <button className="sidebar-button" aria-label="Choose search mode" aria-haspopup="menu" title="Search modes">
            <Menu size={20} />
            {expanded && <span>Search modes</span>}
          </button>
          {!expanded && (
            <div className="mode-hover-card" role="menu">
              <p className="mode-hover-eyebrow">SEARCH BY</p>
              {SEARCH_MODES.map(({ key, label, hint, icon: Icon }) => (
                <button type="button" role="menuitem" key={key}
                  className={`mode-hover-option ${mode === key ? "active" : ""}`}
                  onClick={() => onModeChange?.(key)}>
                  <span className="mode-hover-icon"><Icon size={17} /></span>
                  <span><strong>{label}</strong><small>{hint}</small></span>
                </button>
              ))}
            </div>
          )}
        </div>

        <button className="sidebar-button sidebar-expand-button" onClick={onToggleExpanded}
          aria-label={expanded ? "Collapse sidebar" : "Expand sidebar"}
          title={expanded ? "Collapse sidebar" : "Expand sidebar"}>
          {expanded ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
          {expanded && <span>Collapse panel</span>}
        </button>

        {!expanded && <>
          <button className="sidebar-button" onClick={onReset} title="New search" aria-label="New search"><Plus size={20} /></button>
          <button className="sidebar-button" onClick={onOpenPreview} title="Preview video" aria-label="Preview video"><Video size={20} /></button>
        </>}

        {expanded && (
          <div className="sidebar-expanded-content">
            <section className="sidebar-section">
              <p className="sidebar-section-label">ACTIVE MODE</p>
              <h2>{getModeLabel(mode)} Search</h2>
              <div className="expanded-mode-list">
                {SEARCH_MODES.map(({ key, label, icon: Icon }) => (
                  <button type="button" key={key} className={mode === key ? "active" : ""}
                    onClick={() => onModeChange?.(key)}>
                    <Icon size={16} /><span>{label}</span>
                  </button>
                ))}
              </div>
            </section>

            <section className="sidebar-section clause-builder">
              <div className="clause-builder-heading">
                <div>
                  <p className="sidebar-section-label">QUERY BUILDER</p>
                  <span>{useSplit
                    ? (mode === "temporal" || mode === "auto" ? "Each text/image is an event" : "Each text/image is a query")
                    : "Text and images are combined as one query"}</span>
                </div>
                <button type="button" onClick={addClause} aria-label="Add query clause" title="Add clause"><Plus size={16} /></button>
              </div>

              <div className="clause-stack">
                {clauses.map((clause, index) => (
                  <div className="clause-block" key={index}>
                    {index > 0 && <div className="clause-connector"><span /><strong>{getConnector(mode)}</strong><span /></div>}
                    <div className="clause-input-shell">
                      <span className="clause-number">{String(index + 1).padStart(2, "0")}</span>
                      <textarea rows={2} value={clause}
                        placeholder={index === 0 ? "Describe the first scene..." : "Describe the next clause..."}
                        onChange={(event) => updateClause(index, event.target.value)}
                        onPaste={(event) => {
                          const images = Array.from(event.clipboardData?.files ?? []).filter((file) => file.type.startsWith("image/"));
                          if (!images.length) return;
                          event.preventDefault();
                          onAddClauseImages?.(index, images);
                        }} />
                      <label className="clause-attach-button" title="Attach image to this clause">
                        <ImagePlus size={14} />
                        <input type="file" accept="image/*" multiple onChange={(event) => {
                          onAddClauseImages?.(index, event.target.files);
                          event.target.value = "";
                        }} />
                      </label>
                      <button type="button" onClick={() => removeClause(index)} aria-label={`Remove clause ${index + 1}`}>
                        <Trash2 size={14} />
                      </button>
                    </div>
                    {(clauseImages[index] ?? []).length > 0 && (
                      <div className="clause-image-strip">
                        {clauseImages[index].map((item, imageIndex) => (
                          <div className="clause-image-chip" key={item.id} title={item.name}>
                            <img src={item.url} alt={item.name} />
                            <span>Q{String(imageQueryPosition(index, imageIndex)).padStart(2, "0")}</span>
                            <button type="button" onClick={() => onRemoveClauseImage?.(index, item.id)} aria-label={`Remove ${item.name}`}>
                              <X size={10} />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <button type="button" className="sidebar-run-search" onClick={onSearch}
                disabled={loading || disabled || (!clauses.some((clause) => clause.trim()) && !clauseImages.some((items) => items.length > 0))}>
                <Search size={17} /><span>{loading ? "Searching..." : "Run search"}</span>
              </button>
            </section>

            <button className="sidebar-preview-wide" type="button" onClick={onOpenPreview}><Video size={17} /> Preview video</button>
          </div>
        )}
      </div>

      <div className="sidebar-bottom">
        <button className="sidebar-button" onClick={onOpenSettings} title="Settings"><Settings size={20} />{expanded && <span>Settings</span>}</button>
        <button className="sidebar-button" onClick={onToggleTheme} title="Toggle theme">
          {theme === "dark" ? <Moon size={20} /> : <Sun size={20} />}
          {expanded && <span>{theme === "dark" ? "Dark theme" : "Light theme"}</span>}
        </button>
        <div className="avatar">T</div>
      </div>
    </aside>
  );
}
