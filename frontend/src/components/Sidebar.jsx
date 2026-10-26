import { ChevronLeft, ChevronRight, History, MessageSquare, Moon, Plus, Search, Settings, Sparkles, Sun, Video } from "lucide-react";
import { useVideoFilter } from "./VideoFilter";

export default function Sidebar({ theme, expanded, onToggleExpanded, onReset, onOpenSettings,
  onOpenPreview, onToggleTheme, onFocusComposer, onOpenChat, chatOpen, chatAvailable, onCloseChat, children }) {
  const videoFilter = useVideoFilter();
  return <aside className={`sidebar ${expanded ? "is-expanded" : ""} ${chatOpen ? "is-chat-open" : ""}`}>
    <div className="sidebar-top">
      <button className="sidebar-button" type="button" aria-label="Video ID cache history" title="Video ID cache"
        onClick={() => videoFilter?.setOpen(!videoFilter.open)}><History size={20} />{expanded && <span>Video cache</span>}</button>
      <div className="sidebar-brand-row"><button type="button" className="logo-button brand-variant is-blue" aria-label="New search and change brand icon"
        title="New search" onClick={onReset}><Sparkles size={24} /></button>
        {expanded && <div className="sidebar-brand-copy"><strong>VIRETA</strong><span>Video Retrieval</span></div>}</div>
      <button className="sidebar-button sidebar-expand-button" type="button" onClick={chatOpen ? onCloseChat : onToggleExpanded}
        aria-label={chatOpen ? "Back to navigation" : expanded ? "Collapse sidebar" : "Expand sidebar"}>{expanded ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}{expanded && <span>{chatOpen ? "Back to menu" : "Collapse"}</span>}</button>
      <button className="sidebar-button" type="button" onClick={onReset} aria-label="New search" title="New search"><Plus size={20} />{expanded && <span>New search</span>}</button>
      <button className="sidebar-button" type="button" onClick={onFocusComposer} aria-label="Focus main search" title="Main search"><Search size={20} />{expanded && <span>Main search</span>}</button>
      {chatAvailable && <button className="sidebar-button" type="button" onClick={onOpenChat} aria-label="Open assistant log" title="Assistant log"><MessageSquare size={20} />{expanded && <span>Assistant log</span>}</button>}
      <button className="sidebar-button" type="button" onClick={onOpenPreview} aria-label="Preview video" title="Preview video"><Video size={20} />{expanded && <span>Preview video</span>}</button>
    </div>
    {children}
    <div className="sidebar-bottom">
      <button className="sidebar-button" type="button" onClick={onOpenSettings} title="Settings"><Settings size={20} />{expanded && <span>Settings</span>}</button>
      <button className="sidebar-button" type="button" onClick={onToggleTheme} title="Toggle theme">{theme === "dark" ? <Moon size={20} /> : <Sun size={20} />}{expanded && <span>Theme</span>}</button>
      <div className="avatar">T</div>
    </div>
  </aside>;
}
