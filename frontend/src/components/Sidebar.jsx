import { Moon, Sun, Plus, Menu, Sparkles, Settings } from "lucide-react";

export default function Sidebar({
  theme,
  onToggleTheme,
  onReset,
  onOpenSettings,
}) {
  return (
    <aside className="sidebar">
      <div className="sidebar-top">
        <button className="logo-button" onClick={onReset}>
          <Sparkles size={24} />
        </button>

        <button className="sidebar-button">
          <Menu size={20} />
        </button>

        <button className="sidebar-button" onClick={onReset}>
          <Plus size={20} />
        </button>
      </div>

      <div className="sidebar-bottom">
        <button className="sidebar-button" onClick={onOpenSettings}>
          <Settings size={20} />
        </button>

        <button className="sidebar-button" onClick={onToggleTheme}>
          {theme === "dark" ? <Moon size={20} /> : <Sun size={20} />}
        </button>

        <div className="avatar">T</div>
      </div>
    </aside>
  );
}