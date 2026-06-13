import { CheckCircle, XCircle, AlertTriangle, Info } from "lucide-react";

export default function ToastHost({ toasts }) {
  return (
    <div className="toast-host">
      {toasts.map((toast) => (
        <div key={toast.id} className={`toast-card ${toast.type}`}>
          {toast.type === "correct" && <CheckCircle size={18} />}
          {toast.type === "wrong" && <XCircle size={18} />}
          {toast.type === "warning" && <AlertTriangle size={18} />}
          {toast.type === "pending" && <Info size={18} />}

          <div>
            <strong>{toast.title}</strong>
            {toast.message && <p>{toast.message}</p>}
          </div>
        </div>
      ))}
    </div>
  );
}
