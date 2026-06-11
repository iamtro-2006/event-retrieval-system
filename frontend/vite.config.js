import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Cấu hình bắt buộc cho Vite khi chạy qua proxy/tunnel như ngrok
    allowedHosts: [
      "exemplifiable-gauntly-naomi.ngrok-free.dev", // Điền chính xác domain ngrok hiện tại của bạn
      ".ngrok-free.dev"                             // Hoặc thêm pattern này để cho phép mọi subdomain của ngrok
    ],
    // Hoặc nếu muốn nhanh gọn cho phép tất cả các host trong môi trường dev:
    // allowedHosts: "all"
  }
});