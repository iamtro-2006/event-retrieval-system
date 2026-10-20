# Frontend

Giao diện React 19 + Vite 8 cho tìm kiếm video AIC/CAM. Dependency nằm trong `package.json` và `package-lock.json`; frontend không dùng `requirements.txt` Python.

## Chạy và kiểm tra

```powershell
cd frontend
Copy-Item .env.example .env
npm ci
npm run dev
```

Đặt `VITE_API_BASE_URL` trong `.env` trỏ tới backend (mặc định `http://127.0.0.1:8000`). Sửa biến `VITE_*` xong cần khởi động lại Vite. Lệnh có sẵn: `npm run build`, `npm run lint`, `npm run test:ui-mocks`, `npm run test:media-urls`.

## File tham gia luồng chính

| Vị trí | Vai trò |
| --- | --- |
| `src/main.jsx`, `src/App.jsx` | Mount ứng dụng, state/UI chính, chọn dataset, tìm kiếm và xem kết quả. |
| `src/components/` | Thanh tìm kiếm, bộ lọc, grid/card, detail, các modal media, settings và queue nộp bài. |
| `src/hooks/useRetrievalSearch.js` | State và yêu cầu tìm kiếm; bỏ qua response cũ khi có request mới. |
| `src/config/searchModes.js` | Mode hiển thị và phím tắt: semantic, temporal, auto, OCR, ASR, fusion, color. |
| `src/api/retrievalAPI.js` | Gọi backend, chuẩn hóa kết quả và giải URL media. |
| `src/api/submissionAPI.js`, `godmodeAPI.js`, `googleTranslate.js` | DRES, relay God Mode; `googleTranslate.js` là helper dịch riêng. |
| `src/api/mediaUrls.js` | Quy tắc tạo URL keyframe, video, map theo dataset. |
| `src/mocks/searchScenarios.js`, `scripts/validate-mock-scenarios.mjs` | Kịch bản mock và kiểm tra dữ liệu UI. |

Backend phải có index/model và data root phù hợp để search thật; xem [README backend](../backend/README.md).

## Biến môi trường và media

Sao chép [.env.example](.env.example) rồi đặt các URL cần dùng:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_KEYFRAMES_BASE_URL=http://127.0.0.1:8000/keyframes
VITE_VIDEOS_BASE_URL=http://127.0.0.1:8000/videos
VITE_MAP_KEYFRAMES_BASE_URL=http://127.0.0.1:8000/mapkeyframes
```

Ba base URL có thể ở ba host khác nhau. Mỗi URL trỏ tới thư mục chứa các folder dataset; frontend tự ghép `/<DATASET>/` (ví dụ `AIC/`, `CAM/`) rồi tới relative path API trả về (`image_rel_path`, `video_rel_path`, `map_rel_path`). Local API phục vụ `data/keyframes/<DATASET>/`, `data/videos/<DATASET>/`, `data/mapkeyframes/<DATASET>/` qua `/keyframes/`, `/videos/`, `/mapkeyframes/` khi các thư mục đó tồn tại lúc khởi động. Ví dụ `data/keyframes/AIC/L21/L21_V001/000001.jpg` có URL `http://127.0.0.1:8000/keyframes/AIC/L21/L21_V001/000001.jpg`. Không dùng đường dẫn ổ đĩa Windows trong trình duyệt. Server video cần phục vụ đúng tên/đuôi file và hỗ trợ HTTP range để tua.

Nếu một base URL để trống, frontend dùng URL localhost do backend trả về cho loại media đó. Ba biến này chọn host theo **loại media**; host được chọn phải chứa dataset đang xem. Cấu hình không tự thử host khác khi server trả 404. Các biến DRES (`VITE_SUBMISSION`, `VITE_TEAM_ID`, `VITE_TEAM_PASSWORD`, `VITE_DRES_EVALUATION_ID`), relay (`VITE_GODMODE_ENDPOINT`), dịch (`VITE_GOOGLE_TRANSLATE`) và debug nằm trong `.env.example`. Mọi giá trị `VITE_*` được đưa vào bundle trình duyệt, nên không đặt bí mật dài hạn ở đó.
