# Frontend

Test backend mock và chat Codex: chạy mock server cổng 8001 rồi `npm run dev:mock`. Bật **Reasoning** ở thanh search để chat với Codex; model và mức suy luận chọn trong cùng popup. **Assistant log** nằm trong sidebar trái. Xem [hướng dẫn](../backend/docs/MOCK_SERVER.md). Chế độ này dùng `.env.mock`, giữ nguyên `.env`.

Giao diện React 19 + Vite 8 cho tìm kiếm video V3C và LHE. Cần Node.js 22.12+ (hoặc Node 24). Dependency nằm trong `package.json` và `package-lock.json`; frontend không dùng `requirements.txt` Python.

## Thanh tìm kiếm và assistant

- Thanh search luôn mở rộng, cao tối thiểu 156 px; cùng style ở trang chủ và trang kết quả.
- Reasoning tắt: chọn retrieval model và search mode. Reasoning bật: dùng Codex, chỉ giữ toggle và popup model/effort; không có Account default hoặc chọn số worker trên UI.
- Model, mức mặc định và các mức effort hợp lệ lấy từ backend. Worker dùng cùng model/effort với coordinator; frontend đặt giới hạn 3 worker nội bộ.
- Assistant log thay thế nội dung menu khi mở sidebar trái; Back to menu/Close quay về menu và giữ lịch sử. Chỉ thanh search chính có nhập/gửi/dừng; nội dung assistant và worker được render Markdown.
- Trang kết quả cho ảnh cuộn phía sau composer nổi. Composer cách đáy 18 px, mờ nhẹ khi không tương tác (opacity 0.68), hiện rõ khi hover/focus/chạy hoặc mở popup. Hiệu ứng đang xử lý là glow tím; Stop màu đỏ. Queue màu xanh ở góc phải ngoài.
- Style composer hiện tại nằm ở `SearchBar.css`; log ở `AgentChat.css`, Markdown ở `AssistantMarkdown.css`, popup ở `AgentReasoningPicker.css`. Tránh thêm các lớp ghi đè composer vào `index.css`.

## Chạy và kiểm tra

```powershell
cd frontend
Copy-Item .env.example .env
npm ci
npm run dev
```

Đặt `VITE_API_BASE_URL` trong `.env` trỏ tới backend (mặc định `http://127.0.0.1:8000`). Sửa biến `VITE_*` xong cần khởi động lại Vite. Lệnh có sẵn: `npm run build`, `npm run lint`, `npm run test:ui-mocks`, `npm run test:agent` (stream và chọn model/effort), `npm run build:mock`.

## File tham gia luồng chính

| Vị trí | Vai trò |
| --- | --- |
| `src/main.jsx`, `src/App.jsx` | Mount ứng dụng, state/UI chính, chọn dataset, tìm kiếm và xem kết quả. |
| `src/components/` | Thanh tìm kiếm, bộ lọc, grid/card, detail, các modal media, settings và queue nộp bài. |
| `src/hooks/useRetrievalSearch.js` | State và yêu cầu tìm kiếm; bỏ qua response cũ khi có request mới. |
| `src/config/searchModes.js` | Mode hiển thị và phím tắt: semantic, temporal, auto, OCR, ASR, fusion, color. |
| `src/components/AgentChat.jsx`, `AgentReasoningPicker.jsx`, `AssistantMarkdown.jsx` | Chat, model/effort và Markdown. |
| `src/api/agentAPI.js`, `src/utils/agentModels.js` | NDJSON và chuẩn hóa lựa chọn model/effort. |
| `src/api/retrievalAPI.js` | Gọi backend, chuẩn hóa kết quả và giải URL media. |
| `src/api/submissionAPI.js`, `godmodeAPI.js`, `googleTranslate.js` | DRES, relay God Mode; `googleTranslate.js` là helper dịch riêng. |
| `src/api/mediaUrls.js` | Quy tắc tạo URL keyframe, video, map theo dataset. |
| `src/mocks/searchScenarios.js`, `scripts/validate-mock-scenarios.mjs` | Kịch bản mock và kiểm tra dữ liệu UI. |

Backend phải có index/model và data root phù hợp để search thật; xem [README backend](../backend/README.md).

## Phát video và dải keyframe

`VideoModal` được dùng chung cho kết quả tìm kiếm, Preview video và Queue. Bên dưới
video là dải ảnh `VideoKeyframeStrip` được virtualize; timeline ruler/zoom/range cũ
đã bỏ. Bấm ảnh để tua, khi phát thì ảnh active và dấu sao tự đổi theo thời gian;
ảnh được căn giữa và dừng cuộn tự nhiên ở hai đầu. Nhãn trên ảnh là `keyframe_id`,
khác với decoded `frame_idx` trong header/Copy ID. Các nút seek-step, Exact ms,
Copy ID, Queue, Submit và native controls được giữ lại.

`getVideoKeyframes` lấy metadata/thumbnail từ API, giải URL bằng `mediaUrls` và các
biến host media hiện có. Backend thật lấy từ metadata index của dataset; mock lấy
từ CSV trong cùng data root và gắn cờ rõ khi fallback sang timing giả lập. FE không
đọc đường dẫn ổ đĩa hay CSV trực tiếp. `npm run test:video` kiểm tra chọn keyframe,
căn giữa ở hai đầu, danh tính keyframe và URL ảnh qua host media.

## Biến môi trường và media

Sao chép [.env.example](.env.example) rồi đặt các URL cần dùng:

`.env` là cấu hình riêng của máy, được `.gitignore` bỏ qua; không commit tài khoản
hoặc URL tunnel cá nhân. `.env.example` và `.env.mock` vẫn được đưa vào Git để
máy khác có thể cấu hình và chạy mock. Việc ngừng theo dõi `.env` không xóa các
phiên bản của file đã tồn tại trong lịch sử Git.

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_KEYFRAMES_BASE_URL=http://127.0.0.1:8000/keyframes
VITE_VIDEOS_BASE_URL=http://127.0.0.1:8000/videos
VITE_MAP_KEYFRAMES_BASE_URL=http://127.0.0.1:8000/mapkeyframes
```

Ba base URL có thể ở ba host khác nhau. Mỗi URL trỏ tới thư mục chứa folder dataset; frontend tự ghép `/V3C/` hoặc `/LHE/` rồi tới relative path API trả về (`image_rel_path`, `video_rel_path`, `map_rel_path`). Local API phục vụ `data/keyframes/{V3C,LHE}/`, `data/videos/{V3C,LHE}/`, `data/map_keyframes/{V3C,LHE}/` qua `/keyframes/`, `/videos/`, `/mapkeyframes/` khi các thư mục đó tồn tại lúc khởi động. Ví dụ `data/keyframes/V3C/00001/000001.jpg` có URL `http://127.0.0.1:8000/keyframes/V3C/00001/000001.jpg`. Không dùng đường dẫn ổ đĩa Windows trong trình duyệt. Server video cần phục vụ đúng tên/đuôi file và hỗ trợ HTTP range để tua.

Nếu một base URL để trống, frontend dùng URL localhost do backend trả về cho loại media đó. Ba biến này chọn host theo **loại media**; host được chọn phải chứa dataset đang xem. Cấu hình không tự thử host khác khi server trả 404. Các biến DRES (`VITE_SUBMISSION`, `VITE_TEAM_ID`, `VITE_TEAM_PASSWORD`, `VITE_DRES_EVALUATION_ID`), relay (`VITE_GODMODE_ENDPOINT`), dịch (`VITE_GOOGLE_TRANSLATE`) và debug nằm trong `.env.example`. Mọi giá trị `VITE_*` được đưa vào bundle trình duyệt, nên không đặt bí mật dài hạn ở đó.
