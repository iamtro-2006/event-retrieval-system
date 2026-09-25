# React + Vite

## Media URLs for AIC and CAM

The frontend asks `VITE_API_BASE_URL` for search results and uses the result's
`image_rel_path`, `video_rel_path`, and `map_rel_path` to build media URLs. The
three media types can be served from separate HTTP servers. For example, with
keyframes and maps on your computer and videos on another host:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_KEYFRAMES_BASE_URL=http://127.0.0.1:3174/{DATASET}/keyframes
VITE_VIDEOS_BASE_URL=https://videos.example.org/{DATASET}/videos
VITE_MAP_KEYFRAMES_BASE_URL=http://127.0.0.1:3174/{DATASET}/map-keyframes
```

`{DATASET}` expands to `AIC` or `CAM`; `{dataset}` expands to lowercase. Each
URL must point to the directory that contains the relative paths returned by
the API. For example, `image_rel_path=L21/L21_V001/000001.jpg` becomes
`http://127.0.0.1:3174/AIC/keyframes/L21/L21_V001/000001.jpg` in this setup.
The browser needs an HTTP URL for local files; a Windows filesystem path in
`.env` is not sufficient. The video host must serve the file name and extension
reported by `video_rel_path` and support seeking through HTTP range requests.

When one dataset uses a different host or folder layout, set the corresponding
override, such as `VITE_CAM_VIDEOS_BASE_URL=https://cam.example.org/videos`.
The resolution order for each asset is: dataset override, shared URL for that
media type, legacy `VITE_DATA_BASE_URL`, then the API-provided URL. Leaving one
media URL empty does not affect the others. Restart Vite after changing `.env`.

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
