# God Mode verified-result relay

This service is intentionally independent from the retrieval backend. It submits to DRES,
persists only `correct` verdicts in SQLite, and broadcasts those verified frames to clients
subscribed to the same evaluation ID.

## Start

From `backend/`:

```powershell
pip install -r requirements-godmode.txt
uvicorn godmode_server:app --host 0.0.0.0 --port 8081
```

Optional environment variables:

- `GODMODE_DB_PATH`: SQLite file, default `backend/data/godmode.sqlite3`.
- `GODMODE_MAX_RESULTS`: retained frames per evaluation, default `200`.
- `GODMODE_CORS_ORIGINS`: comma-separated frontend origins; use explicit origins in production.

Configure every frontend client with the same public relay URL:

```dotenv
VITE_GODMODE_ENDPOINT=https://relay.example.com
```

`VITE_SOCKET_URL` is accepted as a backward-compatible alias. Use one variable, not both.

Restart the frontend after changing a `VITE_*` value. In Retrieval Settings, enter the DRES
evaluation ID and enable God Mode. `http(s)` is automatically converted to `ws(s)` for the
live subscription.

## Endpoints

- `GET /health`
- `GET /api/godmode/results?evaluation_id=...`
- `POST /api/godmode/submit`
- `WS /ws?evaluation_id=...`

Deploy behind HTTPS/WSS when clients are not on a trusted LAN. The relay never broadcasts
DRES passwords or session IDs. Apply normal reverse-proxy rate limits and restrict CORS for
competition deployment.
