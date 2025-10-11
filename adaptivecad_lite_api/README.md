# AdaptiveCAD-Lite API

FastAPI microservice that exposes the AdaptiveCAD-Lite endpoints used by ChatGPT integrations. The initial release ships with placeholder geometry hooks so the shape pipeline can be validated before wiring in the analytic pi_a kernels.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The service defaults to `http://127.0.0.1:8000`. Navigate to `http://127.0.0.1:8000/docs` for the interactive Swagger explorer or `http://127.0.0.1:8000/openapi.json` for the raw specification.

## Endpoints

- `GET /version` — Build metadata so the client can confirm compatibility.
- `GET /list_shapes` — Returns the available template identifiers.
- `POST /generate_shape` — Generates an artifact (stubbed) and returns a signed asset URL.
- `GET /describe_shape` — Quick curvature and pi_a profile summary for a template.

## Manifest integration

The ChatGPT manifest lives in `.well-known/ai-plugin.json` and references `openapi.yaml`. Update the `api.url` once the service is deployed behind your domain. Keep the manifest name stable when you upgrade the API so ChatGPT can discover the new capabilities automatically.

## Next steps

1. Swap the stubbed URL builder in `main.py` for the real geometry backend (pi_a kernel, ARP field, etc.).
2. Extend `AVAILABLE_SHAPES` with richer metadata as you onboard new templates.
3. Add `/simulate_field` and `/optimize_shape` endpoints in the next release and bump the version to `0.2.0`.
