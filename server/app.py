from __future__ import annotations

import os

import uvicorn

from .app_logic import AspirePathEnv
from .models import Action, Observation

try:
    from openenv.core.env_server import create_app
except ImportError:
    try:
        from openenv_core.env_server import create_app
    except ImportError:
        create_app = None


def build_app():
    if create_app is None:
        raise ImportError(
            "OpenEnv server helpers are unavailable. Install project dependencies first."
        )
    return create_app(
        env=AspirePathEnv,
        action_cls=Action,
        observation_cls=Observation,
        env_name="aspirepath-v1",
    )


app = build_app() if create_app is not None else None

if app is not None:
    @app.get("/")
    def root() -> dict[str, str]:
        return {"status": "ok", "message": "AspirePath environment server is running."}

    # If a Gradio demo is available, mount it at the root path.
    try:
        import gradio as gr  # type: ignore[import-not-found]
        from .gradio_app import demo  # type: ignore[import-not-found]
    except Exception:
        gr = None
        demo = None

    if gr is not None and demo is not None:
        app = gr.mount_gradio_app(app, demo, path="/")


def main() -> None:
    if app is None:
        raise ImportError(
            "OpenEnv server helpers are unavailable. Install project dependencies first."
        )
    host = os.getenv("HOST", "0.0.0.0")
    # README sets app_port to 8000 for this Space.
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
