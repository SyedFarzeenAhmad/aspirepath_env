from __future__ import annotations

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


def main() -> None:
    if app is None:
        raise ImportError(
            "OpenEnv server helpers are unavailable. Install project dependencies first."
        )
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
