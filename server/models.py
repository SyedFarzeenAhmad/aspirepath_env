from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action as OpenEnvActionBase
    from openenv.core.env_server.types import Observation as OpenEnvObservationBase
except ImportError:
    try:
        from openenv_core.env_server.types import Action as OpenEnvActionBase
        from openenv_core.env_server.types import Observation as OpenEnvObservationBase
    except ImportError:
        class OpenEnvActionBase(BaseModel):
            metadata: Dict[str, Any] = Field(default_factory=dict)

        class OpenEnvObservationBase(BaseModel):
            done: bool = False
            reward: Optional[float] = None
            metadata: Dict[str, Any] = Field(default_factory=dict)


AcademicStream = Literal["PCM", "PCB", "Commerce", "Humanities"]
CareerCluster = Literal["STEM", "Business", "Arts", "Healthcare"]


class Observation(OpenEnvObservationBase):
    student_id: str = Field(..., description="Unique task identifier such as S1, S2, or S3.")
    analytical_score: float = Field(..., ge=0.0, le=1.0)
    creative_score: float = Field(..., ge=0.0, le=1.0)
    verbal_score: float = Field(..., ge=0.0, le=1.0)
    interests: List[str] = Field(default_factory=list)
    task_description: str = Field(..., min_length=1)


class Action(OpenEnvActionBase):
    recommended_stream: AcademicStream = Field(...)
    career_cluster: CareerCluster = Field(...)
    justification: str = Field(..., min_length=8)


class Reward(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
