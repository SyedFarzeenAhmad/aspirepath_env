from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import uuid4

from .models import Action, Observation, Reward, State

try:
    from openenv.core.env_server.interfaces import Environment as OpenEnv
    from openenv.core.env_server.types import EnvironmentMetadata
except ImportError:
    try:
        from openenv_core.env_server.interfaces import Environment as OpenEnv
        from openenv_core.env_server.types import EnvironmentMetadata
    except ImportError:
        class OpenEnv:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

        class EnvironmentMetadata:  # type: ignore[no-redef]
            def __init__(self, name: str, description: str, version: str) -> None:
                self.name = name
                self.description = description
                self.version = version


@dataclass(frozen=True)
class TaskDefinition:
    key: str
    student_id: str
    analytical_score: float
    creative_score: float
    verbal_score: float
    interests: tuple[str, ...]
    target_stream: str
    target_cluster: str
    task_description: str
    rationale_keywords: tuple[str, ...]


class AspirePathEnv(OpenEnv):
    env_id = "aspirepath-v1"
    version = "1.0.0"

    VALID_STREAMS = {
        "pcm": "PCM",
        "pcb": "PCB",
        "commerce": "Commerce",
        "humanities": "Humanities",
    }
    VALID_CLUSTERS = {
        "stem": "STEM",
        "business": "Business",
        "arts": "Arts",
        "healthcare": "Healthcare",
    }
    MIN_VALIDATOR_SCORE = 0.01
    MAX_VALIDATOR_SCORE = 0.99

    def __init__(self, default_task: str = "easy", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tasks: Dict[str, TaskDefinition] = {
            "easy": TaskDefinition(
                key="easy",
                student_id="S1",
                analytical_score=0.9,
                creative_score=0.2,
                verbal_score=0.4,
                interests=("Mathematics", "Coding", "Problem Solving"),
                target_stream="PCM",
                target_cluster="STEM",
                task_description=(
                    "Student S1 has very high analytical ability and should be guided "
                    "toward the most suitable stream and career cluster."
                ),
                rationale_keywords=("analytical", "logic", "math", "coding", "problem solving"),
            ),
            "medium": TaskDefinition(
                key="medium",
                student_id="S2",
                analytical_score=0.4,
                creative_score=0.9,
                verbal_score=0.6,
                interests=("Design", "Writing", "Visual Storytelling"),
                target_stream="Humanities",
                target_cluster="Arts",
                task_description=(
                    "Student S2 has very high creative ability and should be guided "
                    "toward the most suitable stream and career cluster."
                ),
                rationale_keywords=("creative", "design", "writing", "expression", "arts"),
            ),
            "hard": TaskDefinition(
                key="hard",
                student_id="S3",
                analytical_score=0.6,
                creative_score=0.5,
                verbal_score=0.9,
                interests=("Corporate Law", "Debate", "Public Speaking"),
                target_stream="Commerce",
                target_cluster="Business",
                task_description=(
                    "Student S3 has high verbal ability with a strong law interest and "
                    "should be guided toward the most suitable stream and career cluster."
                ),
                rationale_keywords=("verbal", "law", "corporate law", "debate", "commerce"),
            ),
        }
        if default_task not in self._tasks:
            default_task = "easy"
        self._current_task_key = default_task
        self._state = State(
            episode_id=self._new_episode_id(),
            current_task=default_task,
        )
        self._current_observation: Optional[Observation] = self._build_observation(
            task=self._tasks[default_task],
            reward=None,
            reasoning=None,
        )

    @property
    def state(self) -> State:
        return self._state

    def available_tasks(self) -> tuple[str, ...]:
        return tuple(self._tasks.keys())

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        del seed, kwargs
        selected_task = task_name or self._current_task_key
        if selected_task not in self._tasks:
            raise ValueError(
                f"Unknown task '{selected_task}'. Valid tasks: {', '.join(self.available_tasks())}"
            )

        self._current_task_key = selected_task
        task = self._tasks[selected_task]
        self._state = State(
            episode_id=episode_id or self._new_episode_id(),
            current_task=task.key,
            step_count=0,
            done=False,
            last_action=None,
            last_reward=None,
            last_reasoning=None,
        )
        self._current_observation = self._build_observation(task=task, reward=None, reasoning=None)
        return self._current_observation

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        del timeout_s, kwargs
        if self._current_observation is None:
            raise RuntimeError("Environment has not been reset. Call reset() before step().")

        task = self._tasks[self._current_task_key]
        if self._state.done:
            reward = Reward(score=self.MIN_VALIDATOR_SCORE)
            reasoning = "episode already completed; reset before taking another step"
            self._state.last_action = action
            self._state.last_reward = reward
            self._state.last_reasoning = reasoning
            self._current_observation = self._build_observation(
                task=task,
                reward=reward,
                reasoning=reasoning,
            )
            return self._current_observation

        reward, reasoning = self._grade_action(task, action)

        self._state.step_count += 1
        self._state.done = True
        self._state.last_action = action
        self._state.last_reward = reward
        self._state.last_reasoning = reasoning

        self._current_observation = self._build_observation(
            task=task,
            reward=reward,
            reasoning=reasoning,
        )
        return self._current_observation

    def close(self) -> None:
        self._current_observation = None

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name=self.env_id,
            description="Grade 10 career counseling environment with dense reward shaping.",
            version=self.version,
        )

    def _build_observation(
        self,
        task: TaskDefinition,
        reward: Optional[Reward],
        reasoning: Optional[str],
    ) -> Observation:
        return Observation(
            student_id=task.student_id,
            analytical_score=task.analytical_score,
            creative_score=task.creative_score,
            verbal_score=task.verbal_score,
            interests=list(task.interests),
            task_description=task.task_description,
            done=self._state.done,
            reward=reward.score if reward else None,
            metadata={
                "task_name": task.key,
                "target_stream": task.target_stream,
                "target_cluster": task.target_cluster,
                "reward_reasoning": reasoning,
            },
        )

    def _grade_action(self, task: TaskDefinition, action: Action) -> tuple[Reward, str]:
        normalized_stream = self._normalize_value(action.recommended_stream, self.VALID_STREAMS)
        normalized_cluster = self._normalize_value(action.career_cluster, self.VALID_CLUSTERS)
        justification = action.justification.strip()
        justification_lower = justification.lower()

        score = 0.0
        reasoning_parts: list[str] = []

        if normalized_stream:
            score += 0.05
            reasoning_parts.append("valid stream format")
        else:
            reasoning_parts.append("invalid stream format")

        if normalized_cluster:
            score += 0.05
            reasoning_parts.append("valid cluster format")
        else:
            reasoning_parts.append("invalid cluster format")

        if len(justification.split()) >= 8:
            score += 0.05
            reasoning_parts.append("justification has enough detail")
        else:
            reasoning_parts.append("justification is too short")

        if any(keyword in justification_lower for keyword in task.rationale_keywords):
            score += 0.10
            reasoning_parts.append("justification references profile evidence")
        else:
            reasoning_parts.append("justification misses profile evidence")

        if normalized_cluster == task.target_cluster:
            score += 0.35
            reasoning_parts.append(f"correct cluster {task.target_cluster}")
        else:
            reasoning_parts.append(f"target cluster is {task.target_cluster}")

        if normalized_stream == task.target_stream:
            score += 0.40
            reasoning_parts.append(f"correct stream {task.target_stream}")
        else:
            reasoning_parts.append(f"target stream is {task.target_stream}")

        bounded_score = min(round(score, 4), 1.0)
        if bounded_score >= 1.0:
            bounded_score = self.MAX_VALIDATOR_SCORE
        elif bounded_score <= 0.0:
            bounded_score = self.MIN_VALIDATOR_SCORE
        reward = Reward(score=bounded_score)
        return reward, "; ".join(reasoning_parts)

    def _normalize_value(self, raw_value: str, mapping: Dict[str, str]) -> Optional[str]:
        return mapping.get(raw_value.strip().lower())

    def _new_episode_id(self) -> str:
        return f"aspirepath-{uuid4().hex[:12]}"
