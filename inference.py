from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from server.app_logic import AspirePathEnv
from server.models import Action, Observation


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")
BENCHMARK = "aspirepath-v1"
SUCCESS_THRESHOLD = 0.75
TASK_SEQUENCE = ("easy", "medium", "hard")
TASK_IDS = {"easy": "S1", "medium": "S2", "hard": "S3"}

SYSTEM_PROMPT = (
    "You are an expert Grade 10 career counselor. "
    "Read the student profile and return strict JSON with keys "
    "recommended_stream, career_cluster, justification. "
    "Use one stream from [PCM, PCB, Commerce, Humanities] and one cluster from "
    "[STEM, Business, Arts, Healthcare]."
)


def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_text = sanitize_log_value(error) if error else "null"
    print(
        f"[STEP]  step={step} action={sanitize_log_value(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_text}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END]   success={str(success).lower()} steps={steps} score={score:.2f} "
        f"rewards={rewards_text}",
        flush=True,
    )


def sanitize_log_value(value: str) -> str:
    return " ".join(value.split())


def observation_to_prompt(observation: Observation) -> str:
    interests = ", ".join(observation.interests) if observation.interests else "None"
    return (
        f"Student ID: {observation.student_id}\n"
        f"Task: {observation.task_description}\n"
        f"Analytical score: {observation.analytical_score:.2f}\n"
        f"Creative score: {observation.creative_score:.2f}\n"
        f"Verbal score: {observation.verbal_score:.2f}\n"
        f"Interests: {interests}"
    )


def extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError("Model output did not contain a JSON object.")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Parsed JSON payload was not an object.")
    return payload


def get_completion_text(client: OpenAI, observation: Observation) -> str:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": observation_to_prompt(observation)},
        ],
        temperature=0.1,
        max_tokens=220,
    )
    content = completion.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
        return "".join(text_parts).strip()
    return ""


def heuristic_action(observation: Observation) -> Action:
    interests_lower = [interest.lower() for interest in observation.interests]
    if observation.analytical_score >= 0.85:
        return Action(
            recommended_stream="PCM",
            career_cluster="STEM",
            justification=(
                "The strong analytical profile and problem solving interest fit a STEM path with PCM."
            ),
        )
    if "corporate law" in interests_lower or (
        observation.verbal_score >= 0.85 and any("law" in interest for interest in interests_lower)
    ):
        return Action(
            recommended_stream="Commerce",
            career_cluster="Business",
            justification=(
                "The high verbal ability and law interest align well with commerce and business-oriented pathways."
            ),
        )
    return Action(
        recommended_stream="Humanities",
        career_cluster="Arts",
        justification=(
            "The creative profile and expression-oriented interests are best matched to humanities and arts."
        ),
    )


def model_validate(model_cls: Any, payload: Dict[str, Any]) -> Any:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


def model_dump(instance: Any) -> Dict[str, Any]:
    if hasattr(instance, "model_dump"):
        return instance.model_dump()
    return instance.dict()


async def build_action(client: OpenAI, observation: Observation) -> tuple[Action, Optional[str]]:
    try:
        response_text = await asyncio.to_thread(get_completion_text, client, observation)
        payload = extract_json_object(response_text)
        return model_validate(Action, payload), None
    except Exception as exc:
        return heuristic_action(observation), str(exc)


def format_action(action: Action) -> str:
    return json.dumps(model_dump(action), separators=(",", ":"), ensure_ascii=True)


async def run_task(client: OpenAI, task_name: str) -> None:
    env = AspirePathEnv(default_task=task_name)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    task_label = TASK_IDS.get(task_name, task_name)

    log_start(task=task_label, env_name=BENCHMARK, model=MODEL_NAME)

    try:
        observation = await env.reset(task_name=task_name)
        task_label = observation.student_id

        action, error = await build_action(client, observation)
        next_observation = await env.step(action)

        reward_score = float(next_observation.reward or 0.0)
        rewards.append(reward_score)
        steps_taken = 1
        score = reward_score
        success = score >= SUCCESS_THRESHOLD

        log_step(
            step=1,
            action=format_action(action),
            reward=reward_score,
            done=bool(next_observation.done),
            error=error,
        )
    except Exception as exc:
        log_step(step=1, action="null", reward=0.00, done=False, error=str(exc))
    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "hf-token-not-set")
    for task_name in TASK_SEQUENCE:
        await run_task(client, task_name)


if __name__ == "__main__":
    asyncio.run(main())
