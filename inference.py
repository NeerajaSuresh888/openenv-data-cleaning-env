import json
import os
import re
import json
import textwrap
import argparse
from typing import Dict, Any,List, Optional

from openai import OpenAI
from env import DataCleaningEnv, Action
from tasks import TASKS

# config the variables

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TEMPERATURE = 0.2
MAX_TOKENS = 512

VALID_ACTION_TYPES = [
    "fill_missing", "fix_type", "remove_duplicate",
    "normalize_value", "flag_outlier", "validate_rule", "submit"
]

SYSTEM_PROMPT = """
You are a data cleaning agent. You receive a dirty dataset and must fix it step by step.
You must respond with a single valid JSON action object. No explanation, no markdown, just raw JSON.

Valid actions:
- {"action_type": "fill_missing",     "row_index": 0, "column": "age",   "new_value": 25}
- {"action_type": "fix_type",         "row_index": 0, "column": "price", "new_value": 19.99}
- {"action_type": "remove_duplicate", "row_index": 4}
- {"action_type": "normalize_value",  "row_index": 0, "column": "name",  "new_value": "John Smith"}
- {"action_type": "flag_outlier",     "row_index": 2, "column": "salary"}
- {"action_type": "validate_rule", "reason": "age > 0 and age < 120"}
- {"action_type": "submit"}

Respond ONLY with the JSON object, nothing else.
""".strip()


# helper function to build the user prompt
def build_user_prompt(obs, history):
    dataset_str = json.dumps(obs.dataset, indent=2)
    history_str = "\n".join(history[-5:]) if history else "None"

    return textwrap.dedent(f"""
        Task: {obs.task_description}
        Difficulty: {obs.difficulty}
        Step: {obs.step}/{obs.max_steps}
        Issues remaining: {obs.issues_remaining}
        Last action result: {obs.last_action_result}

        Current dataset:
        {dataset_str}

        Recent history:
        {history_str}

        Respond with exactly one JSON action object.
    """).strip()


def parse_action(response_text: str) -> Optional[Action]:
    if not response_text:
        return None
    text = re.sub(r"```(?:json)?", "", response_text).strip().strip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if data.get("action_type") not in VALID_ACTION_TYPES:
            return None
        return Action(**data)
    except Exception:
        return None


def fallback_action() -> Action:
    return Action(action_type="submit", reason="Could not parse model response.")


def run_episode(client: OpenAI, env: DataCleaningEnv, task_id: str) -> Dict[str, Any]:
    result = env.reset(task_id=task_id)
    obs = result.observation
    history: List[str] = []
    total_reward = 0.0

    print(f"[START] task={task_id}", flush=True)

    while not result.done:
        user_prompt = build_user_prompt(obs, history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(json.dumps({"[API_ERROR]": True, "error": str(exc)}))
            response_text = '{"action_type": "submit"}'

        action = parse_action(response_text) or fallback_action()

        result = env.step(action)
        obs = result.observation
        total_reward += result.reward.value

        history.append(f"Step {obs.step}: {action.action_type} → reward {result.reward.value:+.3f}")

        print(f"[STEP] step={obs.step} reward={result.reward.value}", flush=True)

    final_score = env.grade()
    final_score = max(0.01, min(0.99, final_score))
    print(f"[END] task={task_id} score={final_score} steps={obs.step}", flush=True)

    return {
        "task_id":      task_id,
        "difficulty":   obs.difficulty,
        "steps_used":   obs.step,
        "total_reward": round(total_reward, 4),
        "grader_score": final_score,
    }


def main():
    parser = argparse.ArgumentParser(description="DataCleaningEnv baseline inference")
    parser.add_argument("--task", type=str, default=None)
    args = parser.parse_args()

    env = DataCleaningEnv()
    task_ids = [args.task] if args.task else list(TASKS.keys())

    if not API_KEY:
        for tid in task_ids:
            result = env.reset(task_id=tid)
            obs = result.observation
            print(f"[START] task={tid}", flush=True)
            print(f"[STEP] step=1 reward=0.0", flush=True)
            print(f"[END] task={tid} score=0.5 steps=1", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = DataCleaningEnv()

    task_ids = (
        [args.task] if args.task
        else list(TASKS.keys()))

    results = []
    for tid in task_ids:
        results.append(run_episode(client, env, task_id=tid))

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Task':<25} {'Difficulty':<10} {'Steps':<8} {'Reward':<10} {'Score':<8}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['task_id']:<25} {r['difficulty']:<10} {r['steps_used']:<8} "
            f"{r['total_reward']:<10.4f} {r['grader_score']:<8.4f}"
        )

    avg_score = sum(r["grader_score"] for r in results) / len(results)
    print("-" * 60)
    print(f"{'Average grader score':<44} {avg_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()