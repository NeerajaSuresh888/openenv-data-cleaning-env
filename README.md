# DataCleaningEnv

An **OpenEnv-compliant reinforcement learning environment** for data cleaning and validation.

Agents receive a dirty dataset and must identify and fix real-world data quality issues step by step using a structured action API. Built for the **Meta PyTorch OpenEnv Hackathon x Scaler School of Technology 2026**.

---

## Motivation

Data cleaning is one of the most time-consuming tasks in real-world data pipelines — estimated to consume 60–80% of a data scientist's time. Training AI agents that can autonomously clean messy datasets has immediate practical value for data engineering, analytics, and ML pipelines. This environment provides a structured, gradable benchmark for evaluating such agents.

---

## Environment Overview

| Property | Value |
|---|---|
| API | `step()` / `reset()` / `state()` — OpenEnv spec |
| Tasks | 3 (easy → medium → hard) |
| Reward | Shaped, continuous [−1.0, 1.0] |
| Deploy | HF Spaces (Docker, port 7860) |

---

## Action Space

Each step the agent sends one action object:

| Action Type | Required Fields | Description |
|---|---|---|
| `fill_missing` | `row_index`, `column`, `new_value` | Fill a null/missing cell |
| `fix_type` | `row_index`, `column`, `new_value` | Correct a wrong data type |
| `remove_duplicate` | `row_index` | Delete a duplicate row |
| `normalize_value` | `row_index`, `column`, `new_value` | Standardize formatting (casing, dates) |
| `flag_outlier` | `row_index`, `column` | Mark a statistical outlier |
| `submit` | — | End the episode |

---

## Observation Space

Each observation contains:

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Unique task identifier |
| `task_description` | string | Natural-language task description |
| `difficulty` | string | `easy` / `medium` / `hard` |
| `dataset` | list of dicts | Current state of the dataset |
| `step` | int | Current step number |
| `max_steps` | int | Maximum steps for this task |
| `issues_remaining` | int | Number of unfixed issues |
| `last_action_result` | string | Feedback from the previous action |
| `done` | bool | Whether the episode has ended |

---

## Tasks

### Task 1 — `fill_missing_easy` (Easy, max 10 steps)

A 5-row customer table with 5 missing (null) cells. The agent must fill every missing cell with a reasonable value using `fill_missing` actions.

**Issues:** 5 missing values across `age`, `name`, `email`, `city` columns.

**Baseline score:** ~0.60

---

### Task 2 — `fix_types_medium` (Medium, max 20 steps)

A 6-row product inventory table with mixed type errors and one exact duplicate row.

**Issues (6 total):**
- String price that should be `float`
- String quantity that should be `int`
- Unparseable price string
- String boolean `"yes"` instead of `True`
- One exact duplicate row
- One missing price

**Baseline score:** ~0.50

---

### Task 3 — `normalize_hard` (Hard, max 30 steps)

A 10-row HR employee table with 10 issues spanning normalization, outliers, and missing data.

**Issues (10 total):**
- Name casing inconsistencies (ALL CAPS, all lowercase)
- Non-ISO date formats (`DD/MM/YYYY`, `DD-MM-YYYY`)
- Salary outlier (9,999,999)
- Negative salary outlier (−5,000)
- Missing salary and email fields
- Inconsistent department capitalization

**Baseline score:** ~0.30

---

## Reward Function

Reward is shaped continuously across the full trajectory — not just a binary end-of-episode signal.

| Event | Reward |
|---|---|
| Fixing an issue (`fill_missing`, `fix_type`, `normalize_value`) | +0.15 |
| Removing a duplicate row | +0.20 |
| Flagging an outlier | +0.10 |
| Completion bonus (all issues resolved) | up to +0.20 |
| Invalid / no-op action | −0.10 |
| Wasted step | −0.02 |
| Business rule validation (validate_rule) | +0.15 |

---

## Graders

Each task has a deterministic programmatic grader that returns a score in [0.0, 1.0]:

- **Easy grader:** fraction of originally-null cells that are now non-null
- **Medium grader:** 6-point checklist (1 point per resolved issue)
- **Hard grader:** 10-point checklist (1 point per resolved issue)

Graders are deterministic and reproducible. The same cleaned dataset always yields the same score.

---

## Setup & Usage

### Local

```bash
# Clone the repo
git clone https://huggingface.co/spaces/<your-username>/data-cleaning-env

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

### Run baseline inference

```bash
export HF_TOKEN=your_token_here
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1

python inference.py                           # all 3 tasks
python inference.py --task fill_missing_easy  # single task
```

---

## HTTP API

### `POST /reset`
```json
{"task_id": "fill_missing_easy", "seed": 42}
```
Returns: `session_id` + initial observation.

### `POST /step`
```json
{
  "session_id": "<id>",
  "action": {"action_type": "fill_missing", "row_index": 1, "column": "age", "new_value": 28}
}
```

### `GET /state/{session_id}`
Returns current env state snapshot.

### `POST /grade`
```json
{"session_id": "<id>"}
```
Returns final grader score [0.0, 1.0].

---

## Baseline Scores

| Task | Difficulty | Avg Grader Score |
|---|---|---|
| `fill_missing_easy` | Easy | ~0.60 |
| `fix_types_medium` | Medium | ~0.50 |
| `normalize_hard` | Hard | ~0.30 |
| **Average** | | **~0.47** |

Scores obtained with `Qwen/Qwen2.5-72B-Instruct` via HF inference router, temperature 0.2.

---

## Project Structure

```
data-cleaning-env/
├── env.py           # Core OpenEnv environment (step/reset/state)
├── tasks.py         # 3 tasks with dirty datasets and graders
├── inference.py     # Baseline inference script
├── app.py           # FastAPI server for HF Spaces
├── openenv.yaml     # OpenEnv metadata
├── Dockerfile       # Containerized deployment
├── requirements.txt
└── README.md
```

---

## License

Apache 2.0