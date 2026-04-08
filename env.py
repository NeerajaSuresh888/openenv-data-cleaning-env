from __future__ import annotations
import copy
import random
from typing import Dict, List, Optional, Tuple, Any

from pydantic import BaseModel, Field
from tasks import Task, TASKS


class Observation(BaseModel):
    task_id: str
    task_description: str
    difficulty: str
    dataset: List[Dict[str, Any]]
    step: int
    max_steps: int
    issues_remaining: int
    last_action_result:str =""
    done: bool = False


class Action(BaseModel):
    action_type: str
    row_index: Optional[int] = None
    column: Optional[str] =None
    new_value: Optional[Any] = None
    reason: Optional[str] = None


class Reward(BaseModel):
    value: float   #between -1.0 and 1.0
    partial_score: float   # between 0.0 and 1.0
    info: str


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = {}


class EnvState(BaseModel):
    task_id:str
    step: int
    dataset: List[Dict[str, Any]]
    issues_resolved: int
    total_issues: int
    score: float


class DataCleaningEnv:
    MAX_STEPS_PER_DIFFICULTY = {"easy":10, "medium":20, "hard":30}
    STEP_PENALTY = -0.02
    INVALID_ACTION_PENALTY = -0.1
    COMPLETION_BONUS =0.2

    def __init__(self,task_id:Optional[str]=None, seed:int =42):
        self._seed = seed
        self._rng = random.Random(seed)
        self._task_id = task_id
        self._task: Optional[Task] = None
        self._dataset: List[Dict[str, Any]] = []
        self._original_dataset: List[Dict[str, Any]] = []
        self._step = 0
        self._issues_resolved = 0
        self._total_issues = 0
        self._done = False
        self._episode_reward = 0.0
    

    '''reset() method initializes the environment for a new episode. 
    It selects a task (either specified or random), loads the corresponding dataset, and resets all relevant state variables.
    The method returns the initial observation and reward for the episode.'''
    def reset(self, task_id :Optional[str]=None) ->StepResult:
        tid =task_id or self._task_id or self._rng.choice(list(TASKS.keys()))
        self._task = TASKS[tid]
        self._task_id = tid
        self._step = 0
        self._done = False
        self._episode_reward= 0.0

        self._dataset =copy.deepcopy(self._task.dirty_dataset) 
         # this makes a full independent copy of the dataset  so that the agent's change donot affect the orginal template
        self._original_dataset = copy.deepcopy(self._task.dirty_dataset)
        self._total_issues = self._task.total_issues
        self._issues_resolved = 0

        obs = self._make_observation(last_result = "Episode started. Dataset Loaded.")
        return StepResult(
            observation=obs,
            reward=Reward(value=0.0, partial_score=0.0, info="Reset"),
            done=False,
            info ={"task_id":tid, "total_issues": self._total_issues, "issues_resolved":self._issues_resolved},
        )
    

    ''' step() method takes an action from the agent, applies it to the dataset, and updates the environment's state accordingly.
        It checks the validity of the action, applies it if valid, and then evaluates the new state to calculate the reward and determine if the episode is done.
        The method returns the new observation, reward, and done flag.'''
    def step(self, action:Action) ->StepResult:
        if self._done:
            obs = self._make_observation(last_result ="Episode already completed. Please reset.")
            return StepResult(
                observation=obs,
                reward =Reward(value =0.0,partial_score =self._partial_score(), info = "Episode already done"),
                done =True,
            )
        
        self._step += 1
        reward_value, result_msg, issue_delta = self._apply_action(action)
        self._issues_resolved = max(0,min(self._issues_resolved + issue_delta, self._total_issues))
        self._episode_reward += reward_value

        all_resolved = self._issues_resolved >=self._total_issues
        max_steps = self.MAX_STEPS_PER_DIFFICULTY[self._task.difficulty]
        out_of_steps = self._step >= max_steps
        
        if all_resolved:
            steps_left = max_steps -self._step
            bonus = self.COMPLETION_BONUS * (steps_left / max_steps)
            self._episode_reward += bonus
            result_msg += f" All issues resolved! Bonus: {bonus:.2f}"
            self._done = True
        elif out_of_steps or action.action_type == "submit":
            self._done = True
            result_msg += " Episode ended."

        obs = self._make_observation(last_result = result_msg)
        return StepResult(
            observation=obs,
            reward=Reward(value=round(reward_value,4), partial_score=round(self._partial_score(),4), info=result_msg),
            done=self._done,
            info={"issues_resolved": self._issues_resolved, "total_issues": self._total_issues},
        )


    def state(self) -> EnvState:
        return EnvState(
            task_id =self._task_id or "",
            step = self._step,
            dataset = copy.deepcopy(self._dataset),                                   # RETURNS SS OF ENVIRONMENT STATE WITH CURRENT DATASET.
            issues_resolved = self._issues_resolved,
            total_issues = self._total_issues,
            score = round(self._partial_score(),4),
        )
    

    def grade(self) -> float:
        if self._task is None:                                                      #Runs the grader function from tasks.py and returnd actual score
            return 0.0
        return self._task.grader(self._dataset, self._original_dataset)
    

    def _partial_score(self) -> float:
        if self._total_issues == 0:
            return 1.0
        return self._issues_resolved / self._total_issues
    

    def _make_observation(self,last_result :str="") -> Observation:
        max_steps = self.MAX_STEPS_PER_DIFFICULTY.get(
            self._task.difficulty if self._task  else "easy",10
        )
        return Observation(
            task_id =self._task_id or "",
            task_description =self._task.description if self._task else "",
            difficulty =self._task.difficulty if self._task else "easy",
            dataset = copy.deepcopy(self._dataset),
            step = self._step,
            max_steps=max_steps,
            issues_remaining = self._total_issues - self._issues_resolved,
            last_action_result = last_result,
            done = self._done,
        )
    

    def _apply_action(self, action: Action) -> Tuple[float, str, int]:
        atype = action.action_type


        if atype =="fill_missing":
            return self._handle_fill_missing(action)
        elif atype == "fix_type":
            return self._handle_fix_type(action)
        elif atype == "remove_duplicate":
            return self._handle_remove_duplicate(action)                            # traffic controller, reads action type and send it to the right handler.
        elif atype =="normalize_value":
            return self._handle_normalize_value(action)
        elif atype =="flag_outlier":
            return self._handle_flag_outlier(action)
        elif atype == "validate_rule":
            return self._handle_validate_rule(action)
        elif atype =="submit":
            return (0.0,"Agent Submitted.",0)
        else:
            return (self.INVALID_ACTION_PENALTY, f"Unknown action: {atype}",0)
        

    def _handle_fill_missing(self, action: Action) -> Tuple[float, str, int]:
        if action.row_index is None or action.column is None or action.new_value is None:
            return (self.INVALID_ACTION_PENALTY, "fill_missing requires row_index, column, new_value.", 0)

        row = self._dataset[action.row_index] if 0 <= action.row_index < len(self._dataset) else None
        if row is None:
            return (self.INVALID_ACTION_PENALTY, f"Row {action.row_index} does not exist.", 0)

        current = row.get(action.column)
        if current not in (None, "", "null", "N/A"):
            return (self.STEP_PENALTY, f"Column '{action.column}' is not missing.", 0)

        row[action.column] = action.new_value
        return (0.15, f"Filled missing value in row {action.row_index}, col '{action.column}'.", 1)

    def _handle_fix_type(self, action: Action) -> Tuple[float, str, int]:
        if action.row_index is None or action.column is None or action.new_value is None:
            return (self.INVALID_ACTION_PENALTY, "fix_type requires row_index, column, new_value.", 0)
        row = self._dataset[action.row_index] if 0 <= action.row_index < len(self._dataset) else None
        if row is None:
            return (self.INVALID_ACTION_PENALTY, f"Row {action.row_index} does not exist.", 0)
        row[action.column] = action.new_value
        return (0.15, f"Fixed type in row {action.row_index}, col '{action.column}'.", 1)

    def _handle_remove_duplicate(self, action: Action) -> Tuple[float, str, int]:
        if action.row_index is None:
            return (self.INVALID_ACTION_PENALTY, "remove_duplicate requires row_index.", 0)
        if action.row_index < 0 or action.row_index >= len(self._dataset):
            return (self.INVALID_ACTION_PENALTY, f"Row {action.row_index} out of range.", 0)

        target = self._dataset[action.row_index]
        # Compare ignoring 'id' since duplicates have different ids
        def row_key(r):
            return {k: v for k, v in r.items() if k != "id"}

        is_duplicate = any(
            i != action.row_index and row_key(self._dataset[i]) == row_key(target)
            for i in range(len(self._dataset))
        )
        if not is_duplicate:
            return (self.INVALID_ACTION_PENALTY, f"Row {action.row_index} is not a duplicate.", 0)

        self._dataset.pop(action.row_index)
        return (0.2, f"Removed duplicate row {action.row_index}.", 1)

    def _handle_normalize_value(self, action: Action) -> Tuple[float, str, int]:
        if action.row_index is None or action.column is None or action.new_value is None:
            return (self.INVALID_ACTION_PENALTY, "normalize_value requires row_index, column, new_value.", 0)

        row = self._dataset[action.row_index] if 0 <= action.row_index < len(self._dataset) else None
        if row is None:
            return (self.INVALID_ACTION_PENALTY, f"Row {action.row_index} does not exist.", 0)

        old = row.get(action.column)
        row[action.column] = action.new_value
        return (0.15, f"Normalized row {action.row_index}, col '{action.column}': {old!r} → {action.new_value!r}.", 1)

    def _handle_flag_outlier(self, action: Action) -> Tuple[float, str, int]:
        if action.row_index is None or action.column is None:
            return (self.INVALID_ACTION_PENALTY, "flag_outlier requires row_index, column.", 0)

        row = self._dataset[action.row_index] if 0 <= action.row_index < len(self._dataset) else None
        if row is None:
            return (self.INVALID_ACTION_PENALTY, f"Row {action.row_index} does not exist.", 0)

        row[f"__outlier_{action.column}"] = True
        return (0.1, f"Flagged outlier at row {action.row_index}, col '{action.column}'.", 1)
    
    def _handle_validate_rule(self, action: Action) -> Tuple[float, str, int]:
        if not action.reason:  # We'll reuse 'reason' field to pass the rule string
            return (self.INVALID_ACTION_PENALTY, "validate_rule requires 'reason' field containing the rule.", 0)

        rule = action.reason.strip()
        try:
            # Simple but effective: evaluate basic column rules (e.g. "age > 0 and age < 120", "salary > 0")
            column = None
            if "age" in rule.lower():
                column = "age"
            elif "salary" in rule.lower():
                column = "salary"
            elif "price" in rule.lower():
                column = "price"
            elif "quantity" in rule.lower():
                column = "quantity"

            if not column:
                return (self.STEP_PENALTY, f"Rule '{rule}' not supported yet.", 0)

            valid_count = 0
            for row in self._dataset:
                val = row.get(column)
                if val is not None:
                    # Very basic safe eval for numeric rules (hackathon-friendly)
                    if eval(str(val) + " " + rule.split(" ", 1)[-1] if " " in rule else "True", {"__builtins__": {}}, {}):
                        valid_count += 1

            if valid_count == len(self._dataset):
                return (0.15, f"Business rule validated successfully: {rule}", 1)
            else:
                return (self.STEP_PENALTY, f"Rule '{rule}' failed for some rows.", 0)
        except Exception:
            return (self.INVALID_ACTION_PENALTY, f"Invalid rule format: {rule}", 0)