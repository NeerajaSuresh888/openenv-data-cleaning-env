from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional,Callable

@dataclass
class Task:
    task_id: str
    description: str
    difficulty: str
    dirty_dataset: List[Dict[str, Any]]
    total_issues: int
    grader: Callable[[List[Dict], List[Dict]], float]


_EASY_DIRTY = [
    {"id": 1, "name": "Alice", "age": 30,   "email": "alice@example.com", "city": "New York"},
    {"id": 2, "name": "Bob",   "age": None,  "email": "bob@example.com",   "city": "London"},
    {"id": 3, "name": None,    "age": 25,   "email": "carol@example.com", "city": "Paris"},
    {"id": 4, "name": "Dave",  "age": 40,   "email": None,                "city": "Berlin"},
    {"id": 5, "name": "Eve",   "age": None,  "email": "eve@example.com",   "city": None},
]

def _grade_easy(cleaned: List[Dict], original: List[Dict]) -> float:
    missing_cells = [
        (row["id"], col)
        for row in original
        for col, val in row.items()
        if val is None
    ]
    if not missing_cells:
        return 1.0

    fixed = 0
    id_to_row = {r["id"]: r for r in cleaned}
    for (rid, col) in missing_cells:
        row = id_to_row.get(rid)
        if row and row.get(col) not in (None, "", "null", "N/A"):
            fixed += 1

    return round(fixed / len(missing_cells), 4)


TASK_EASY = Task(
    task_id="fill_missing_easy",
    description=(
        "You have a customer table with 5 rows. "
        "Some cells contain null/missing values. "
        "Fill every missing cell with a reasonable value. "
        "Use fill_missing actions. When done, call submit."
    ),
    difficulty="easy",
    dirty_dataset=_EASY_DIRTY,
    total_issues=5,
    grader=_grade_easy,
)


_MEDIUM_DIRTY = [
    {"id": 1, "product": "Widget A", "price": "19.99",   "quantity": 100,     "in_stock": True},
    {"id": 2, "product": "Widget B", "price": 29.99,     "quantity": "fifty", "in_stock": True},
    {"id": 3, "product": "Widget C", "price": "bad_val", "quantity": 200,     "in_stock": "yes"},
    {"id": 4, "product": "Widget D", "price": 9.99,      "quantity": 75,      "in_stock": False},
    {"id": 5, "product": "Widget B", "price": 29.99,     "quantity": "fifty", "in_stock": True},
    {"id": 6, "product": "Widget E", "price": None,      "quantity": 300,     "in_stock": True},
]


def _grade_medium(cleaned: List[Dict], original: List[Dict]) -> float:
    score = 0.0
    total = 6.0

    def get(rid: int):
        return next((r for r in cleaned if r.get("id") == rid), None)

    r1 = get(1)
    if r1 and isinstance(r1.get("price"), (int, float)):
        score += 1

    r2 = get(2)
    if r2 and isinstance(r2.get("quantity"), int):
        score += 1

    r3 = get(3)
    if r3 and isinstance(r3.get("in_stock"), bool):
        score += 1

    if r3 and isinstance(r3.get("price"), (int, float)):
        score += 1

    r6 = get(6)
    if r6 and r6.get("price") not in (None, "", "null", "N/A"):
        score += 1

    if len(cleaned) < len(original):
        seen = set()
        is_duplicate_free = True
        for r in cleaned:
            key = (r.get("product"), str(r.get("price")), str(r.get("quantity")))
            if key in seen:
                is_duplicate_free = False
                break
            seen.add(key)
        if is_duplicate_free:
            score += 1

    return round(score / total, 4)


TASK_MEDIUM = Task(
    task_id="fix_types_medium",
    description=(
        "You have a product inventory table with 6 rows. "
        "Issues include: string prices that should be floats, "
        "string quantities that should be ints, a string boolean, "
        "one duplicate row, and one missing price. "
        "Use fix_type, remove_duplicate, and fill_missing actions. Submit when done."
    ),
    difficulty="medium",
    dirty_dataset=_MEDIUM_DIRTY,
    total_issues=6,
    grader=_grade_medium,
)


_HARD_DIRTY = [
    {"id": 1,  "name": "john smith",   "dob": "1990-05-14", "salary": 55000,   "dept": "engineering"},
    {"id": 2,  "name": "JANE DOE",     "dob": "14/06/1988", "salary": 58000,   "dept": "Engineering"},
    {"id": 3,  "name": "Bob Johnson",  "dob": "1995-11-22", "salary": 9999999, "dept": "HR"},
    {"id": 4,  "name": "alice wu",     "dob": "1992-03-30", "salary": None,    "dept": "hr"},
    {"id": 5,  "name": "Charlie",      "dob": "07-08-1985", "salary": 62000,   "dept": "Sales"},
    {"id": 6,  "name": "diana prince", "dob": "1993-12-01", "salary": 60000,   "dept": "sales"},
    {"id": 7,  "name": "Eve Torres",   "dob": "1991-07-19", "salary": -5000,   "dept": "Engineering"},
    {"id": 8,  "name": "FRANK",        "dob": "1988-02-28", "salary": 57000,   "dept": "engineering"},
    {"id": 9,  "name": "Grace Hopper", "dob": "1906-12-09", "salary": 70000,   "dept": "Engineering"},
    {"id": 10, "name": "Hank Pym",     "dob": "1994-09-15", "salary": 53000,   "dept": "HR"},
]


def _grade_hard(cleaned: List[Dict], original: List[Dict]) -> float:
    score = 0.0
    total = 10.0

    def get(rid):
        return next((r for r in cleaned if r.get("id") == rid), None)

    r1 = get(1)
    if r1 and r1.get("name", "").istitle():
        score += 1

    r2 = get(2)
    if r2 and not r2.get("name", "").isupper():
        score += 1

    if r2:
        parts = r2.get("dob", "").split("-")
        if len(parts) == 3 and len(parts[0]) == 4:
            score += 1

    r3 = get(3)
    if r3 and r3.get("__outlier_salary"):
        score += 1

    r4 = get(4)
    if r4 and r4.get("salary") not in (None, "", "null"):
        score += 1

    if r4 and r4.get("dept") not in ("hr", "HR"):
        score += 1

    r5 = get(5)
    if r5:
        parts = r5.get("dob", "").split("-")
        if len(parts) == 3 and len(parts[0]) == 4:
            score += 1

    r6 = get(6)
    if r6 and r6.get("name", "").istitle():
        score += 1

    r7 = get(7)
    if r7 and r7.get("__outlier_salary"):
        score += 1

    r8 = get(8)
    if r8 and not r8.get("name", "").isupper():
        score += 1

    return round(score / total, 4)


TASK_HARD = Task(
    task_id="normalize_hard",
    description=(
        "You have a company HR table with 10 employees. "
        "Issues include: inconsistent name casing, "
        "non-ISO date formats, salary outliers, "
        "missing salary, and inconsistent department names. "
        "Use normalize_value, flag_outlier, and fill_missing actions. "
        "Submit when done."
    ),
    difficulty="hard",
    dirty_dataset=_HARD_DIRTY,
    total_issues=10,
    grader=_grade_hard,
)


TASKS = {
    "fill_missing_easy": TASK_EASY,
    "fix_types_medium": TASK_MEDIUM,
    "normalize_hard": TASK_HARD,
}

