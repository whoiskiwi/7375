import sqlite3
import time
import argparse
from pipeline.experts import chain_of_experts
from mcts.search import MCTS
from evaluation.evaluator import extract_answer, evaluate

DB_PATH = 'data/testset.db'

# Datasets reported in the paper (Table 1 + Table 2)
PAPER_DATASETS = {"nl4opt", "nlp4lp", "complexor", "mamo_easy", "mamo_complex", "industryor"}


def load_problems(dataset=None, limit=None, paper_only=False):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    query = "SELECT id, dataset, question, answer FROM problems"
    params = []
    conditions = []
    if dataset:
        conditions.append("dataset = ?")
        params.append(dataset)
    if paper_only:
        placeholders = ",".join("?" * len(PAPER_DATASETS))
        conditions.append(f"dataset IN ({placeholders})")
        params.extend(PAPER_DATASETS)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def init_results_table(table="results"):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            problem_id INTEGER PRIMARY KEY,
            dataset TEXT,
            expected REAL,
            predicted REAL,
            success INTEGER,
            correct INTEGER,
            output TEXT,
            error TEXT,
            FOREIGN KEY (problem_id) REFERENCES problems(id)
        )
    """)
    conn.commit()
    conn.close()


def save_result(problem_id, dataset, expected, predicted, success, correct, output, error, table="results"):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(f"""
        INSERT OR REPLACE INTO {table}
        (problem_id, dataset, expected, predicted, success, correct, output, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (problem_id, dataset, expected, predicted, int(success), int(correct), output, error))
    conn.commit()
    conn.close()


def get_completed_ids(table="results"):
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(f"SELECT problem_id FROM {table}").fetchall()
        conn.close()
        return {r[0] for r in rows}
    except sqlite3.OperationalError:
        conn.close()
        return set()


def print_summary(table="results"):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        datasets = conn.execute(f"SELECT DISTINCT dataset FROM {table} ORDER BY dataset").fetchall()
    except sqlite3.OperationalError:
        print(f"No results in table '{table}' yet.")
        conn.close()
        return
    print("\n" + "=" * 60)
    print(f"[{table}]")
    print(f"{'Dataset':<20} {'Total':>6} {'Exec':>6} {'Correct':>8} {'ER%':>7} {'SA%':>7}")
    print("-" * 60)
    total_all, exec_all, correct_all = 0, 0, 0
    for row in datasets:
        ds = row['dataset']
        stats = conn.execute(
            f"SELECT COUNT(*) as total, SUM(success) as exec, SUM(correct) as corr FROM {table} WHERE dataset=?",
            (ds,)
        ).fetchone()
        t, e, c = stats['total'], stats['exec'] or 0, stats['corr'] or 0
        total_all += t; exec_all += e; correct_all += c
        print(f"{ds:<20} {t:>6} {e:>6} {c:>8} {100*e/t:>6.1f}% {100*c/t:>6.1f}%")
    print("-" * 60)
    print(f"{'TOTAL':<20} {total_all:>6} {exec_all:>6} {correct_all:>8} {100*exec_all/total_all:>6.1f}% {100*correct_all/total_all:>6.1f}%")
    print("=" * 60)
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Run specific dataset only")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems")
    parser.add_argument("--summary", action="store_true", help="Print summary of existing results")
    parser.add_argument("--method", type=str, default="coe", choices=["coe", "mcts"],
                        help="Method: coe (chain-of-experts baseline) or mcts (SolverLLM)")
    parser.add_argument("--iterations", type=int, default=20,
                        help="MCTS iterations per problem (default: 20, same as paper)")
    parser.add_argument("--paper-only", action="store_true",
                        help="Only run the 6 datasets reported in the paper")
    args = parser.parse_args()

    # Each method stores results in its own table so we can compare side-by-side
    table = "results" if args.method == "coe" else "mcts_results"

    if args.summary:
        print_summary("results")
        print_summary("mcts_results")
        exit()

    init_results_table(table)
    completed = get_completed_ids(table)
    problems = load_problems(dataset=args.dataset, limit=args.limit, paper_only=args.paper_only)
    problems = [p for p in problems if p['id'] not in completed]
    total = len(problems)

    if total == 0:
        print("All problems already completed.")
        print_summary(table)
        exit()

    print(f"Method: {args.method.upper()}")
    if args.method == "mcts":
        print(f"MCTS iterations per problem: {args.iterations}")
    print(f"Running {total} problems (skipping {len(completed)} already done)...")
    start = time.time()

    mcts_solver = MCTS(n_iterations=args.iterations) if args.method == "mcts" else None

    for i, p in enumerate(problems):
        print(f"\n[{i+1}/{total}] (id={p['id']}) [{p['dataset']}] {p['question'][:80]}...")

        try:
            if args.method == "mcts":
                result = mcts_solver.search(p['question'], expected_answer=p['answer'])
            else:
                result = chain_of_experts(p['question'])
        except Exception as e:
            result = {"success": False, "output": "", "error": str(e)}

        predicted = extract_answer(result["output"])
        correct = evaluate(predicted, p['answer'])

        save_result(
            problem_id=p['id'],
            dataset=p['dataset'],
            expected=p['answer'],
            predicted=predicted,
            success=result["success"],
            correct=correct,
            output=result["output"][:2000],
            error=result["error"][:2000],
            table=table,
        )

        status = "CORRECT" if correct else ("EXEC_OK" if result["success"] else "FAIL")
        print(f"  [{status}] expected={p['answer']}, predicted={predicted}")

        elapsed = time.time() - start
        avg = elapsed / (i + 1)
        remaining = avg * (total - i - 1)
        print(f"  elapsed={elapsed:.0f}s, avg={avg:.1f}s/problem, ETA={remaining/60:.0f}min")

    print_summary(table)
