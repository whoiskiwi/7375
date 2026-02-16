import sqlite3
import time
import argparse
from pipeline.experts import chain_of_experts
from evaluation.evaluator import extract_answer, evaluate

DB_PATH = 'data/testset.db'

def load_problems(dataset=None, limit=None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    query = "SELECT id, dataset, question, answer FROM problems"
    params = []
    if dataset:
        query += " WHERE dataset = ?"
        params.append(dataset)
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]

def init_results_table():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
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

def save_result(problem_id, dataset, expected, predicted, success, correct, output, error):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO results
        (problem_id, dataset, expected, predicted, success, correct, output, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (problem_id, dataset, expected, predicted, int(success), int(correct), output, error))
    conn.commit()
    conn.close()

def get_completed_ids():
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT problem_id FROM results").fetchall()
        conn.close()
        return {r[0] for r in rows}
    except sqlite3.OperationalError:
        conn.close()
        return set()

def print_summary():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    datasets = conn.execute("SELECT DISTINCT dataset FROM results ORDER BY dataset").fetchall()
    print("\n" + "=" * 60)
    print(f"{'Dataset':<20} {'Total':>6} {'Exec':>6} {'Correct':>8} {'ER%':>7} {'SA%':>7}")
    print("-" * 60)
    total_all, exec_all, correct_all = 0, 0, 0
    for row in datasets:
        ds = row['dataset']
        stats = conn.execute(
            "SELECT COUNT(*) as total, SUM(success) as exec, SUM(correct) as corr FROM results WHERE dataset=?",
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
    args = parser.parse_args()

    if args.summary:
        print_summary()
        exit()

    init_results_table()
    completed = get_completed_ids()
    problems = load_problems(dataset=args.dataset, limit=args.limit)
    # skip already completed
    problems = [p for p in problems if p['id'] not in completed]
    total = len(problems)

    if total == 0:
        print("All problems already completed.")
        print_summary()
        exit()

    print(f"Running {total} problems (skipping {len(completed)} already done)...")
    start = time.time()

    for i, p in enumerate(problems):
        print(f"\n[{i+1}/{total}] (id={p['id']}) [{p['dataset']}] {p['question'][:80]}...")
        try:
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
            error=result["error"][:2000]
        )

        status = "CORRECT" if correct else ("EXEC_OK" if result["success"] else "FAIL")
        print(f"  [{status}] expected={p['answer']}, predicted={predicted}")

        elapsed = time.time() - start
        avg = elapsed / (i + 1)
        remaining = avg * (total - i - 1)
        print(f"  elapsed={elapsed:.0f}s, avg={avg:.1f}s/problem, ETA={remaining/60:.0f}min")

    print_summary()
