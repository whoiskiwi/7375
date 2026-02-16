import json
import sqlite3

db = sqlite3.connect('data/testset.db')
db.execute("""
    CREATE TABLE IF NOT EXISTS problems (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
           dataset TEXT,
           question TEXT,
           answer REAL,
           original TEXT,
           original_index INTEGER
)
""")

files = {
    "nl4opt": "data/testset/nl4opt_test.jsonl",
    "complexor": "data/testset/complexor.jsonl",
    "nlp4lp": "data/testset/nlp4lp.jsonl",
    "industryor": "data/testset/industryor.jsonl",
    "mamo_complex": "data/testset/mamo_complex_test.jsonl",
    "mamo_easy": "data/testset/mamo_easy_test.jsonl",
    "optibench": "data/testset/optibench.jsonl",
    "optmath_bench": "data/testset/optmath_bench.jsonl",
    "task3": "data/testset/task3_test.jsonl",
}

for dataset, path in files.items():
    with open(path, 'r') as f:
        for line in f:
            row = json.loads(line)
            db.execute(
                "INSERT INTO problems (dataset, question, answer, original, original_index) VALUES (?, ?, ?, ?, ?)",
                (dataset, row["question"], float(row["answer"]), row["ori"], row["index"])
            )

db.commit()
db.close()