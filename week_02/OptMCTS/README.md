# OptMCTS Baseline

Baseline replication for **SolverLLM** (NeurIPS 2025), using a Chain-of-Experts pipeline to solve optimization problems.

## Method

The pipeline sends each optimization problem through three LLM stages, then executes the generated code:

1. **Interpretation** - Extract key information from the problem
2. **Formulation** - Write the mathematical formulation (variables, objective, constraints)
3. **Code Generation** - Generate Python/scipy code to solve the problem
4. **Execution** - Run the code and extract the answer (with up to 3 retries on failure)

Model: `gpt-4o-mini` | Temperature: 0

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your-key-here
```

## Data

9 datasets (2634 problems total) are stored in a single SQLite database (`data/testset.db`).

To reload the database from raw JSONL files:

```bash
python load_data.py
```

## Usage

Run the full benchmark:

```bash
python main.py
```

Run a specific dataset:

```bash
python main.py --dataset nl4opt
```

Print results summary:

```bash
python main.py --summary
```

Progress is saved to the database automatically. Rerunning skips already-completed problems.

Query the database directly:

```bash
# View all tables
sqlite3 data/testset.db ".tables"

# Results summary by dataset
sqlite3 data/testset.db "SELECT dataset, COUNT(*) as total, SUM(success) as exec, SUM(correct) as correct FROM results GROUP BY dataset;"

# View a specific result
sqlite3 data/testset.db "SELECT * FROM results WHERE problem_id = 1;"
```

## Results

| Dataset | Total | ER% | SA% |
|---|--:|--:|--:|
| complexor | 18 | 100.0% | 33.3% |
| industryor | 100 | 81.0% | 17.0% |
| mamo_complex | 211 | 78.7% | 21.3% |
| mamo_easy | 652 | 100.0% | 57.4% |
| nl4opt | 230 | 100.0% | 68.3% |
| nlp4lp | 242 | 100.0% | 64.5% |
| optibench | 605 | 97.2% | 57.0% |
| optmath_bench | 166 | 61.4% | 4.8% |
| task3 | 410 | 99.3% | 68.0% |
| **TOTAL** | **2634** | **94.4%** | **52.7%** |

- **ER%**: Execution Rate (code runs successfully)
- **SA%**: Solving Accuracy (answer within 10% of ground truth)

### Comparison with Paper Baselines (SA%)

| Dataset | Paper CoE (GPT-4) | Paper GPT-4 Directly | Ours (gpt-4o-mini) |
|---|:-:|:-:|:-:|
| NL4Opt | 64.2% | 47.3% | **68.3%** |
| NLP4LP | 53.1% | 35.8% | **64.5%** |
| ComplexOR | 38.1% | 9.5% | **33.3%** |
| MamoEasy | N/A | 66.5% | **57.4%** |
| MamoComplex | N/A | 14.6% | **21.3%** |
| IndustryOR | N/A | 28.0% | **17.0%** |
| OptiBench | N/A | N/A | **57.0%** * |
| OptMathBench | N/A | N/A | **4.8%** * |
| Task3 | N/A | N/A | **68.0%** * |

\* Not reported in the paper, no baseline available for comparison.

- **Paper CoE**: Chain-of-Experts baseline from Table 1, using GPT-4
- **Paper GPT-4 Directly**: Direct prompting baseline from Table 1 & 2

On the 3 datasets with Chain-of-Experts comparison, our gpt-4o-mini results closely match or exceed the paper's GPT-4 results, confirming successful baseline replication.

## Project Structure

```
├── main.py                 # Entry point: run benchmark and save results
├── load_data.py            # Load JSONL datasets into SQLite
├── pipeline/
│   └── experts.py          # Chain-of-Experts pipeline
├── core/
│   ├── llm.py              # OpenAI API wrapper
│   └── executor.py         # Code execution with timeout
├── evaluation/
│   └── evaluator.py        # Answer extraction and evaluation
└── data/
    ├── testset.db          # SQLite database (problems + results)
    └── testset/            # Raw JSONL dataset files
```
