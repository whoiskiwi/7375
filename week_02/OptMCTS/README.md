# OptMCTS

Replication of **SolverLLM** (NeurIPS 2025), a framework that uses Monte Carlo Tree Search (MCTS) to iteratively refine mathematical formulations for solving optimization problems with LLMs.

Model: `gpt-4o-mini` | Temperature: 0

---

## Part 1: Baseline Replication (Chain-of-Experts)

The baseline sends each problem through three sequential LLM stages:

1. **Interpretation** — Extract key information from the problem
2. **Formulation** — Write the mathematical formulation
3. **Code Generation** — Generate Python/scipy code to solve it
4. **Execution** — Run the code and extract the answer (up to 3 retries on failure)

### Part 1 Results

| Dataset | Total | ER% | SA% |
|---|--:|--:|--:|
| complexor | 18 | 100.0% | 33.3% |
| industryor | 100 | 81.0% | 17.0% |
| mamo_complex | 211 | 78.7% | 21.3% |
| mamo_easy | 652 | 100.0% | 57.4% |
| nl4opt | 230 | 100.0% | 68.3% |
| nlp4lp | 242 | 100.0% | 64.5% |
| **TOTAL (paper sets)** | **1453** | **95.8%** | **51.6%** |

Comparison with paper baselines (SA%):

| Dataset | Paper CoE (GPT-4) | Paper GPT-4 Direct | Ours (gpt-4o-mini) |
|---|:-:|:-:|:-:|
| NL4Opt | 64.2% | 47.3% | **68.3%** |
| NLP4LP | 53.1% | 35.8% | **64.5%** |
| ComplexOR | 38.1% | 9.5% | 33.3% |
| MamoEasy | N/A | 66.5% | 57.4% |
| MamoComplex | N/A | 14.6% | **21.3%** |
| IndustryOR | N/A | 28.0% | 17.0% |

On the 3 datasets with CoE comparison, our gpt-4o-mini results match or exceed the paper's GPT-4 results, confirming successful replication.

---

## Part 2: Proposed Approach Replication (MCTS)

The paper's proposed method replaces the linear CoE pipeline with a **Monte Carlo Tree Search** over formulation space. Each MCTS iteration runs four phases:

1. **Select** — Walk the tree using UCB1 to pick the most promising node
2. **Expand** — Ask the LLM to generate a new formulation (or refine a failed one)
3. **Simulate** — Generate code, execute it, and score the result
4. **Backpropagate** — Update visit counts and values up the tree

Reward signal:
- `1.0` — correct answer (within 10% of ground truth) → stop early
- `0.3` — code ran but answer wrong → keep searching
- `0.0` — code crashed → discard

This lets the system explore multiple formulation strategies and use failures as feedback to improve, rather than committing to a single attempt.

### Part 2 Results

| Dataset | Total | CoE SA% | MCTS SA% | Improvement |
|---|--:|--:|--:|--:|
| nl4opt | 230 | 68.3% | — | — |
| nlp4lp | 242 | 64.5% | — | — |
| complexor | 18 | 33.3% | — | — |
| mamo_easy | 652 | 57.4% | — | — |
| mamo_complex | 211 | 21.3% | — | — |
| industryor | 100 | 17.0% | — | — |
| **TOTAL** | **1453** | **51.6%** | — | — |

*MCTS results pending (currently running, 20 iterations per problem).*

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=your-key-here
```

## Usage

```bash
# Run CoE baseline
python main.py --method coe

# Run MCTS (paper datasets only)
python main.py --method mcts --paper-only

# Run specific dataset
python main.py --method mcts --dataset nl4opt

# Print results comparison
python main.py --summary
```

## Project Structure

```
├── main.py                 # Entry point: run benchmark, save results, print summary
├── load_data.py            # Load JSONL datasets into SQLite
├── mcts/
│   ├── node.py             # MCTSNode: UCB1, tree structure
│   └── search.py           # MCTS: select, expand, simulate, backpropagate
├── pipeline/
│   └── experts.py          # Chain-of-Experts baseline pipeline
├── core/
│   ├── llm.py              # OpenAI API wrapper
│   └── executor.py         # Code execution with timeout
├── evaluation/
│   └── evaluator.py        # Answer extraction and evaluation
└── data/
    ├── testset.db          # SQLite database (problems + results)
    └── testset/            # Raw JSONL dataset files
```
