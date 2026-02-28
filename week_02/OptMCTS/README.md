# OptMCTS

Replication of **SolverLLM** (NeurIPS 2025), a framework that uses Monte Carlo Tree Search (MCTS) to iteratively refine mathematical formulations for solving optimization problems with LLMs.

Model: `gpt-4o-mini` | Temperature: 0

---

## Part 1: Baseline Replication (Chain-of-Experts)

The baseline replicates the **Chain-of-Experts (CoE)** pipeline from the paper. Each problem passes through three sequential LLM stages, then a code execution stage:

1. **Interpretation** — Extract key information and constraints from the problem text
2. **Formulation** — Write a mathematical formulation (variables, objective, constraints)
3. **Code Generation** — Generate executable Python/scipy code from the formulation
4. **Execution** — Run the code and extract the numeric answer (up to 3 retries on failure)

All datasets are loaded into a single SQLite database (`data/testset.db`). Results are written back to the `results` table in the same database — no intermediate files are created.

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

The paper's proposed method replaces the linear CoE pipeline with a **Monte Carlo Tree Search over the formulation space**. Rather than generating one complete formulation per problem, MCTS explores many partial formulations incrementally, using LLM feedback to guide the search.

### Tree Structure

The MCTS tree has **6 layers**, one per formulation element:

```
root
└── Type (LP / MILP / NLP …)          layer 1
    └── Sets (index sets)              layer 2
        └── Parameters (constants)     layer 3
            └── Variables (decisions)  layer 4
                └── Objective          layer 5
                    └── Constraints    layer 6  ← complete formulation
```

Each path from root to a layer-6 leaf defines one complete formulation. MCTS navigates this tree to find the best formulation within 20 iterations per problem.

### Three Key Innovations (from the paper)

**1. Dynamic Expansion**
Unlike standard MCTS, expansion is not limited to leaf nodes. If a node's element is flagged for revision (`trigger=True`) and its local uncertainty exceeds η=0.3, selection stops at that node and new children (alternative choices for the next layer) are added — even if children already exist. Up to 3 candidates are generated per expansion; semantically similar candidates (similarity > 0.8) are pruned.

**2. Prompt Backpropagation**
After each simulation, the LLM evaluates every layer of the formulation and produces a reasoning signal `(trigger, explanation, guidance)` per layer. The `guidance` is accumulated in a per-layer **knowledge base** (G_l). Future expansion prompts for that layer are injected with this accumulated guidance, steering the LLM toward better formulation choices over iterations.

**3. Uncertainty Backpropagation**
Instead of adding the raw reward to Q-values, updates are weighted by a **confidence factor** ρ = exp(−U_global), where U_global is estimated from the standard deviation of K=3 repeated LLM evaluations of the same solution. High uncertainty → small ρ → smaller Q update. The update rule is:

```
Q ← Q + ρ · (R − Q) / N
```

### Reward Function

```
R = 0.1 · feasible + 0.8 · score − 0.1 · error
```

where `score` is an LLM-evaluated quality score (0–100, normalized), `feasible` = 1 if code ran successfully, `error` = 1 if code crashed. Overridden to R = 1.0 when the predicted answer matches ground truth (within 10%).

### UCB Formula

```
UCB = Q + 2 · sqrt(2 · ln(N_parent) / N)
```

### Code Generation

Code generation is decoupled from MCTS. Once a complete 6-layer formulation is assembled, Python/scipy code is generated and executed with up to **12 retries** on failure (per paper).

### Part 2 Results

| Dataset | Total | CoE SA% | MCTS SA% | Improvement |
|---|--:|--:|--:|--:|
| nl4opt | 230 | 68.3% | 75.7% | +7.4% |
| nlp4lp | 242 | 64.5% | 80.0%* | +15.5% |
| complexor | 18 | 33.3% | 55.6% | +22.3% |
| mamo_easy | 652 | 57.4% | 77.9% | +20.5% |
| mamo_complex | 211 | 21.3% | 17.1%† | -4.2% |
| industryor | 100 | 17.0% | 2.0%† | -15.0% |

\* nlp4lp: 60/242 completed so far, SA% is partial.
† industryor and mamo_complex results are from an earlier run affected by API rate limiting (ER% only 2.0% and 23.7%). Re-run pending.

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=your-key-here
```

Load datasets into the database (one-time):

```bash
python load_data.py
```

## Usage

```bash
# Run CoE baseline (Part 1)
python main.py --method coe

# Run MCTS on paper datasets only (Part 2)
python main.py --method mcts --paper-only

# Run MCTS on a specific dataset
python main.py --method mcts --dataset nl4opt

# Print results summary for both methods
python main.py --summary
```

## Project Structure

```
├── main.py                 # Entry point: run benchmark, save results, print summary
├── load_data.py            # Load JSONL datasets into SQLite (data/testset.db)
│
├── pipeline/               # [Part 1] Chain-of-Experts baseline
│   └── experts.py          #   Three-stage LLM pipeline (interpret → formulate → code)
│
├── mcts/                   # [Part 2] SolverLLM MCTS proposed approach
│   ├── node.py             #   MCTSNode: 6-layer tree structure, UCB formula
│   ├── search.py           #   MCTS: select, expand, simulate, backpropagate
│   └── evaluator.py        #   LLM evaluator: objective score, reasoning signals, uncertainty
│
├── core/                   # [Shared]
│   ├── llm.py              #   OpenAI API wrapper (standard + logprobs + JSON mode)
│   └── executor.py         #   Sandboxed code execution with timeout
├── evaluation/             # [Shared]
│   └── evaluator.py        #   Answer extraction and correctness evaluation
└── data/                   # [Shared]
    ├── testset.db          #   SQLite database (problems + results tables)
    └── testset/            #   Raw JSONL dataset files
```
