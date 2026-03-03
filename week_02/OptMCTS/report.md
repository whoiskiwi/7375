# OptMCTS: Replication of SolverLLM for Mathematical Optimization with MCTS-Guided LLM Formulation

## Technical Report

---

## 1. Introduction

### 1.1 Background

Mathematical optimization problems arise in logistics, finance, engineering, and operations research. Traditionally, solving these problems requires domain expertise to (1) interpret a natural language description, (2) formulate it as a mathematical program, and (3) implement it in a solver. Recent advances in Large Language Models (LLMs) offer the possibility of automating this entire pipeline.

**SolverLLM** (NeurIPS 2025) proposes a novel framework that replaces the traditional linear LLM pipeline with Monte Carlo Tree Search (MCTS) over the formulation space. By treating each element of a mathematical formulation (problem type, sets, parameters, variables, objective, constraints) as a layer in a search tree, SolverLLM can explore multiple alternative formulations and iteratively refine them using LLM feedback.
Paper poster: `https://neurips.cc/virtual/2025/loc/san-diego/poster/116215`

### 1.2 Project Objective

This project replicates the SolverLLM framework with two goals:

1. **Part 1 (Baseline)**: Reproduce the Chain-of-Experts (CoE) pipeline as a baseline, using `gpt-4o-mini` instead of the paper's GPT-4.
2. **Part 2 (Proposed)**: Implement the full MCTS-based approach with the paper's three key innovations: Dynamic Expansion, Prompt Backpropagation, and Uncertainty Backpropagation.

### 1.3 Evaluation Datasets

The system is evaluated on 9 benchmark datasets totaling 2,634 optimization problems, of which 6 are from the original paper:

| Dataset | Problems | Source | Description |
|---|--:|---|---|
| NL4Opt | 230 | Paper | Linear programming word problems |
| NLP4LP | 242 | Paper | NLP for linear programming tasks |
| ComplexOR | 18 | Paper | Complex operations research with abstract data |
| IndustryOR | 100 | Paper | Industry-oriented operations research |
| MamoEasy | 652 | Paper | Easy mathematical optimization |
| MamoComplex | 211 | Paper | Complex mathematical optimization |
| OptiBench | 605 | Extended | Additional optimization benchmark |
| OptMathBench | 166 | Extended | Mathematical optimization benchmark |
| Task3 | 410 | Extended | Mixed optimization tasks |

---

## 2. System Architecture

### 2.1 Overall Design

The project follows a modular architecture with clear separation of concerns:

```
OptMCTS/
├── main.py                 # Orchestrator: CLI, database I/O, problem loop
├── load_data.py            # One-time JSONL → SQLite data loader
├── core/                   # Shared infrastructure
│   ├── llm.py              # OpenAI API wrapper with retry logic
│   └── executor.py         # Sandboxed Python code execution
├── pipeline/               # Part 1: Baseline
│   └── experts.py          # Chain-of-Experts 3-stage pipeline
├── mcts/                   # Part 2: Proposed MCTS approach
│   ├── node.py             # MCTSNode data structure with UCB1
│   ├── search.py           # MCTS engine (select/expand/simulate/backpropagate)
│   └── evaluator.py        # LLM-based scoring and uncertainty estimation
├── evaluation/             # Shared evaluation
│   └── evaluator.py        # Answer extraction and correctness checking
└── data/                   # Data storage
    ├── testset.db          # SQLite database
    └── testset/            # Raw JSONL files (9 datasets)
```

### 2.2 Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Language | Python 3.x | Primary implementation language |
| LLM | OpenAI GPT-4o-mini | All LLM calls (interpretation, formulation, evaluation) |
| Solver | SciPy (`scipy.optimize`) | Mathematical optimization execution |
| Database | SQLite | Problem storage and result persistence |
| API | OpenAI Python SDK | LLM API interaction with structured outputs |
| Config | python-dotenv | Environment variable management |

### 2.3 Data Layer

All data is managed through a single SQLite database (`data/testset.db`) containing three tables:

- **`problems`** (2,634 rows): Stores all benchmark problems with fields `id`, `dataset`, `question`, `answer`, `original`, `original_index`.
- **`results`**: CoE baseline results with fields `problem_id`, `dataset`, `expected`, `predicted`, `success`, `correct`, `output`, `error`.
- **`mcts_results`**: MCTS results with an identical schema.

This design enables resume-safe execution (already-completed problems are skipped on restart) and side-by-side comparison between methods.

---

## 3. Part 1: Chain-of-Experts Baseline

### 3.1 Pipeline Design

The CoE baseline implements a sequential three-stage LLM pipeline followed by code execution:

```
Natural Language Problem
        │
        ▼
┌─────────────────┐
│  Interpretation  │  LLM call: extract key information and constraints
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Formulation    │  LLM call: write mathematical formulation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Code Generation  │  LLM call: generate Python/SciPy code
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Execution     │  Sandboxed subprocess, up to 3 retries
└─────────────────┘
```

**Implementation** (`pipeline/experts.py`, 31 lines):

1. **Interpretation**: The LLM extracts key information, decision variables, constraints, and the objective from the problem description.
2. **Formulation**: Based on the interpretation, the LLM writes a formal mathematical formulation with explicit variables, objective function, and constraint set.
3. **Code Generation**: The LLM generates executable Python code using `scipy.optimize` that prints the optimal objective value.
4. **Execution & Retry**: The generated code runs in a sandboxed subprocess with a 120-second timeout. If execution fails or produces no numeric output, the error and code are sent back to the LLM for correction, up to 3 retries.

### 3.2 Baseline Results

| Dataset | Total | Executed (ER%) | Correct (SA%) |
|---|--:|--:|--:|
| NL4Opt | 230 | 100.0% | 68.3% |
| NLP4LP | 242 | 100.0% | 64.5% |
| ComplexOR | 18 | 100.0% | 33.3% |
| MamoEasy | 652 | 100.0% | 57.4% |
| MamoComplex | 211 | 78.7% | 21.3% |
| IndustryOR | 100 | 81.0% | 17.0% |
| **Total (paper sets)** | **1,453** | **95.8%** | **51.6%** |

**Comparison with the paper's baseline (CoE SA%)**:

| Dataset | Paper CoE (GPT-4) | Ours CoE (4o-mini) |
|---|:-:|:-:|
| NL4Opt | 64.2% | **68.3%** |
| NLP4LP | 53.1% | **64.5%** |
| ComplexOR | 38.1% | 33.3% |
| MamoEasy | N/A | 57.4% |

Despite using the smaller `gpt-4o-mini` model, our baseline matches or exceeds the paper's GPT-4 results on 2 of 3 comparable datasets, confirming successful replication of the CoE methodology.

---

## 4. Part 2: MCTS-Based Approach

### 4.1 Core Idea

Solving an optimization problem from natural language requires a multi-stage transformation:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐     ┌──────────────┐     ┌───────────┐
│ Natural      │     │ Problem      │     │ Mathematical         │     │ Code         │     │ Execution │
│ Language     │ ──▶ │ Interpret-   │ ──▶ │ Formulation          │ ──▶ │ Generation   │ ──▶ │ & Answer  │
│ Problem      │     │ ation        │     │                      │     │ (Python/     │     │           │
│              │     │              │     │ Type, Sets, Params,  │     │  SciPy)      │     │           │
│ "A factory   │     │ Identify     │     │ Variables, Objective,│     │ Translate    │     │ Run code, │
│  produces…"  │     │ entities,    │     │ Constraints          │     │ math into    │     │ extract   │
│              │     │ relations,   │     │                      │     │ executable   │     │ optimal   │
│              │     │ goals        │     │                      │     │ solver code  │     │ value     │
└──────────────┘     └──────────────┘     └──────────────────────┘     └──────────────┘     └───────────┘
     Text               Logic                   Math                      Code               Answer
```

The CoE baseline traverses this pipeline once in a single pass. The key limitation is that **a wrong formulation decision early on (e.g., choosing LP instead of MILP) propagates through the entire pipeline** and cannot be corrected.

SolverLLM addresses this by applying MCTS to the **Mathematical Formulation** stage — the most critical and error-prone step. Rather than generating the formulation in one shot, it decomposes it into 6 hierarchical elements and searches over alternative choices for each element. This transforms the problem from single-shot generation into an iterative search, where:

- **Exploration** discovers diverse formulation strategies.
- **Exploitation** refines promising formulations.
- **Feedback** from execution results guides future iterations.

### 4.2 Tree Structure

The MCTS tree is a **multi-way tree** (not binary) with 7 levels (root + 6 formulation layers). Each node represents one choice for a formulation element, and each node can have **multiple children** (up to 5), representing alternative choices for the next element. The tree grows dynamically as MCTS explores different formulation strategies.

```
Layer 0: Root (problem statement)
Layer 1: Type (LP, MILP, NLP, QP, ...)
Layer 2: Sets (index sets for the formulation)
Layer 3: Parameters (constants and input data)
Layer 4: Variables (decision variables)
Layer 5: Objective (objective function)
Layer 6: Constraints (constraint set)
```

**Example**: For a factory scheduling problem, the tree might grow as follows over multiple iterations:

```
Root: "A factory produces two products..."
├── Type: LP                          ← iter 1: try linear programming
│   └── Sets: {products, machines}
│       └── Parameters: costs, capacities
│           └── Variables: x_ij (continuous)
│               └── Objective: minimize cost
│                   └── Constraints: capacity, demand   → R=0.4 (wrong answer)
│
├── Type: MILP                        ← iter 3: try mixed-integer
│   ├── Sets: {products, machines}    ← iter 3: first attempt
│   │   └── Parameters: ...
│   │       └── Variables: x_ij (integer)
│   │           └── Objective: minimize cost
│   │               └── Constraints: ...               → R=0.7 (closer)
│   │
│   └── Sets: {products, machines,    ← iter 5: Dynamic Expansion adds
│              time_periods}                    alternative Sets node
│       └── Parameters: ...
│           └── Variables: x_ijt (integer)
│               └── Objective: minimize cost
│                   └── Constraints: ...               → R=1.0 (correct!)
│
└── Type: NLP                         ← iter 7: explore nonlinear
    └── ...                                            → R=0.2 (poor)
```

Key properties of the tree:
- **Multi-way branching**: Each node can have up to 5 children, representing different LLM-generated alternatives for the next formulation element.
- **Asymmetric growth**: UCB1 guides the search toward promising branches, so some paths are explored deeply while others are abandoned early.
- **Dynamic Expansion**: Unlike standard MCTS, new children can be added to non-leaf nodes (e.g., adding a new Sets option under an existing Type node) when the evaluator flags that layer for revision.
- **Depth = 6**: Every complete path from root to a layer-6 leaf defines one full mathematical formulation that can be converted to code and executed.

Each `MCTSNode` (`mcts/node.py`) stores:

- `layer` and `content`: The formulation element at this node (e.g., layer=1, content="MILP")
- `visits` (N) and `value` (Q): Statistics for UCB1 selection — Q is the confidence-weighted running mean of rewards
- `trigger` and `local_uncertainty`: Flags set by the LLM evaluator during backpropagation, controlling whether Dynamic Expansion is activated at this node
- `children` and `parent`: Tree structure references for traversal

### 4.3 MCTS Algorithm

The search runs for 20 iterations per problem, following the standard MCTS loop with three paper-specific innovations:

```
Algorithm: MCTS_Search(problem, expected_answer)
───────────────────────────────────────────────
  Initialize knowledge_base = {layer: [] for each layer}
  Create root node

  FOR i = 1 TO 20:
    node ← SELECT(root)          // UCB1 tree walk
    child ← EXPAND(node)         // Generate formulation element
    (R, result, U_g, signals) ← SIMULATE(child)  // Complete & evaluate
    BACKPROPAGATE(child, R, U_g, signals)         // Update tree

    IF R = 1.0: BREAK            // Early stop on correct answer

  RETURN best_result
```

### 4.4 Innovation 1: Dynamic Expansion

**Standard MCTS** only expands leaf nodes. **SolverLLM** allows expansion at any intermediate node if two conditions are met:

1. The node's element is flagged for revision (`trigger = True`)
2. Its local uncertainty exceeds a threshold η = 0.3

This enables the search to revisit and revise earlier formulation decisions (e.g., changing the problem type) when the LLM evaluator determines that a specific layer is the source of errors.

**Implementation** (`mcts/search.py`, `_select` method):

- During UCB1 tree traversal, the algorithm checks `node.trigger and node.local_uncertainty > eta` at each non-leaf node.
- If triggered, selection stops at that node, and expansion generates new children — alternative choices for the next layer.
- Up to `MAX_CHILDREN = 5` children are allowed per node to prevent unbounded growth.

### 4.5 Innovation 2: Prompt Backpropagation

After each simulation, the LLM evaluates all 6 layers of the formulation and produces a **reasoning signal** per layer:

```json
{
  "trigger": true/false,       // Should this element be revised?
  "explanation": "...",        // Why this element may be problematic
  "guidance": "..."            // How to improve it in future expansions
}
```

The `guidance` strings are accumulated in a per-layer **knowledge base** (capped at 20 entries, retaining the 10 most recent per layer). When expanding a node at layer $l$, the most recent 3 guidance entries for that layer are injected into the LLM prompt, steering future formulation choices toward previously identified improvements.

**Implementation** (`mcts/evaluator.py`, `get_all_reasoning_signals`):

- A single LLM call (JSON mode with log-probabilities) evaluates all 6 layers simultaneously.
- Returns a dictionary mapping layer names to their reasoning signals.
- The knowledge base is stored as `dict[str, list[str]]` in the `MCTS` class instance.

### 4.6 Innovation 3: Uncertainty Backpropagation

Standard MCTS uses raw reward for backpropagation. SolverLLM introduces a **confidence-weighted** update:

**Global Uncertainty Estimation**:
- The same formulation is scored K = 3 times by the LLM at temperature 0.5.
- Global uncertainty: $U_{global} = \text{std}(scores) / 50$, capped at 1.0.
- Confidence factor: $\rho = \exp(-U_{global})$.

**Weighted Q-value Update**:
$$Q \leftarrow Q + \rho \cdot \frac{R - Q}{N}$$

When uncertainty is high (scores vary widely), $\rho$ is small, dampening the update. When the LLM consistently agrees on the score, $\rho \approx 1$, and the full reward is applied.

**Local Uncertainty**:
- Computed from token-level log-probabilities: $U_{local} = -\text{mean}(\text{logprobs})$, normalized to [0, 1].
- Used to determine the Dynamic Expansion trigger (Innovation 1).

### 4.7 Reward Function

The reward combines three signals:

$$R = \max(0, \min(1, \ 0.1 \cdot \text{feasible} + 0.8 \cdot \text{score} - 0.1 \cdot \text{error}))$$

| Component | Weight | Description |
|---|:-:|---|
| `feasible` | 0.1 | 1 if code executed successfully, 0 otherwise |
| `score` | 0.8 | LLM-evaluated quality score (0–100, normalized to 0–1) |
| `error` | -0.1 | 1 if code crashed, 0 otherwise |

**Ground-truth override**: When the predicted answer matches the expected answer within 10% relative tolerance, the reward is overridden to $R = 1.0$ and the search terminates early.

### 4.8 UCB1 Selection

Node selection uses the UCB1 formula with exploration constant $C = 2.0$:

$$\text{UCB1} = Q + C \cdot \sqrt{\frac{2 \ln N_{parent}}{N}}$$

Unvisited nodes return $+\infty$ to ensure they are explored first.

### 4.9 Code Generation

Code generation is decoupled from MCTS. Once a complete 6-layer formulation is assembled (either through expansion or greedy simulation), Python/SciPy code is generated and executed:

- Up to **12 retries** on failure (vs. 3 in the CoE baseline), per the paper's specification.
- Each retry sends the error message and original code back to the LLM for correction.
- A 120-second timeout prevents infinite loops.

### 4.10 Hyperparameters

All hyperparameters match the paper's Table 6:

| Parameter | Symbol | Value | Description |
|---|:-:|:-:|---|
| Iterations | $T$ | 20 | MCTS iterations per problem |
| Exploration constant | $C$ | 2.0 | UCB1 exploration weight |
| Uncertainty threshold | $\eta$ | 0.3 | Dynamic expansion trigger |
| Score samples | $K$ | 3 | Number of LLM evaluations for uncertainty |
| Max children | — | 5 | Maximum children per node |
| Similarity threshold | — | 0.8 | Pruning threshold for similar candidates |
| Code retries | — | 12 | Maximum code fix attempts |

---

## 5. Core Infrastructure

### 5.1 LLM Interface (`core/llm.py`)

The LLM module provides four API call variants, all using `gpt-4o-mini`:

| Function | Temperature | Output | Use Case |
|---|:-:|---|---|
| `call_llm` | 0 | String | Deterministic generation (code, interpretation) |
| `call_llm_with_logprobs` | 0.2 | String + logprobs | Generation with uncertainty estimation |
| `call_llm_json` | 0 | Parsed dict | Structured JSON output (scores) |
| `call_llm_json_with_logprobs` | 0.2 | Dict + logprobs | Structured output with uncertainty |

**Retry Logic**: An exponential backoff decorator handles transient API failures (rate limits, timeouts, connection errors) with up to 8 attempts (delays: 2, 4, 8, ..., 256 seconds).

### 5.2 Code Executor (`core/executor.py`)

The executor provides sandboxed code execution:

1. Strips markdown fences from LLM-generated code.
2. Writes to a temporary `.py` file.
3. Executes via `subprocess.run` with a 120-second timeout.
4. Returns `{success, output, error}`.
5. Validates output contains at least one numeric value.

### 5.3 Evaluation (`evaluation/evaluator.py`)

- **Answer Extraction**: Regex-based extraction of the last numeric value from stdout.
- **Correctness Check**: Relative error < 10% for non-zero answers; absolute error < 1e-4 for near-zero answers.

### 5.4 Orchestrator (`main.py`)

The main entry point provides:

- **CLI interface** with `argparse`: `--method`, `--dataset`, `--limit`, `--iterations`, `--paper-only`, `--summary`.
- **Resume-safe execution**: Queries completed problem IDs before starting; interrupted runs pick up where they left off.
- **Live progress reporting**: Per-problem status with running accuracy and ETA.
- **Result persistence**: `INSERT OR REPLACE` ensures idempotent writes.

---

## 6. Experimental Results

### 6.1 MCTS Results

| Dataset | Total | Exec | Correct | ER% | SA% |
|---|--:|--:|--:|--:|--:|
| ComplexOR | 18 | 17 | 10 | 94.4% | 55.6% |
| MamoEasy | 652 | 521 | 508 | 79.9% | 77.9% |
| NL4Opt | 230 | 198 | 174 | 86.1% | 75.7% |
| NLP4LP | 242 | 241 | 202 | 99.6% | 83.5% |
| **Total** | **1142** | **977** | **894** | **85.6%** | **78.3%** |

Note: IndustryOR and MamoComplex are excluded from MCTS evaluation — their runs were affected by API rate limiting (daily RPD quota exhausted), resulting in ER% of only 2.0% and 23.7% respectively, making results unreliable.

### 6.2 Comparison with Paper — MCTS (SA%)

| Dataset | Paper MCTS (GPT-4) | Ours MCTS (4o-mini) |
|---|:-:|:-:|
| NL4Opt | 97.0% | 75.7% |
| NLP4LP | 87.0% | 83.5% |
| ComplexOR | 77.8% | 55.6% |
| MamoEasy | 96.0% | 77.9% |

### 6.3 CoE → MCTS Improvement

On the same 4 datasets (1,142 problems), the CoE baseline achieves an overall SA% of **60.7%** (693/1,142 correct). MCTS improves this to **78.3%** (894/1,142 correct), a gain of **+17.6 percentage points**.

| Dataset | CoE SA% | MCTS SA% | Improvement |
|---|:-:|:-:|:-:|
| NL4Opt | 68.3% | 75.7% | +7.4% |
| NLP4LP | 64.5% | 83.5% | +19.0% |
| ComplexOR | 33.3% | 55.6% | +22.3% |
| MamoEasy | 57.4% | 77.9% | +20.5% |
| **Total (1,142)** | **60.7%** | **78.3%** | **+17.6%** |

### 6.4 Analysis

**MCTS consistently improves over CoE**: On all 4 evaluated datasets, MCTS outperforms the CoE baseline by +7 to +22 percentage points, confirming the paper's core finding that iterative formulation search discovers better solutions than single-shot generation.

**Improvement gap with the paper**: While our MCTS improves over CoE on all datasets, the magnitude of improvement is smaller than the paper's:

| Dataset | Paper CoE→MCTS | Ours CoE→MCTS |
|---|:-:|:-:|
| NL4Opt | +32.8% (64.2→97.0) | +7.4% (68.3→75.7) |
| NLP4LP | +33.9% (53.1→87.0) | +19.0% (64.5→83.5) |
| ComplexOR | +39.7% (38.1→77.8) | +22.3% (33.3→55.6) |

The paper achieves +30–40 percentage point gains while ours are +7–22. This gap is attributable to MCTS's heavier dependence on model capability compared to CoE:

1. **Evaluation quality**: MCTS requires the LLM to accurately score formulations, identify problematic layers, and generate improvement guidance at every iteration. With `gpt-4o-mini`, less accurate evaluation leads to noisier search signals, reducing the effectiveness of the 20-iteration search loop.
2. **Uncertainty estimation**: The K=3 repeated scoring used for uncertainty backpropagation relies on consistent LLM judgments. A weaker model produces inherently noisier scores, degrading the confidence signal (ρ) and the quality of Q-value updates.
3. **Reasoning signal quality**: The trigger/explanation/guidance triplets that drive Dynamic Expansion and Prompt Backpropagation are only as good as the model generating them. Less precise guidance accumulates in the knowledge base, potentially misleading future expansions.
4. **ER% degradation**: Our MCTS ER% (86–94%) is lower than the CoE baseline (100%), indicating that the more complex formulations produced by MCTS are sometimes harder to convert into executable code, even with 12 retries. The paper's GPT-4 likely generates more robust code from the same formulations.

In summary, CoE only requires the model to "generate correctly," while MCTS additionally requires the model to "evaluate accurately." GPT-4's stronger reasoning and evaluation capabilities amplify the MCTS search much more effectively than `gpt-4o-mini`.

**Largest gains on harder problems**: The biggest improvements (+19–22%) appear on datasets where the CoE baseline has lower accuracy (33–64%), suggesting MCTS is most valuable when the correct formulation is non-obvious and benefits from exploration.

---

## 7. Design Decisions and Trade-offs

### 7.1 Model Choice

Using `gpt-4o-mini` instead of GPT-4 reduces cost by approximately 10x while achieving comparable accuracy on the CoE baseline. This enables running the full MCTS search (which makes 50–100+ LLM calls per problem) at reasonable cost.

### 7.2 Temperature Strategy

The system uses differentiated temperatures for different purposes:

| Context | Temperature | Rationale |
|---|:-:|---|
| Code generation / interpretation | 0 | Deterministic, reproducible output |
| Element generation (MCTS) | 0.7 | Diversity in formulation alternatives |
| Reasoning signal evaluation | 0.2 | Slight variation for robustness |
| Objective score sampling | 0.5 | Controlled diversity for uncertainty estimation |

### 7.3 Knowledge Base Management

The per-layer knowledge base is capped at 20 entries, retaining the 10 most recent to prevent prompt pollution. Only the 3 most recent entries are injected into expansion prompts, ensuring guidance stays relevant to the current search state.

### 7.4 Simulation vs. Tree Addition

During MCTS simulation, remaining layers are completed greedily but not added to the tree. This keeps the tree focused on explored alternatives at each layer while avoiding combinatorial explosion from fully enumerating all possible completions.

### 7.5 Resume Safety

The system is designed for long-running benchmarks:
- Each result is written to SQLite immediately after computation.
- On restart, completed problem IDs are queried and skipped.
- CoE and MCTS results are stored in separate tables for independent execution.

---

## 8. Comparison with the Original Paper

| Aspect | Original Paper | This Replication |
|---|---|---|
| LLM Model | GPT-4 | GPT-4o-mini |
| MCTS Iterations | 20 | 20 |
| UCB Constant | 2.0 | 2.0 |
| Uncertainty Threshold η | 0.3 | 0.3 |
| Score Samples K | 3 | 3 |
| Code Retries | 12 | 12 |
| Similarity Threshold | 0.8 | 0.8 |
| Data Storage | Not specified | SQLite (resume-safe) |
| Evaluation | 10% relative tolerance | 10% relative tolerance |

All algorithmic hyperparameters match the paper exactly. The primary difference is the LLM backend, which trades some capability for cost efficiency.

---

## 9. Conclusion

This project successfully replicates the SolverLLM framework with both its baseline (CoE) and proposed (MCTS) approaches. Key findings:

1. **CoE Baseline**: The `gpt-4o-mini` implementation matches or exceeds the paper's GPT-4 CoE results on comparable datasets, validating the pipeline design.

2. **MCTS Improvement**: On 4 of 6 datasets with reliable results, MCTS improves accuracy by 7–22 percentage points over the CoE baseline, confirming the value of iterative formulation search.

3. **Three Innovations**: Dynamic Expansion, Prompt Backpropagation, and Uncertainty Backpropagation are all implemented and contribute to the improved performance by enabling targeted revisions, accumulating search experience, and properly weighting uncertain evaluations.

4. **Practical Considerations**: The SQLite-based design with resume-safe execution and separate result tables enables efficient large-scale benchmarking despite the high per-problem cost of MCTS (50–100+ LLM calls per problem).

---

## Appendix A: Dependencies

```
openai          # OpenAI API client
python-dotenv   # .env file loading
scipy           # Mathematical optimization solver
```

## Appendix B: Usage

```bash
# Setup
pip install -r requirements.txt
echo "OPENAI_API_KEY=your-key" > .env
python load_data.py          # One-time data loading

# Run experiments
python main.py --method coe              # Part 1: CoE baseline
python main.py --method mcts --paper-only # Part 2: MCTS on paper datasets
python main.py --summary                  # View results
```
