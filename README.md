# TrustMarine-AI
### Trustworthy Agentic AI for Maritime Condition Monitoring and Predictive Maintenance

**Independent research and portfolio project · v0.2 · Python / Scikit-learn / Ollama**

> **What is this?** A personal, portfolio-oriented research prototype demonstrating how an open-weight language model can use restricted engineering tools to inspect synthetic shipboard sensor data, identify possible anomalies, and explain its evidence while keeping safety-related decisions under human oversight. **It is not a deployed or validated maritime maintenance system.**

## 1. Why this project exists

A ship's electrical and machinery systems produce many measurements. A high bearing-vibration reading, low oil pressure, or unstable bus frequency might indicate a problem, but a single reading does not establish its cause. A useful diagnostic assistant should **check data, call analysis tools, explain its reasoning, acknowledge uncertainty, and escalate uncertain or abnormal cases to an engineer**.

A large language model (LLM) alone cannot reliably measure equipment condition: it predicts text and may generate unsupported conclusions. TrustMarine-AI therefore separates **numerical evidence and safety gates (Python)** from **tool selection and natural-language synthesis (LLM)**.

The central research question is:

> Can a locally deployed, tool-using LLM produce traceable, reviewable maritime diagnostic summaries without being trusted to make safety-critical decisions?

## 2. System architecture

**Architecture (text):** Synthetic sensor readings → deterministic anomaly detection and diagnostic tools → optional Ollama tool-calling agent → host-enforced human review → JSON audit record.

The workflow is:

1. **Synthetic vessel sensor generator** creates labeled normal and fault examples.
2. **Anomaly detector** learns a baseline from normal data and flags unusual samples.
3. **Deterministic diagnostic tools** expose readings, anomaly scores, heuristic fault evidence, and example maintenance notes.
4. **Ollama-hosted LLM** chooses which *allowlisted, read-only* tools to call and summarizes the evidence.
5. **Host-enforced safety gate** requires qualified engineer review when a baseline anomaly is detected or the LLM workflow is incomplete.
6. **Audit trail** stores the tool calls and returned data for later inspection.

**Important distinction:** The LLM does **not** directly operate machinery, issue control commands, or decide whether a ship is safe.

## 3. What data does the prototype analyze?

The current generator (`src/core.py`) creates five synthetic measurements. The simulation draws baseline values from probability distributions and applies scenario-specific shifts to approximate faulty operation. It does not simulate full physical dynamics:

| Sensor | Unit | Why it matters in this simplified model |
|---|---|---|
| Generator load | kW | Context for load-dependent operating conditions |
| Coolant temperature | °C | Elevated temperature may suggest cooling degradation |
| Bearing vibration | mm/s | Elevated vibration may suggest bearing wear |
| Oil pressure | bar | Reduced pressure may suggest lubrication issues |
| Bus frequency | Hz | Deviations may suggest electrical/load instability |

Five classes are generated: `normal`, `cooling_degradation`, `bearing_wear`, `lubrication_loss`, and `load_instability`. Faults are simulated by shifting selected sensor distributions. These are **illustrative statistical scenarios**, not measurements from real vessels or physics-validated digital twins.

## 4. How does anomaly detection work?

The project compares two approaches:

**Fixed-threshold baseline.** A binary decision is obtained by comparing selected sensor values to fixed numerical limits. In mathematical terms, a reading is flagged if it crosses one of the selected upper or lower bounds. Thresholds are transparent but may not generalize to varying operating regimes.

**Fixed-threshold baseline.** A transparent rule flags selected measurements that exceed predefined limits. This is easy to interpret but can miss subtle, multivariate anomalies.

**Isolation Forest.** Isolation Forest builds random partitioning trees; observations that can be isolated with fewer splits are treated as more unusual. Here, the model uses 200 trees and a contamination setting of 0.02, fitted on synthetic normal readings. It does not know the fault class during training.

**Isolation Forest.** An unsupervised model is trained on *normal* samples. It scores how unusual a new sensor vector looks relative to that baseline. Its raw score is **not a calibrated probability of equipment failure**.

A separate heuristic diagnosis uses standardized deviations from normal readings (z-scores) to rank possible fault categories. This ranking is an **illustrative hypothesis**, not a verified root-cause diagnosis.

### Experimental results: synthetic binary anomaly detection

The repository includes the original synthetic evaluation results (`results/metrics.csv`). These numbers assess the **anomaly detectors**, not the LLM.

| Detector | Precision | Recall | F1 |
|---|---:|---:|---:|
| Fixed threshold | 1.000 | 0.786 | 0.880 |
| Isolation Forest | 0.990 | 0.338 | 0.503 |





**Confusion-matrix terms:** TP = a real synthetic anomaly correctly flagged; FP = normal data incorrectly flagged; FN = a synthetic anomaly missed; TN = normal data correctly left unflagged. Precision = TP/(TP+FP), recall = TP/(TP+FN), and F1 = 2×precision×recall/(precision+recall). The CSV contains the underlying counts.

**How to interpret the metrics:** Precision asks *“Of the alarms, how many were truly anomalous?”* Recall asks *“Of the anomalous samples, how many were detected?”* F1 balances the two. Here, Isolation Forest missed many synthetic faults (low recall), so it should **not** be described as superior to the threshold baseline. The dataset design favors some easily thresholded faults; more varied conditions and temporal features are needed.

## 5. Where does the real LLM fit?

The LLM integration is implemented in `src/llm_agent.py` using Ollama's local `/api/chat` interface. It supports a tool-calling-capable open-weight model such as `qwen2.5:7b`.

The model may request only these tools:

| Tool | What it returns |
|---|---|
| `inspect_sensors` | The five numeric sensor readings |
| `detect_anomaly` | Detector flag and raw anomaly score |
| `diagnostic_baseline` | Rule-based fault hypothesis, evidence, and human-review flag |
| `get_maintenance_note` | A static, synthetic reference note for an allowed topic |

**Example conceptual sequence (illustrative, not a recorded real-LLM result):**

```text
User: Assess this synthetic generator sample.
LLM -> inspect_sensors()
Tool -> coolant temperature, vibration, oil pressure, frequency, load
LLM -> detect_anomaly()
Tool -> anomaly flag and raw score
LLM -> diagnostic_baseline()
Tool -> possible fault, numeric evidence, engineer-review flag
LLM -> get_maintenance_note(topic="bearing")
Tool -> synthetic reference note
LLM -> evidence-based summary, limitations, and request for engineer review
```

The host validates tool names and arguments, limits tool calls, and logs responses. The LLM's narrative is **not itself a verified engineering conclusion**. Maintenance notes are treated as untrusted information rather than executable instructions.

## 6. Why is it called *Trustworthy*?

The key architectural principle is **separation of authority**. The language model may propose interpretations and choose among predefined information tools, but a separate Python component evaluates whether the sample should be escalated. This matters because natural-language output can sound confident even when it is wrong. A safety-critical decision must not depend solely on whether the model writes reassuring text.


The project implements some **trustworthiness-oriented safeguards**, not a formal guarantee of trustworthiness:

- **Tool allowlist:** The model can only call specified read-only tools.
- **Auditability:** Each tool call, input argument, result, and status is recorded.
- **Bounded execution:** The agent has a maximum tool-call budget.
- **Independent human-review gate:** Model-generated prose cannot override the host-side review flag.
- **Uncertainty-aware wording:** Prompts instruct the model to distinguish observed readings from heuristic hypotheses.
- **Local model option:** Ollama can run a model locally without sending samples to a cloud LLM service.

These mechanisms are *design features*. Robustness against hallucinations, prompt injection, and incorrect tool choices has **not yet been established through real-model benchmarking**.

## 7. Open the project and inspect it

Download or clone the repository, then **open the `TrustMarine-AI` folder** in VS Code or another editor. Recommended reading order:

1. `README.md` — problem, architecture, evidence, and limitations.
2. `src/core.py` — synthetic sensor generation, anomaly detection, and deterministic assessment.
3. `run_experiment.py` — dataset split, comparison, and result generation.
4. `src/llm_agent.py` — Ollama API client, tool schemas, tool execution, audit trail, and review gate.
5. `run_llm.py` — command-line example of a local LLM diagnostic run.
6. `tests/` — automated software tests.
7. `assets/` and `results/` — architecture figures, plots, and saved experiment outputs.

### Installation and baseline experiment

Python 3.10+ is recommended.

```bash
git clone https://github.com/Mates3834/TrustMarine-AI.git
cd TrustMarine-AI
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
python run_experiment.py
python -m pytest -q
```

If the repository is downloaded as ZIP instead, extract it and open the **inner `TrustMarine-AI` folder** before running the commands. The `git clone` URL works after this repository has been published under the shown GitHub account.

### Run with a real local LLM

Install [Ollama](https://ollama.com/). In a terminal:

```bash
ollama pull qwen2.5:7b
ollama serve
```

In another terminal, from the project folder:

```bash
python run_llm.py --model qwen2.5:7b --fault bearing_wear
```

If Ollama is already running, a second `ollama serve` is unnecessary. Inspect the generated `results/llm_run.json` to see the response, tools used, and audit trace. **A real Ollama inference run has not been verified in the supplied project environment**; results will depend on your installed model and hardware.

## 8. What has actually been verified?

| Component | Current status |
|---|---|
| Synthetic data generator | Implemented |
| Fixed-threshold and Isolation Forest evaluation | Measured on synthetic samples |
| Evaluation plots and CSV | Included in this repository |
| Host-side tool orchestration | Implemented; automated tests included |
| Automated tests | 6 passed in the preparation environment |
| Real Ollama model run | Not yet verified |
| LLM tool-selection accuracy | Not measured |
| Hallucination / evidence fidelity | Not measured |
| Prompt-injection robustness | Not measured |
| Real vessel validation | Not performed |

## 9. Research limitations and next steps

**What the project demonstrates versus what it does not:** It demonstrates a modular software architecture, reproducible synthetic-data anomaly evaluation, and a tool-calling interface designed for a locally hosted LLM. It does **not** demonstrate that the LLM has high diagnostic accuracy, that the synthetic fault model reflects a specific ship, or that its human-review gate is sufficient for any regulated operational context.


The current results come from deliberately simplified, synthetic scenarios. The dataset does not model all time dependencies, sensor drift, maintenance history, or physical failure mechanisms. Rule-based diagnosis is heuristic. The safety gate is a prototype-level escalation rule, **not** a certified safety mechanism.

Future experiments should measure: tool-selection accuracy, grounded-claim accuracy, refusal/escalation behavior, robustness to malicious maintenance notes, and performance on more realistic time-series or appropriately licensed vessel datasets. An ablation study could compare a baseline LLM, a tool-using LLM, and a tool-using LLM with an independent verification layer.

## 10. Author and project purpose

**Mehmet Ateş** — Electrical and Electronics Engineering; M.Sc. candidate in Control and Automation Engineering.

This project was independently developed for **research exploration and a technical portfolio**, bringing together interests in maritime engineering, autonomous systems, machine learning, and trustworthy agentic AI. It is **not** presented as institutionally funded research, a production-ready solution, or a system approved for operational maritime use.
