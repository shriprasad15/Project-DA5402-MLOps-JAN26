# AI Declaration

**Name:** S Shriprasad
**Roll No:** DA25E054
**Course:** DA5402 — MLOps

I used AI-assisted tools during the development and documentation of this project, mainly as a support aid for code review, debugging, explanation, and drafting. The core project decisions, implementation choices, testing, validation, and final submission remain my own responsibility.

AI assistance was used in the following limited ways:

- To help review and improve the clarity of Python code, configuration files, and documentation across the multi-service Docker Compose stack.
- To suggest possible fixes for errors encountered while building the Airflow DAG, DVC pipeline, FastAPI backend, MLflow model registry, and Prometheus/Grafana monitoring stack.
- To help structure written material such as the project report, RUNBOOK, HLD/LLD documents, test plan, test report, and user manual.
- To generate alternative wording for technical descriptions, which I then checked and edited to match the actual implementation in the repository.
- To support troubleshooting by explaining error messages, Docker logs, and command outputs encountered during development — for example, NVIDIA container runtime configuration errors, SQLAlchemy version conflicts, and AlertManager template parse errors.
- To provide occasional code completion suggestions for repetitive helper logic, Pydantic schema definitions, Prometheus metric registrations, and small validation checks.

AI was not used as a substitute for understanding the project or for blindly generating the final solution. I reviewed and adapted all AI-assisted suggestions before including them. The multi-task model architecture, data synthesis and preprocessing workflow, training and evaluation pipeline, API behaviour, feedback loop design, deployment setup, and monitoring configuration were checked against the working repository and executed outputs at every stage.

The project implements a Passive-Aggressive Email Detector using a fine-tuned multi-task DistilBERT model with three simultaneous output heads (passive-aggression score, sarcasm score, 5-class tone classification), trained with Uncertainty-Weighted Multi-Task Loss (Kendall & Gal, NeurIPS 2018). It uses a reproducible DVC pipeline for data ingestion and preprocessing, MLflow for experiment tracking and model registry management with a `champion` alias, Apache Airflow for orchestrated retraining, a FastAPI inference backend and Streamlit frontend as loosely coupled Docker services, PostgreSQL as a unified state store, and Prometheus/Grafana/AlertManager for near-real-time observability and alerting. Any AI-assisted text or code changes were validated against these project components to ensure they were consistent with the actual running system.

---

## Example Prompts and AI-Assisted Tasks

The following are representative examples of prompts or comment-based instructions used during development. They are included to show the type of AI assistance used, not as a complete transcript.

| Area | Example prompt or comment | How it was used |
| --- | --- | --- |
| Debugging | "The Airflow train task is failing with 'no GPU found' — what could cause this inside a Docker container?" | Used to identify that `NVIDIA_VISIBLE_DEVICES=all` env var was missing from the scheduler service, which I then added to `docker-compose.yml` and verified with `nvidia-smi`. |
| Documentation | "Improve this HLD section so it clearly explains the five architectural layers and the loose coupling strategy." | Used to refine wording after the architecture had already been designed and implemented; all service names, ports, and interactions were verified against the running stack. |
| Code review | "Review this FastAPI feedback endpoint for validation and error-handling issues." | Used as a checklist; I then added the 404 check for missing prediction IDs and the retraining threshold logic. |
| Error explanation | "AlertManager is not sending emails — the logs show a Go template parse error. What does `U+002D bad char` mean?" | Used to identify that emoji characters in the Subject template caused the error; I removed them from `config.yml` and confirmed email delivery. |
| Testing | "Suggest unit test cases for the multi-task model output shapes and the label mapping logic." | Used to think through missing edge cases; I then wrote and ran the tests myself, fixing the assertions where the expected values did not match the actual implementation. |
| Architecture | "How should I wire the negative feedback threshold to automatically trigger an Airflow DAG retrain?" | Used to understand the pattern; I then implemented `backend/app/services/retrainer.py`, updated `feedback.py`, enabled the Airflow REST API basic auth, and tested the end-to-end trigger. |

I also used code completion suggestions in some places where comments described the intended logic. For example:

```python
# Compute KS two-sample test between rolling input length window and reference distribution
```

This could produce a completion using `scipy.stats.ks_2samp`. I reviewed the suggested parameters, verified the reference distribution loading logic, and tested the output against live metrics in Prometheus.

```python
# Register model to MLflow and set champion alias if val_macro_f1 beats previous best
```

This could assist with the MLflow client calls for `search_model_versions`, `log_model`, and `set_registered_model_alias`. The metric name, comparison logic, and alias name were verified against the actual MLflow model registry.

```python
# Increment feedback_votes_total Prometheus counter and check retrain threshold
```

This could help complete the Prometheus counter increment and the retrainer call. The threshold value, minimum vote count, and DAG run ID format were defined by me and verified against the running backend logs.

In all cases, autocomplete suggestions were treated as draft code. I accepted only suggestions that matched the project design, edited them when needed, and tested or checked them against the running Docker Compose stack.

I understand that I am responsible for the correctness, originality, and academic integrity of the submitted work. AI assistance was used as a productivity and learning aid, while the final judgement, verification, and submission decisions were made by me.
