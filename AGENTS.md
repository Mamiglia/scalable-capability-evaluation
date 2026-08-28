# Execution environment

This checkout is on Morio's laptop. Small jobs may run locally.

- Before starting work likely to use substantial RAM, CPU, disk I/O, or time, explain the expected load and agree with Morio whether to run it locally or on RunPod.
- Treat loading large Parquet datasets or many Inspect `.eval` logs, full-data analyses, model inference/judging, and parallel data-processing agents as heavy work requiring that discussion.
- Do not run several potentially resource-intensive local jobs concurrently.
- Lightweight source inspection, small tests, small scripts, file edits, and scoped Git operations may run locally without discussion.
- When resource use is uncertain, ask first. Do not automatically default small jobs to RunPod.
