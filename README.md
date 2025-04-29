# pFedMARL - Personalized Federated Learning with Multi-Agent Off-Policy Reinforcement Learning

**Project status: pre-release**  
The public codebase is still being refactored and aligned with our internal implementation.
A fully cleaned-up, reproducible version including complete documentation and tests will be pushed no later than the companion paper’s formal publication.
Thank you for your patience while we prepare a robust release.

## Getting Started

1. Open the code inside a VS Code development container.
    - Ctrl+Shift+P -> Remote-Containers: Reopen in Container
2. Create example config.
    - Debugger Tab -> Run and Debug -> Create example config
3. Run the example config.
    - Debugger Tab -> Run and Debug -> Train with example config
4. See results in tensorboard.
    - Open a terminal and run `tensorboard --logdir /workspaces/pFedMARL/.logs/tensorboard/pFedMARL --port 6007 --host 0.0.0.0`

## Attribution

This repository includes files derived from the following projects:

- [DCASE2023 Task 2 Baseline AE](https://github.com/nttcslab/dcase2023_task2_baseline_ae) (MIT License)
- [DCASE2024 Task 2 Evaluator](https://github.com/nttcslab/dcase2024_task2_evaluator) (MIT License)
