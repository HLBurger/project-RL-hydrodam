# READ ME
This repository contains a reinforcement learning (RL) project with multiple algorithms (e.g., Q-Learning, DQN), a shared environment, reward shaping experiments, and supporting datasets. 
The codebase is organized to clearly separate core source code, experiment notebooks, and data.

This README explains the proposed repository structure, the role of each component, how to install, and how to run the code and experiments.

## Repository Structure
├── README.md <br>
├── requirements.txt <br>
├── data/ <br>
│ ├── train.xlsx <br>
│ └── validate.xlsx <br>
│ <br>
├── src/ <br>
│ ├── env/ <br>
│ │ ├── test_env.py # Base environment (formerly TestEnv.py) <br>
│ │ └── reward_shaping.py # Reward shaping variants <br>
│ │ <br>
│ ├── models/ <br>
│ │ ├── Qlearning.py # Q-Learning implementation <br>
│ │ └── DQN.py # Deep Q-Network implementation <br>
│ │ <br>
│ ├── policies/ <br>
│ │ └── policies.py # Baseline and heuristic policies <br>
│ │ <br>
│ ├── utils/ <br>
│ │ └── analytics.py # Logging, metrics, plotting helpers <br>
│ │ <br>
│ └── main.py # Entry point for training / evaluation <br>
│ <br>
├── notebooks/ <br>
│ ├── qlearning.ipynb # Q-Learning experiments <br>
│ ├── dqn.ipynb # DQN experiments <br>
│ ├── model_tuning.ipynb # Hyperparameter tuning <br>
│ ├── analytics.ipynb # Result analysis and visualization <br>
│ └── policies.ipynb # Policy baseline experiments <br>

## Overview
**Environment**
- `TestEnv` is the base environment used by all models.
- `reward_shaping` changes the base reward from `TestEnv` and adjusts it for training purposes. Reward shaping is not used for the testing phase.

**Models**
- Model implementations are located in `src/models/`.
- Model experiments are found in `notebooks/`.

**Baseline policies**
- Baseline policy implementations are located in `policies/policies.py`.
- Baseline policy experiments are found in `notebooks/policies.ipynb`.

**Data**
- `train.xlsx` and `validate.xlsx` are used for training and validating model performance on the environment, and are located in `data/`.

## How to Run the Code?
**1. Create and activate a virtual environment:**
```
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate   	 # Windows
pip install -r requirements.txt
```
**2. running the code:** <br>
You can directly run the code using the following code below, depending on the specific model you want to use.
```
python src/main.py --model qlearning
python src/main.py --model hour_policy
```

**3. running experiments:** <br>
Alternatively, you can run the experiments of each model in the python notebooks, which is recommended.
Just open one of the following notebooks in the `notebooks/` directory:
- `Qlearning.ipynb`
- `DQN.ipynb`
- `policies.ipynb`

