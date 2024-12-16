# V2X with RSU and DRL

This repository contains code for optimizing the offloading ratio in a V2X (Vehicle-to-Everything) environment using various Reinforcement Learning (RL) algorithms.

## Table of Contents
- [Introduction](#introduction)
- [Environment](#environment)
- [Reinforcement Learning Algorithms](#reinforcement-learning-algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project aims to optimize the offloading ratio in a V2X environment using different RL algorithms. The algorithms implemented include DDPG, SAC, and TD3.

## Environment
The environment file `MECenvirontment_V2X_V3.py` sets up the simulation environment for the V2X scenario. It includes the necessary configurations and parameters for the RL algorithms to interact with.

## Reinforcement Learning Algorithms
The following RL algorithms are implemented in this repository:
- **DDPG**: Deep Deterministic Policy Gradient
- **SAC**: Soft Actor-Critic
- **TD3**: Twin Delayed DDPG

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
To run the RL algorithms, use the following commands:
```bash
# For DDPG
python ddpg.py

# For SAC
python SAC.py

# For TD3
python TD3.py
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.