# Multi-Objective Neural Bandits

This repository contains the official implementation of our IJCAI 2025 paper:  
**_Multi-Objective Neural Bandits with Random Scalarization_**

## Overview

We provide code for simulating and solving multi-objective contextual bandit problems using neural-based methods. The key components include:

- **`environments.py`**  
  Contains simulators for multi-objective contextual bandits.  
  To apply the framework to a real-world dataset, implement custom versions of the `_sample_context` and `_eval_expected_reward` methods in a subclass of the base class `moContextMABSimulator`.

- **`agents.py`**  
  Implements multi-objective neural bandit algorithms, including **MONeural-UCB** and **MONeural-TS**.  
  For a quick start, refer to the usage example in **`examples.ipynb`**.

- **`utils.py`**  
  Provides utility functions to support multi-objective optimization tasks.

## Citation

If you find this work useful in your research, please consider citing: