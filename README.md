# FedEmbryo

This is the official repository for the paper of 'Federated Task-Adaptive Learning for Personalized Selection of Human IVF-derived Embryos'.

![FedEmbryo](figure/architecture.jpg)

Here,  we have developed a distributed DL system, termed ‘FedEmbryo’, tailored explicitly for personalized embryo selection while preserving data privacy. Within FedEmbryo, we introduce a Federated Task-Adaptive Learning (FTAL) approach with a hierarchical dynamic weighting adaption (HDWA) mechanism. This approach first uniquely integrates multi-task learning (MTL) with federated learning (FL) by proposing a unified multitask client architecture that consists of shared layers and task-specific layers to accommodate the single- and multi-task learning within each client. Furthermore, the HDWA mechanism mitigates the skewed model performance attributed to data heterogeneity from FTAL. It considers the learning feedback (loss ratios) from the tasks and clients, facilitating a dynamic balance to task attention and client aggregation. Finally, we refine FedEmbryo to address critical clinical scenarios in the IVF processes, including morphology evaluation and live-birth outcomes. We operate each morphological metric as an individual task within the client's model to perform FTAL in morphology evaluation and incorporate embryo images with corresponding clinical factors as multimodal inputs to predict live-birth outcomes.  Experimental results indicate that FedEmbryo outperforms both locally trained models and state-of-the-art (SOTA) FL methods regarding morphology evaluation and live-birth outcome prediction. Our research marks a significant advancement in the development of AI in IVF treatments.

## Setup

```bash
conda create -n fedembryo python=3.9.7
conda activate fedembryo
pip install -r requirements.txt
```

## Structure

- `figure`: method illustration and visualization.
- `model`: architecture of FedEmbryo.
- `Utils`: utility functions 
- `output`: results.
- `mains`: code for training and validation.

## Usage

### Train
train FedEmbryo:

```
python federated_adaption_avg_main.py
```
train FedProx:

````
python federated_prox_main.py
````

train FedAvg:

````
python federated_avg_main.py
````

train model locally:

````
python federated_local_main.py
````

centralized training:

````
python main.py
````
