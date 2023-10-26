## ESSEN: Improving Evolution State Estimation for Temporal Networks using Von Neumann Entropy

### Requirements

There are key required packages:

```python
python==3.7.0
joblib==1.1.0
numpy==1.22.2
pandas==1.4.1
scikit-learn==1.0.2
scipy==1.7.3
torch==1.4.0
sacred==0.8.2
tqdm==4.63.0
```

### Running the code

```python
  $ python ./experiments/experiment_learn_edges.py  -g 0 -b 256 -s ba
```

If you want to change any settings, add a JSON file into  `./update_json/` :

```json
{
    "seed":1,
    "data_name":"bitcoinalpha"
}
```

or change the default settings from  `./default_configuration.py`. The default settings is:

```python
PARAMETER_DICT = {
    'log_name':'ESSEN',
    'data_name':dataset.bitcoinalpha,
    'batch':512,
    'n_epoch':20,
    'n_head':3,
    'drop_out':0.1,
    'attn_mode':'prod',
    'n_layer':2,
    'gpu':'0',
    'lr':0.0001,
    'num_neighbor':60,
    'max_expand_subgraph_size':6000,
    'num_expert':4,
    'seed':1,
}
```



