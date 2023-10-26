import os 


class Dataset():
    def __init__(self):    
        self.wikipedia = 'wikipedia'
        self.math = 'mathoverflow-a2q'
        self.bitcoinalpha = 'bitcoinalpha'
        self.bitcoinotc = 'bitcoinotc'

PATH = os.path.join(os.getcwd())


dataset = Dataset()

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
