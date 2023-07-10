
from .simple_comm import SimpleComm
from .attention_comm import AttentionComm

registration = { 'mlp': SimpleComm,
                'attention':AttentionComm}

def create_comm_protocol(comm_method, input_shape, device,  args):
    return registration[comm_method](input_shape, device, args)
