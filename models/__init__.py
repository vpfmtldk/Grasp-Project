def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'ggcnn':
        from .ggcnn import GGCNN
        return GGCNN
    elif network_name == 'ggcnn2':
        from .ggcnn2 import GGCNN2
        return GGCNN2
    elif network_name in ('grconvnet', 'grconv', 'gr'):
        from .grconvnet import GRConvNet
        return GRConvNet
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
