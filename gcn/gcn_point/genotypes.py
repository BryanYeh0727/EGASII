from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')
Genotype_normal = namedtuple('Genotype_normal', 'normal normal_concat')

PRIMITIVES = [
    'none',
    'skip_connect',
    'conv_1x1',
    'gcnii',
    'ginii',
    'sageii',
    'rsageii',
    'edgeii',
]


genotype_ii1 = Genotype(normal=[('edgeii', 0), ('ginii', 1), ('ginii', 0), ('edgeii', 2), ('edgeii', 1), ('ginii', 3)], normal_concat=range(1, 5))
genotype_ii2 = Genotype(normal=[('conv_1x1', 0), ('sageii', 1), ('rsageii', 0), ('sageii', 1), ('rsageii', 2), ('edgeii', 3)], normal_concat=range(1, 5))
genotype_ii3 = Genotype(normal=[('sageii', 0), ('edgeii', 1), ('ginii', 1), ('rsageii', 2), ('edgeii', 1), ('gcnii', 2)], normal_concat=range(1, 5))
genotype_ii4 = Genotype(normal=[('sageii', 0), ('conv_1x1', 1), ('rsageii', 1), ('ginii', 2), ('skip_connect', 0), ('edgeii', 1)], normal_concat=range(1, 5))
genotype_ii5 = Genotype(normal=[('ginii', 0), ('skip_connect', 1), ('skip_connect', 0), ('conv_1x1', 2), ('skip_connect', 0), ('ginii', 3)], normal_concat=range(1, 5))
genotype_ii6 = Genotype(normal=[('edgeii', 0), ('sageii', 1), ('ginii', 1), ('conv_1x1', 2), ('edgeii', 2), ('edgeii', 3)], normal_concat=range(1, 5))
ModelNet_Best = genotype_ii1

