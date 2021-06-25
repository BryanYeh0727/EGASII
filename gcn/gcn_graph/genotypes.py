from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')
Genotype_normal = namedtuple('Genotype_normal', 'normal normal_concat')

PRIMITIVES = [
    'none',
    'skip_connect',
    'conv_1x1',
    'gcnii',
    'sageii',
    'res_sageii',
    'ginii',
]


genotype_ii1 = Genotype(normal=[('ginii', 0), ('ginii', 1), ('res_sageii', 1), ('ginii', 2), ('conv_1x1', 0), ('ginii', 2)], normal_concat=range(1, 5))
genotype_ii2 = Genotype(normal=[('conv_1x1', 0), ('ginii', 1), ('res_sageii', 0), ('ginii', 2), ('ginii', 2), ('ginii', 3)], normal_concat=range(1, 5))
genotype_ii3 = Genotype(normal=[('conv_1x1', 0), ('ginii', 1), ('res_sageii', 0), ('ginii', 2), ('sageii', 1), ('ginii', 2)], normal_concat=range(1, 5))
genotype_ii4 = Genotype(normal=[('conv_1x1', 0), ('conv_1x1', 1), ('conv_1x1', 1), ('ginii', 2), ('sageii', 1), ('ginii', 3)], normal_concat=range(1, 5))
genotype_ii5 = Genotype(normal=[('conv_1x1', 0), ('ginii', 1), ('ginii', 1), ('ginii', 2), ('res_sageii', 1), ('ginii', 2)], normal_concat=range(1, 5))
genotype_ii6 = Genotype(normal=[('sageii', 0), ('conv_1x1', 1), ('ginii', 1), ('ginii', 2), ('conv_1x1', 0), ('ginii', 3)], normal_concat=range(1, 5))
genotype_ii7 = Genotype(normal=[('conv_1x1', 0), ('sageii', 1), ('conv_1x1', 1), ('ginii', 2), ('gcnii', 0), ('ginii', 3)], normal_concat=range(1, 5))
genotype_ii8 = Genotype(normal=[('conv_1x1', 0), ('gcnii', 1), ('res_sageii', 1), ('ginii', 2), ('conv_1x1', 0), ('ginii', 2)], normal_concat=range(1, 5))
genotype_ii9 = Genotype(normal=[('conv_1x1', 0), ('conv_1x1', 1), ('res_sageii', 0), ('ginii', 2), ('ginii', 2), ('ginii', 3)], normal_concat=range(1, 5))
genotype_ii10 = Genotype(normal=[('ginii', 0), ('res_sageii', 1), ('sageii', 1), ('ginii', 2), ('ginii', 2), ('conv_1x1', 3)], normal_concat=range(1, 5))
PPI_Best = genotype_ii2

