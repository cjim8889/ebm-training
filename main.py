
from experiments.gmm.train_liouville import main as train_liouville_gmm
from experiments.many_well.train_liouville import main as train_liouville_many_well
from experiments.dw4.train_liouville import main as train_liouville_dw4
from experiments.lj13.train_liouville import main as train_liouville_lj13

def execute_function(method, exp):
    main_fn = eval(f'train_{method}_{exp}')

    return main_fn

if __name__ == '__main__':
    # main_fn = execute_function('liouville', 'gmm')
    # main_fn = execute_function('liouville', 'many_well')
    # main_fn = execute_function('liouville', 'dw4')
    main_fn = execute_function('liouville', 'lj13')
    main_fn()