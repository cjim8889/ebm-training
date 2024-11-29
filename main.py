
from experiments.gmm.train_liouville import main as train_liouville_gmm

def execute_function(method, exp):
    main_fn = eval(f'train_{method}_{exp}')

    return main_fn

if __name__ == '__main__':
    main_fn = execute_function('liouville', 'gmm')
    main_fn()