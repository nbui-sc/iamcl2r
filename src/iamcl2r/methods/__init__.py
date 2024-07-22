from . import er
from . import hoc

__factory = {
    'er': er.train,
    'clr_er': er.train_clr,
    'hoc': hoc.train,
}


def get_training_method(method_name):
    return __factory[method_name]
