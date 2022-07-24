import inspect


def get_kw(cls, kwargs):
    '''保留排除cls的入参后的kwargs
    '''
    kwargs_new = {}
    for k in kwargs:
        if k not in set(inspect.getargspec(cls)[0]):
            kwargs_new[k] = kwargs[k]
    return kwargs_new