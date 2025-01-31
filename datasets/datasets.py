datasets = dict()


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(spec):
    args = spec.get('args')
    if args is None:
        args = dict()
    dataset = datasets[spec['name']](**args)
    return dataset
