# methods/registry.py
_MODEL_REG = {}

def register_method(name):
    """装饰器：把类注册到 name"""
    def deco(cls):
        _MODEL_REG[name] = cls
        return cls
    return deco

def build_method(name, **kwargs):
    if name not in _MODEL_REG:
        raise ValueError(f"Unknown method: {name}. Available: {list(_MODEL_REG)}")
    return _MODEL_REG[name](**kwargs)

def list_methods():
    return sorted(list(_MODEL_REG.keys()))
