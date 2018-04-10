from enum import EnumMeta, IntEnum


class DefaultEnumMeta(EnumMeta):
    default = object()

    def __call__(cls, value=default, *args, **kwargs):
        if value is DefaultEnumMeta.default:
            # Assume the first enum is default
            return next(iter(cls))
        # return super().__call__(value, *args, **kwargs)
        return super(DefaultEnumMeta, cls).__call__(value, *args, **kwargs) # PY2


# class MyEnum(IntEnum, metaclass=DefaultEnumMeta):
class MyEnum(IntEnum):
    __metaclass__ = DefaultEnumMeta  # PY2 with enum34
    A = 0
    B = 1
    C = 2

class JacobianEnum(Enum):
    __metaclass__ = DefaultEnumMeta  # PY2 with enum34
    A = 'user-defined'
    B = 'const-positive'
    C = 'const-negative'


assert MyEnum() is MyEnum.A
assert MyEnum(0) is MyEnum.A
assert MyEnum(1) is not MyEnum.A

assert JacobianEnum() is JacobianEnum.A
assert JacobianEnum('user-defined') is JacobianEnum.A
assert JacobianEnum('const-positive') is not JacobianEnum.A
