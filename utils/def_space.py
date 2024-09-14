from utils import def_brick_type


if def_brick_type.BRICK_TYPE == '0':
    from utils import def_space_42
    TYPE2COORD = def_space_42.TYPE2COORD
    MAX_T = def_space_42.MAX_T
    STUD_N = def_space_42.STUD_N
    MULTI_OFFSET = def_space_42.MULTI_OFFSET
elif def_brick_type.BRICK_TYPE == '1':
    from utils import def_space_21
    TYPE2COORD = def_space_21.TYPE2COORD
    MAX_T = def_space_21.MAX_T
    STUD_N = def_space_21.STUD_N
    MULTI_OFFSET = def_space_21.MULTI_OFFSET
else:
    assert 0, "Undefined BRICK_TYPE"
