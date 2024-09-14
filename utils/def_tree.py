from utils import def_brick_type


if def_brick_type.BRICK_TYPE == '0':
    from utils import def_tree_42
    ACTION2TREE = def_tree_42.ACTION2TREE
    TREE2ACTION = def_tree_42.TREE2ACTION
    M_A2T = def_tree_42.M_A2T
    CONNECT_TYPE = def_tree_42.CONNECT_TYPE
    MULTI_A2T = def_tree_42.MULTI_A2T
    MULTI_T2A = def_tree_42.MULTI_T2A
    MULTI_A2T_DOWN = def_tree_42.MULTI_A2T_DOWN
    MULTI_T2A_DOWN = def_tree_42.MULTI_T2A_DOWN
elif def_brick_type.BRICK_TYPE == '1':
    from utils import def_tree_21
    ACTION2TREE = def_tree_21.ACTION2TREE
    TREE2ACTION = def_tree_21.TREE2ACTION
    M_A2T = def_tree_21.M_A2T
    CONNECT_TYPE = def_tree_21.CONNECT_TYPE
    MULTI_A2T = def_tree_21.MULTI_A2T
    MULTI_T2A = def_tree_21.MULTI_T2A
    MULTI_A2T_DOWN = def_tree_21.MULTI_A2T_DOWN
    MULTI_T2A_DOWN = def_tree_21.MULTI_T2A_DOWN
else:
    assert 0, "Undefined BRICK_TYPE"
