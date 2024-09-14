# based on  Brick(1, 2): x = [0,1], y = [0,2]
ACTION2TREE = { True:  {(0,0):1,
                        (0,1):2, (0,-1):3},
                False: {(0.5, 0.5):4, (-0.5, 0.5):5,
                        (0.5,-0.5):6, (-0.5,-0.5):7,} }
TREE2ACTION = { 1 : (0, (0,0,1)),
                2 : (0, (0,1,1)),  3 : (0, (0,-1,1)), 
                4 : (1, (0.5, 0.5,1)),  5 : (1, (-0.5, 0.5,1)), 
                6 : (1, (0.5,-0.5,1)),  7 : (1, (-0.5,-0.5,1)), }

M_A2T = 7
CONNECT_TYPE = 17
MULTI_A2T = {
    ():0, (1,):1, (2,):2, (3,):3, (4,):4, (5,):5, (6,):6, (7,):7, 
    (2, 3):  M_A2T+1, (2, 6):  M_A2T+2, (2, 7):  M_A2T+3, 
    (3, 4):  M_A2T+4, (3, 5):  M_A2T+5, 
    (4, 6):  M_A2T+6, (4, 7):  M_A2T+7,
    (5, 6):  M_A2T+8, (5, 7):  M_A2T+9,  }
MULTI_T2A = {   0:[], 1:[1], 2:[2], 3:[3], 4:[4], 5:[5], 6:[6], 7:[7], 
                M_A2T+1: (2, 3), M_A2T+2: (2, 6),  M_A2T+3: (2, 7),
                M_A2T+4: (3, 4),  M_A2T+5: (3, 5),
                M_A2T+6: (4, 6), M_A2T+7: (4, 7),
                M_A2T+8: (5, 6), M_A2T+9: (5, 7), }
MULTI_A2T_DOWN = {
    ():0, (1,):1, (2,):2, (3,):3, (4,):4, (5,):5, (6,):6, (7,):7, 
    (2, 3):  M_A2T+1, (3, 5):  M_A2T+2, (3, 4):  M_A2T+3, 
    (2, 7):  M_A2T+4, (2, 6):  M_A2T+5, 
    (5, 7):  M_A2T+6, (4, 7):  M_A2T+7,
    (5, 6):  M_A2T+8, (4, 6):  M_A2T+9,  }
MULTI_T2A_DOWN = {  0:[], 1:[1], 2:[2], 3:[3], 4:[4], 5:[5], 6:[6], 7:[7], 
                    M_A2T+1: (2, 3), M_A2T+2: (3, 5),  M_A2T+3: (3, 4),
                    M_A2T+4: (2, 7),  M_A2T+5: (2, 6),
                    M_A2T+6: (5, 7), M_A2T+7: (4, 7),
                    M_A2T+8: (5, 6), M_A2T+9: (4, 6), }