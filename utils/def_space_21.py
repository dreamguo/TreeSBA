# [ direct: [connect_type: [], connect_type: [], ...]
#   direct: [connect_type: [], connect_type: [], ...] ]
# [ "0": ["1": [0,0,1,0], "2": ... "16": [-1, 1, 1, 1]]
#   "1": ["1": [0,0,1,1], "2": ... "16": [-1,-1, 1, 0]] ]
TYPE2COORD = (  {   1: (  0,  0,  1,  0),
                    2: (  0,  1,  1,  0),
                    3: (  0, -1,  1,  0),
                    4: (  1,  0,  1,  1),
                    5: (  0,  0,  1,  1),
                    6: (  1, -1,  1,  1),
                    7: (  0, -1,  1,  1),  },
                {   1: (  0,  0,  1,  1),
                    2: (  1,  0,  1,  1),
                    3: ( -1,  0,  1,  1),
                    4: (  0,  0,  1,  0),
                    5: (  0,  1,  1,  0),
                    6: ( -1,  0,  1,  0),
                    7: ( -1,  1,  1,  0),  })
# TYPE2COORD = (  {   1: (  0,  0,  1,  0),
#                     2: (  0,  1,  1,  0),
#                     3: (  0, -1,  1,  0),
#                     4: (  0.5,  0.5,  1,  1),
#                     5: ( -0.5,  0.5,  1,  1),
#                     6: (  0.5,  -0.5,  1,  1),
#                     7: (  -0.5, -0.5,  1,  1),  },
#                 {   1: (  0,  0,  1,  1),
#                     2: (  1,  0,  1,  1),
#                     3: ( -1,  0,  1,  1),
#                     4: (  0.5, -0.5,  1,  0),
#                     5: (  0.5,  0.5,  1,  0),
#                     6: ( -0.5, -0.5,  1,  0),
#                     7: ( -0.5,  0.5,  1,  0),  })

import ipdb
import torch
from utils import def_tree

MAX_T = def_tree.M_A2T + 1
STUD_N = 1*2

OFFSET = torch.zeros((2, MAX_T, STUD_N, 3)).long()
for brick_i in range(1, MAX_T):
    for dir_i in range(2):
        pos_dir = torch.tensor(TYPE2COORD[dir_i][brick_i])
        OFFSET[dir_i, brick_i] = pos_dir[:3]
        if pos_dir[3] == 0:
            OFFSET[dir_i, brick_i, 0][1] += -1
            OFFSET[dir_i, brick_i, 1][1] += 0
        else:
            OFFSET[dir_i, brick_i, 0][0] += -1
            OFFSET[dir_i, brick_i, 1][0] += 0

OFFSET_DOWN = OFFSET.clone()
OFFSET_DOWN[:, :, :, 2] = -1

# shape (dir, updown, connect_type, 4*STUD_N, 3)
MULTI_OFFSET = torch.zeros((2, 2, def_tree.CONNECT_TYPE, 4*STUD_N, 3)).long()

# CONNECT_TYPE == 0
all_p = [set(), set(), set(), set()]
for brick_i in range(1, MAX_T):
    for i in range(STUD_N):
        all_p[0].add(tuple(OFFSET[0, brick_i, i].tolist()))
        all_p[1].add(tuple(OFFSET[1, brick_i, i].tolist()))
        all_p[2].add(tuple(OFFSET_DOWN[0, brick_i, i].tolist()))
        all_p[3].add(tuple(OFFSET_DOWN[1, brick_i, i].tolist()))
for i, (p1, p2, p3, p4) in enumerate(zip(all_p[0], all_p[1], all_p[2], all_p[3])):
    MULTI_OFFSET[0, 0, 0, i] = torch.tensor(p1)
    MULTI_OFFSET[1, 0, 0, i] = torch.tensor(p2)
    MULTI_OFFSET[0, 1, 0, i] = torch.tensor(p3)
    MULTI_OFFSET[1, 1, 0, i] = torch.tensor(p4)
# CONNECT_TYPE == 1 - 17
for connect_i in range(1, def_tree.CONNECT_TYPE):
    types = def_tree.MULTI_T2A[connect_i]
    if len(types) == 1:
        MULTI_OFFSET[:, 0, connect_i, 0:STUD_N] = OFFSET[:, types[0]]
    else:
        MULTI_OFFSET[:, 0, connect_i, 0:STUD_N] = OFFSET[:, types[0]]
        MULTI_OFFSET[:, 0, connect_i, STUD_N:STUD_N*2] = OFFSET[:, types[1]]

    types = def_tree.MULTI_T2A_DOWN[connect_i]
    if len(types) == 1:
        MULTI_OFFSET[:, 1, connect_i, 0:STUD_N] = OFFSET_DOWN[:, types[0]]
    else:
        MULTI_OFFSET[:, 1, connect_i, 0:STUD_N] = OFFSET_DOWN[:, types[0]]
        MULTI_OFFSET[:, 1, connect_i, STUD_N:STUD_N*2] = OFFSET_DOWN[:, types[1]]
