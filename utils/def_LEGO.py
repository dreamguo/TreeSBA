import ipdb
import torch
from utils import def_tree, def_space, def_brick_type


class LEGO():
    def __init__(self, size, device, brick_num):
        # root_dir default 0: Brick(2, 4) / Brick(1, 2)
        self.coords = []
        self.device = device
        self.str_type = def_brick_type.BRICK_TYPE
        self.brick_num = brick_num
        self.valid_idx = torch.arange(0, size, device=device)
        self.root_coord = torch.tensor([size//2, size//2, size//2, 0], device=device)
        self.voxel = torch.zeros((size, size, size), dtype=torch.int64, device=device)
        if self.str_type == '0':
            assert self.try_add_coord_2_4(self.root_coord, strict=0)
        elif self.str_type == '1':
            assert self.try_add_coord_1_2(self.root_coord, strict=0)
    
    def try_add_coord_2_4(self, coord, strict):
        x, y, z, direct = coord
        x_left = (x - 1) - direct
        x_right = (x + 1) + direct
        y_left = (y - 2) + direct
        y_right = (y + 2) - direct
        z_left = z
        z_right = z + 1

        if (x_left not in self.valid_idx or x_right not in self.valid_idx or
            y_left not in self.valid_idx or y_right not in self.valid_idx or
            z_left not in self.valid_idx or z_right not in self.valid_idx ):
            return 0
        if (self.voxel[x_left:x_right, y_left:y_right, z_left:z_right] == 1).any():
            aa = (self.voxel[x_left:x_right, y_left:y_right, z_left:z_right] == 1).sum()
            if strict:
                return 0
            if aa == 8:
                return 0
        self.voxel[x_left:x_right, y_left:y_right, z_left:z_right] = 1
        self.coords.append(coord)
        return 1

    def try_add_coord_1_2(self, coord, strict):
        x, y, z, direct = coord
        x_left = x - direct
        x_right = (x + 1)
        y_left = (y - 1)  + direct
        y_right = (y + 1)
        z_left = z
        z_right = z + 1

        if (x_left not in self.valid_idx or x_right not in self.valid_idx or
            y_left not in self.valid_idx or y_right not in self.valid_idx or
            z_left not in self.valid_idx or z_right not in self.valid_idx ):
            return 0
        if (self.voxel[x_left:x_right, y_left:y_right, z_left:z_right] == 1).any():
            aa = (self.voxel[x_left:x_right, y_left:y_right, z_left:z_right] == 1).sum()
            if strict:
                return 0
            if aa == 2:
                return 0
        self.voxel[x_left:x_right, y_left:y_right, z_left:z_right] = 1
        self.coords.append(coord)
        return 1

    def try_add_coord_1_1(self, coord):
        x, y, z, direct = coord
        x_left = x
        x_right = x + 1
        y_left = y 
        y_right = y + 1
        z_left = z
        z_right = z + 1

        if (x_left not in self.valid_idx or x_right not in self.valid_idx or
            y_left not in self.valid_idx or y_right not in self.valid_idx or
            z_left not in self.valid_idx or z_right not in self.valid_idx ):
            return 0
        if (self.voxel[x_left:x_right, y_left:y_right, z_left:z_right] == 1).any():
            return 0
        self.voxel[x_left:x_right, y_left:y_right, z_left:z_right] = 1
        self.coords.append(coord)
        return 1

    def add_brick_fast_with_check(self, node_idx, connect_type, up, strict=1):
        if len(self.coords) >= self.brick_num or node_idx >= len(self.coords):
            return 0
        prev_coord = self.coords[node_idx]
        coord = torch.tensor(def_space.TYPE2COORD[prev_coord[3]][connect_type], device=self.device)
        coord[2] = up
        coord[:3] += prev_coord[:3]
        if self.str_type == '0':
            vaild = self.try_add_coord_2_4(coord, strict)
        elif self.str_type == '1':
            vaild = self.try_add_coord_1_2(coord, strict)
        else:
            assert 0, "invalid str_type"
        return vaild


class LEGO_Voxel():
    def __init__(self, size, device, brick_num, root):
        # root_dir default 0: Brick(2, 4) / Brick(1, 2)
        self.root_idx = root
        self.device = device
        self.brick_num = brick_num
        self.coords = [[]] * (self.brick_num - 1)
        self.str_type = def_brick_type.BRICK_TYPE
        self.valid_idx = torch.arange(0, size, device=device)
        self.root_coord = [size//2, size//2, size//2, 0]
        self.voxel = torch.zeros((size, size, size), dtype=torch.int64, device=device)
        assert self.str_type == '0'
        assert self.try_add_coord(self.root_coord, self.root_idx)

    def try_add_coord(self, coord, idx):
        x, y, z, direct = coord
        x_left = (x - 1) - direct
        x_right = (x + 1) + direct
        y_left = (y - 2) + direct
        y_right = (y + 2) - direct
        z_left = z
        z_right = z + 1

        if (self.voxel[x_left:x_right, y_left:y_right, z_left:z_right] == 1).any():
            return 0
        self.voxel[x_left:x_right, y_left:y_right, z_left:z_right] = 1
        self.coords[idx] = coord
        return 1

    def add_brick(self, node_idx, neighbor_idx, rotation, edge_embedding, growup):
        if len(self.coords) >= self.brick_num or node_idx >= len(self.coords):
            return 0
        prev_coord = self.coords[node_idx]
        coord = [0, 0, 0, int(prev_coord[3] ^ rotation)]
        if growup == 1:
            coord[2] = 1
        elif growup == 0:
            coord[2] = -1
        if prev_coord[3] == 0:
            coord[0] = int(edge_embedding[0])
            coord[1] = int(edge_embedding[1])
        elif prev_coord[3] == 1:
            coord[0] = int(edge_embedding[1])
            coord[1] = -int(edge_embedding[0])
        coord[0] += prev_coord[0]
        coord[1] += prev_coord[1]
        coord[2] += prev_coord[2]
        return self.try_add_coord(coord, neighbor_idx)
