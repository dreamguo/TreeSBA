import os
import ipdb
import numpy as np
from utils_RAD import buildLEGO

classes = {}
classes['label01'] = '2blocks'
classes['label02'] = '2blocks-perpendicular'
classes['label11'] = 'tower'
classes['label12'] = 'line'
classes['label13'] = 'flat-block'
classes['label14'] = 'wall'
classes['label15'] = 'tall-block'
classes['label16'] = 'pyramid'
classes['label21'] = 'chair'
classes['label22'] = 'couch'
classes['label23'] = 'cup'
classes['label24'] = 'hollow-cylinder'
classes['label25'] = 'table'
classes['label26'] = 'car'
classes['random'] = 'random'

## 1 cm = 25 LDU (LDraw Unit)
LDR_UNITS_PER_STUD = 20  # 1 stud (1x1) (width/length) = 20 LDU
LDR_UNITS_PER_PLATE = 8  # 1 plate (height) = 8 LDU
PLATES_PER_BRICK = 3  # 1 brick (height) = 3 plate = 24 LDU
# 1 stud height   = 4 LDU
# 1 stud diameter = 12 LDU

def npy2ldr(in_filename, out_filename, str_type):
	print(in_filename)
	my_bricks = np.load(in_filename, allow_pickle = True)
	if str_type == '0':
		lego_type = '3001'
	elif str_type == '1':
		lego_type = '3004'
	with open(out_filename, 'w') as file:
		for brick in my_bricks[()].bricks:
			if brick.get_direction() == 1:
				transformation_string = "1 0 0 0 1 0 0 0 1"
			else:
				transformation_string = "0 0 -1 0 1 0 1 0 0"
			coords = brick.get_position()
			x_coord = round(coords[0] * LDR_UNITS_PER_STUD)
			y_coord = round(-coords[2] * LDR_UNITS_PER_PLATE * PLATES_PER_BRICK)
			z_coord = round(coords[1] * LDR_UNITS_PER_STUD)
			file.write('1 4 {} {} {} {} {}.dat\n'.format(x_coord, y_coord, z_coord, 
												transformation_string, lego_type))


def lego2ldr(lego_dir, ldr_dir, str_type):
	for filename in sorted(os.listdir(lego_dir)):
		if filename.endswith(".npy"):
			class_name = filename.split('_')[0]
			sample_num = filename.split('_')[1].split('.')[0]

			if not os.path.exists(os.path.join(ldr_dir, class_name)):
				os.makedirs(os.path.join(ldr_dir, class_name))
			out_file = os.path.join(ldr_dir, class_name,
									class_name + '_' + sample_num + '.ldr')
			npy2ldr(os.path.join(lego_dir, filename), out_file, str_type)
		else:
			continue
