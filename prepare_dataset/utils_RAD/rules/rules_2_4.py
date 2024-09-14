import ipdb
import numpy as np


PROBS_CONTACTS = np.array([4.0, 9.0, 2.0, 1.0])
PROBS_CONTACTS /= np.sum(PROBS_CONTACTS)

RULE_CONTACTS = [
    # [0, 2], [1, 0] -> 4
    {
        'num_contacts': 4,
        'translations': [[0, 2], [0, -2], [1, 0], [-1, 0]],
        'direction': 0
    },
    # [1, 1], [1, 0], [0, 1], [0, 0] -> 9
    {
        'num_contacts': 4,
        'translations': [[1, 1], [1, -1], [-1, 1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]],
        'direction': 1
    },
    # [0, 1] -> 2
    {
        'num_contacts': 6,
        'translations': [[0, 1], [0, -1]],
        'direction': 0
    },
    # [0, 0] -> 1
    {
        'num_contacts': 8,
        'translations': [[0, 0]],
        'direction': 0
    }
]

LIST_RULES = []
ind = 1
for rule in RULE_CONTACTS:
    cur_direction = rule['direction']
    cur_num_contacts = rule['num_contacts']

    for translation in rule['translations']:
        cur_rule = [ind, [cur_direction, translation, cur_num_contacts]]
        LIST_RULES.append(cur_rule)

        ind += 1

SIZE_UPPER = SIZE_LOWER = [2, 4]
