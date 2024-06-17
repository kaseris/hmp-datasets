ROKOKO_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),
    (3, 4), (4, 5), (5, 6),
    (6, 7), (6, 30), (4, 9),
    (9, 8), (4, 31), (31, 32),
    (0, 53), (53, 54), (54, 55),
    (55, 56), (0, 57), (57, 58),
    (58, 59), (59, 60)
]

ROKOKO_IGNORE_JOINTS = {'left_hand': [33, 50, 51, 52, 46, 47, 48, 49, 42, 43, 44, 45, 38, 39, 40, 41, 34, 35, 36, 37],
                        'right_hand': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]}

__valid_joints = [i for i in range(61) if i not in ROKOKO_IGNORE_JOINTS['left_hand'] + ROKOKO_IGNORE_JOINTS['right_hand']]
ROKOKO_VALID_JOINTS = __valid_joints
