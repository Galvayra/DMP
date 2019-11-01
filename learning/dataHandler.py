from .variables import GRAY_SCALE, INITIAL_IMAGE_SIZE
from PIL import Image
import numpy as np
import json
import math
import sys
from os import path
from DMP.modeling.variables import EXTENSION_OF_IMAGE, KEY_TF_NAME, KEY_IMG_TRAIN, KEY_IMG_VALID, KEY_IMG_TEST, \
    TF_RECORD_PATH, IMAGE_PICKLES_PATH

current_script = sys.argv[0].split('/')[-1]

if current_script == "training.py":
    from DMP.utils.arg_training import READ_VECTOR, DO_SHOW, TYPE_OF_FEATURE, COLUMN_TARGET, IMAGE_PATH, VERSION, \
        show_options
elif current_script == "predict.py":
    from DMP.utils.arg_predict import READ_VECTOR, DO_SHOW, TYPE_OF_FEATURE, COLUMN_TARGET, IMAGE_PATH, VERSION, \
        show_options
elif current_script == "extract_feature.py" or current_script == "print_feature.py":
    from DMP.utils.arg_extract_feature import *
    from DMP.learning.plot import MyPlot
    from collections import OrderedDict
    from sklearn.ensemble import RandomForestClassifier
elif current_script == "convert_images.py":
    from DMP.utils.arg_convert_images import *
elif current_script == "fine_tuning.py":
    from DMP.utils.arg_fine_tuning import READ_VECTOR, DO_SHOW, VERSION, TYPE_OF_FEATURE, COLUMN_TARGET, show_options
    from DMP.learning.variables import IMAGE_RESIZE, DO_NORMALIZE
elif current_script == "predict_tfRecord.py":
    from DMP.utils.arg_predict_tfRecord import READ_VECTOR, DO_SHOW, TYPE_OF_FEATURE, COLUMN_TARGET, IMAGE_PATH, \
        show_options

alivePath = 'alive/'
deathPath = 'death/'


class DataHandler:
    def __init__(self):
        try:
            with open(READ_VECTOR, 'r') as file:
                vector_list = json.load(file)
        except FileNotFoundError:
            print("\nPlease execute encoding script !")
            print("FileNotFoundError] READ_VECTOR is", "'" + READ_VECTOR + "'", "\n\n")
        else:
            print("\nRead vectors -", READ_VECTOR)

            show_options()

            # {
            #   feature: { 0: ["D", "header_name"], ... , n(dimensionality): ["CZ", "header_name"] }
            #   x_train: [ vector 1, ... vector n ], ... x_test, x_valid , ... , y_valid
            # }
            if current_script == "extract_feature.py":
                self.vector_matrix = OrderedDict()
                self.vector_matrix = {
                    "feature": dict(),
                    "x_train": dict(),
                    "y_train": list(),
                    "x_valid": dict(),
                    "y_valid": list(),
                    "x_test": dict(),
                    "y_test": list()
                }
                self.__importance = dict()

            self.feature = vector_list["feature"]
            self.x_train = vector_list["x_train"][TYPE_OF_FEATURE]
            self.y_train = vector_list["y_train"]
            self.x_valid = vector_list["x_valid"][TYPE_OF_FEATURE]
            self.y_valid = vector_list["y_valid"]
            self.x_test = vector_list["x_test"][TYPE_OF_FEATURE]
            self.y_test = vector_list["y_test"]

            if KEY_IMG_TRAIN in vector_list:
                self.img_train = vector_list[KEY_IMG_TRAIN]
            if KEY_IMG_VALID in vector_list:
                self.img_valid = vector_list[KEY_IMG_VALID]
            if KEY_IMG_TEST in vector_list:
                self.img_test = vector_list[KEY_IMG_TEST]
            modeling_path = path.dirname(path.dirname(path.abspath(__file__))) + "/modeling/"
            self.tf_record_path = modeling_path + TF_RECORD_PATH + READ_VECTOR.split('/')[-1] + "/"
            self.img_pickles_path = modeling_path + IMAGE_PICKLES_PATH + READ_VECTOR.split('/')[-1] + "/"

            # TODO: erase if
            if KEY_TF_NAME in vector_list:
                self.tf_name_vector = vector_list[KEY_TF_NAME]
            else:
                self.tf_name_vector = None

            # if current_script == "fine_tuning.py":
            #     if TYPE_OF_MODEL == "tuning":
            #         self.img_train = vector_list[KEY_IMG_TRAIN]
            #         self.img_valid = vector_list[KEY_IMG_VALID]
            #         self.img_test = vector_list[KEY_IMG_TEST]

            # count list
            # index == 0, training // index == 1, valid // index == 2, test
            self.count_all = list()
            self.count_mortality = list()
            self.count_alive = list()
            self.__set_count()
            # self.tf_record_path = path.dirname(path.dirname(path.abspath(__file__))) + "/" + MODELING_PATH + \
            #                       TF_RECORD_PATH + READ_VECTOR.split('/')[-1] + "/"

    @property
    def importance(self):
        return self.__importance

    @staticmethod
    def __hard_coding_for_school_paper(name_of_set):
        # sepsis
        if COLUMN_TARGET == "CU":
            if name_of_set == "train":
                target_index = [1, 19, 31, 43, 50, 56, 67, 73, 86, 90, 99, 104, 108, 109, 112, 116, 134, 135, 138, 142, 143, 145, 146, 150, 161, 164, 170, 171, 172, 173, 175, 178, 179, 180, 187, 206, 211, 222, 233, 239, 242, 246, 251, 253, 271, 277, 284, 289, 290, 295, 297, 298, 305, 309, 314, 315, 316, 319, 320, 324, 325, 331, 334, 341, 343, 346, 350, 351, 356, 359, 360, 363, 373, 376, 384, 393, 399, 400, 418, 423, 427, 429, 431, 449, 450, 460, 462, 465, 480, 490, 492, 499, 503, 504, 510, 511, 525, 528, 537, 542, 549, 557, 559, 560, 562, 576, 577, 587, 589, 594, 598, 600, 606, 608, 609, 612, 613, 614, 637, 638, 639, 640, 643, 644, 646, 649, 650, 651, 655, 658, 659, 660, 666, 677, 679, 689, 694, 695, 696, 700, 712, 714, 717, 718, 729, 730, 732, 736, 741, 744, 747, 751, 754, 758, 760, 769, 780, 781, 785, 787, 795, 805, 809, 813, 825, 829, 835, 836, 837, 841, 842, 843, 848, 850, 858, 860, 863, 864, 866, 876, 878, 879, 880, 888, 890, 894, 903, 906, 918, 919, 921, 922, 926, 927, 941, 942, 943, 955, 956, 959, 964, 966, 973, 974, 975, 986, 988, 989, 990, 1002, 1009, 1016, 1018, 1021, 1024, 1026, 1027, 1028, 1029, 1034, 1036, 1042, 1049, 1051, 1053, 1069, 1070, 1071, 1072, 1074, 1075, 1080, 1085, 1086, 1090, 1104, 1106, 1113, 1123, 1129, 1132, 1134, 1138, 1144, 1146, 1163, 1170, 1172, 1174, 1176, 1179, 1181, 1194, 1196, 1197, 1201, 1205, 1218, 1221, 1223, 1233, 1242, 1245, 1249, 1260, 1269, 1272, 1280, 1285, 1287, 1294, 1311, 1313, 1316, 1321, 1323, 1325, 1331, 1336, 1339, 1344, 1346, 1349, 1354, 1358, 1363, 1368, 1369, 1373, 1379, 1381, 1389, 1401, 1407, 1411, 1414, 1424, 1428, 1432, 1434, 1441, 1443, 1446, 1449, 1450, 1451, 1452, 1467, 1472, 1473, 1475, 1486, 1490, 1499, 1501, 1503, 1505, 1506, 1508, 1510, 1512, 1514, 1520, 1525, 1539, 1545, 1555, 1558, 1560, 1561, 1568, 1575, 1576, 1588, 1591, 1595, 1597, 1599, 1602, 1610, 1613, 1614, 1615, 1621, 1629, 1630, 1631, 1632, 1635, 1637, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1647, 1648, 1649, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1666, 1667, 1677, 1678, 1686, 1691, 1692, 1693, 1702, 1709, 1711, 1713, 1715, 1719, 1720, 1727, 1730, 1734, 1741, 1757, 1758, 1770, 1771, 1777, 1796, 1798, 1816, 1822, 1831, 1833, 1842, 1846, 1847, 1860, 1862, 1898, 1902, 1930, 1932, 1945, 1954, 1961, 1989, 2002, 2007, 2022, 2028, 2029, 2049, 2053, 2067, 2094, 2109, 2212, 2232, 2244, 2254, 2266, 2293, 2303, 2382, 2398, 2405, 2443, 2450, 2472, 2484, 2492, 2493, 2519, 2521, 2528, 2530, 2531, 2532, 2535, 2537, 2541, 2542, 2543, 2545, 2546, 2547, 2550, 2551, 2552, 2553, 2555, 2563, 2564, 2566, 2567, 2569, 2573, 2577, 2580, 2582, 2587, 2588, 2591, 2592, 2593, 2595, 2600, 2601, 2604, 2605, 2607, 2610, 2612, 2613, 2614, 2615, 2616, 2620, 2621, 2622, 2624, 2627, 2629, 2630, 2633, 2634, 2635, 2637, 2638, 2640, 2641, 2642, 2643, 2647, 2649, 2651, 2653, 2654, 2656, 2657, 2659, 2661, 2663, 2668, 2669, 2672, 2673, 2679, 2680, 2685, 2687, 2688, 2690, 2691, 2694, 2695, 2696, 2697, 2698, 2701, 2702, 2703, 2708, 2716, 2717, 2719, 2723, 2728, 2735, 2736, 2739, 2742, 2743, 2744, 2745, 2746, 2748, 2753, 2754, 2757, 2762, 2763, 2766, 2774, 2780, 2790, 2791, 2792, 2795, 2796, 2798, 2800, 2810, 2811, 2815, 2817, 2819, 2827, 2832, 2838, 2839, 2846, 2849, 2852, 2853, 2856, 2859, 2860, 2861, 2862]

            elif name_of_set == "valid":
                target_index = [1, 4, 11, 20, 22, 23, 24, 25, 42, 46, 52, 57, 61, 64, 67, 68, 69, 81, 84, 89, 91, 95, 97, 100, 101, 102, 104, 106, 107, 108, 109, 120, 121, 129, 130, 133, 139, 143, 144, 149, 151, 165, 168, 171, 177, 185, 192, 204, 205, 210, 217, 221, 223, 224, 225, 226, 227, 246, 257, 266, 271, 284, 292, 318, 336, 338, 346, 347, 350, 351, 354, 360, 361, 364, 365, 366, 367]

            else:
                target_index = [0, 2, 13, 27, 36, 64, 71, 72, 73, 75, 76, 79, 82, 83, 84, 89, 90, 91, 96, 97, 98, 102, 104, 124, 127, 129, 131, 132, 133, 151, 152, 156, 166, 169, 170, 172, 178, 189, 197, 204, 209, 214, 215, 216, 217, 218, 219, 237, 239, 278, 286, 290, 295, 326, 330, 333, 341, 343, 344, 345, 347, 348, 351, 353, 360, 361, 362, 366, 369, 370]

        # pneumonia
        elif COLUMN_TARGET == "CS":
            if name_of_set == "train":
                target_index = [19, 42, 43, 56, 57, 67, 104, 130, 134, 152, 164, 170, 171, 173, 175, 178, 180, 195, 225, 229, 233, 242, 253, 265, 271, 280, 281, 282, 284, 285, 287, 292, 297, 298, 307, 309, 314, 320, 340, 346, 356, 359, 401, 415, 418, 427, 432, 449, 454, 483, 492, 507, 510, 538, 560, 574, 589, 600, 606, 610, 650, 655, 658, 660, 667, 669, 677, 695, 710, 730, 732, 736, 754, 759, 766, 791, 813, 837, 841, 842, 843, 866, 888, 890, 903, 921, 922, 934, 936, 938, 941, 955, 964, 966, 973, 975, 976, 989, 998, 1010, 1011, 1029, 1036, 1053, 1056, 1060, 1074, 1080, 1111, 1115, 1129, 1132, 1146, 1150, 1169, 1170, 1173, 1174, 1176, 1185, 1188, 1205, 1209, 1234, 1245, 1259, 1299, 1321, 1323, 1338, 1346, 1354, 1363, 1368, 1373, 1383, 1422, 1424, 1428, 1443, 1446, 1452, 1460, 1473, 1503, 1545, 1586, 1591, 1596, 1597, 1604, 1612, 1621, 1623, 1633, 1635, 1637, 1638, 1643, 1646, 1647, 1648, 1654, 1660, 1666, 1677, 1681, 1683, 1687, 1695, 1700, 1704, 1708, 1712, 1721, 1745, 1746, 1754, 1758, 1762, 1764, 1765, 1766, 1769, 1779, 1780, 1788, 1793, 1795, 1801, 1809, 1810, 1814, 1818, 1828, 1830, 1833, 1836, 1866, 1867, 1876, 1878, 1882, 1884, 1885, 1886, 1887, 1889, 1891, 1896, 1902, 1904, 1905, 1908, 1922, 1925, 1942, 1950, 1951, 1956, 1959, 1969, 1974, 1983, 1985, 2000, 2009, 2027, 2031, 2032, 2040, 2043, 2045, 2049, 2052, 2053, 2060, 2061, 2067, 2079, 2080, 2087, 2089, 2090, 2095, 2098, 2101, 2104, 2112, 2117, 2122, 2124, 2127, 2135, 2136, 2142, 2156, 2158, 2159, 2164, 2166, 2169, 2185, 2199, 2212, 2218, 2227, 2230, 2237, 2240, 2245, 2259, 2275, 2312, 2315, 2319, 2324, 2342, 2349, 2350, 2355, 2358, 2364, 2365, 2371, 2377, 2383, 2391, 2405, 2407, 2408, 2415, 2419, 2422, 2427, 2429, 2430, 2439, 2442, 2452, 2466, 2474, 2475, 2479, 2482, 2485, 2487, 2493, 2503, 2506, 2508, 2514, 2523, 2525, 2537, 2539, 2543, 2545, 2547, 2550, 2552, 2554, 2556, 2557, 2562, 2564, 2565, 2568, 2570, 2573, 2580, 2582, 2591, 2603, 2622, 2624, 2626, 2630, 2631, 2632, 2633, 2639, 2641, 2642, 2643, 2644, 2645, 2646, 2647, 2648, 2650, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2667, 2668, 2672, 2675, 2677, 2678, 2679, 2682, 2684, 2687, 2703, 2704, 2705, 2706, 2707, 2715, 2716, 2720, 2721, 2722, 2734, 2735, 2736, 2742, 2746, 2754, 2755, 2756, 2757, 2760, 2764, 2765, 2766, 2769, 2771, 2772, 2774, 2777, 2778, 2780, 2785, 2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795, 2796, 2799, 2800, 2801, 2802, 2804, 2806, 2807, 2808, 2809, 2811, 2812, 2813, 2815, 2818, 2819, 2821, 2822, 2824, 2825, 2828, 2829, 2831, 2835, 2836, 2837, 2838, 2839, 2840, 2847, 2848, 2849, 2850, 2851, 2852, 2855, 2862, 2863]

            elif name_of_set == "valid":
                target_index = [4, 33, 42, 57, 67, 95, 97, 98, 102, 120, 121, 125, 151, 171, 180, 192, 208, 210, 218, 220, 221, 227, 230, 235, 237, 241, 242, 245, 252, 255, 257, 263, 266, 268, 269, 271, 290, 307, 309, 314, 318, 327, 342, 348, 349, 350, 351, 353, 355, 359, 362, 363, 364, 366, 367, 368, 369, 370]

            else:
                target_index = [2, 5, 42, 64, 71, 72, 96, 101, 109, 114, 133, 187, 197, 220, 228, 232, 233, 235, 255, 258, 264, 265, 270, 281, 283, 286, 298, 306, 311, 312, 321, 333, 339, 342, 345, 346, 349, 350, 351, 352, 353, 354, 355, 356, 357, 359, 360, 366, 370, 372, 373, 374]

        # bacteremia
        else:
            if name_of_set == "train":
                target_index = [2, 18, 42, 50, 99, 136, 143, 194, 253, 271, 297, 305, 314, 334, 363, 369, 423, 460, 463, 465, 503, 511, 515, 528, 551, 552, 576, 582, 594, 612, 614, 651, 657, 660, 667, 681, 683, 712, 715, 731, 741, 744, 774, 781, 793, 827, 860, 862, 866, 871, 872, 878, 903, 953, 963, 964, 974, 988, 1011, 1018, 1022, 1024, 1026, 1029, 1031, 1049, 1057, 1072, 1074, 1075, 1104, 1136, 1154, 1172, 1177, 1182, 1183, 1196, 1217, 1223, 1242, 1249, 1269, 1284, 1288, 1291, 1293, 1311, 1325, 1335, 1373, 1390, 1450, 1475, 1486, 1508, 1520, 1586, 1588, 1602, 1610, 1613, 1614, 1615, 1629, 1630, 1631, 1635, 1639, 1643, 1645, 1651, 1656, 1657, 1666, 1677, 1678, 1686, 1692, 1693, 1702, 1709, 1711, 1713, 1715, 1719, 1720, 1727, 1734, 1741, 1757, 1758, 1770, 1771, 1777, 1796, 1797, 1798, 1816, 1822, 1831, 1846, 1847, 1850, 1881, 1890, 1894, 1901, 1926, 1927, 1930, 1936, 1942, 1958, 1961, 1962, 1964, 1987, 1993, 2002, 2006, 2012, 2028, 2029, 2033, 2049, 2077, 2080, 2100, 2108, 2128, 2202, 2234, 2236, 2271, 2313, 2330, 2344, 2361, 2375, 2383, 2393, 2396, 2398, 2399, 2402, 2434, 2435, 2436, 2437, 2462, 2472, 2487, 2492, 2495, 2496, 2505, 2507, 2509, 2519, 2520, 2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2534, 2535, 2536, 2538, 2539, 2540, 2541, 2542, 2543, 2544, 2549, 2550, 2552, 2554, 2555, 2557, 2558, 2559, 2560, 2563, 2566, 2567, 2569, 2570, 2571, 2572, 2574, 2575, 2576, 2577, 2578, 2579, 2581, 2582, 2583, 2584, 2585, 2587, 2588, 2589, 2590, 2592, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2608, 2609, 2610, 2611, 2613, 2614, 2615, 2616, 2617, 2618, 2620, 2621, 2623, 2625, 2643, 2683, 2694, 2700, 2717, 2726, 2729, 2735, 2741, 2756, 2761, 2762, 2832, 2854]

            elif name_of_set == "valid":
                target_index = [7, 10, 30, 38, 39, 63, 68, 81, 84, 91, 96, 107, 116, 121, 143, 180, 197, 204, 217, 246, 249, 257, 261, 268, 275, 294, 300, 307, 330, 334, 335, 336, 337, 338, 339, 340, 341, 343, 344, 365]

            else:
                target_index = [27, 57, 84, 92, 102, 127, 129, 142, 172, 183, 187, 197, 200, 208, 214, 216, 228, 237, 239, 253, 278, 290, 306, 309, 319, 326, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 343, 344, 345, 346, 347, 348, 349, 350]

        return target_index

    def set_x_y_set(self, name_of_set="test"):
        if COLUMN_TARGET:
            if name_of_set == "train":
                x_target = self.x_train
                y_target = self.y_train
            elif name_of_set == "valid":
                x_target = self.x_valid
                y_target = self.y_valid
            else:
                x_target = self.x_test
                y_target = self.y_test

            target = list()
            target_index = list()

            for index_dim, feature in self.feature.items():
                if feature[0] == COLUMN_TARGET:
                    target.append(int(index_dim))

            # initialize index of target symptom
            if not target:
                target_index = self.__hard_coding_for_school_paper(name_of_set)
                # print("example)")
                # print("python predict.py -model ffnn -vector V -tensor_dir D -save S -target T > target_index")
                # print("There is no Target in vector, Please make a new csv file!\n\n")
                # exit(-1)
            else:
                for index, x in enumerate(x_target):
                    if x[target[1]] == 1.0:
                        target_index.append(index)

            if name_of_set == "train":
                self.x_train = [x_target[index] for index in target_index]
                self.y_train = [y_target[index] for index in target_index]
            elif name_of_set == "valid":
                self.x_valid = [x_target[index] for index in target_index]
                self.y_valid = [y_target[index] for index in target_index]
            else:
                self.x_test = [x_target[index] for index in target_index]
                self.y_test = [y_target[index] for index in target_index]

    def __set_count(self):
        def __count_mortality(_y_data):
            _count = 0

            if len(_y_data[0]) > 1:
                death_vector = [0, 1]
            else:
                death_vector = [1]

            for _i in _y_data:
                if _i == death_vector:
                    _count += 1

            return _count

        self.count_all = [len(self.y_train), len(self.y_valid), len(self.y_test)]
        self.count_mortality = [__count_mortality(self.y_train),
                                __count_mortality(self.y_valid),
                                __count_mortality(self.y_test)]
        self.count_alive = [self.count_all[i] - self.count_mortality[i] for i in range(3)]

    def show_info(self):
        if DO_SHOW:
            print("\n\n\n======== DataSet Count ========")
            print("dims - ", len(self.x_train[0]))

            print("Training   Count -", str(self.count_all[0]).rjust(4),
                  "\t Mortality Count -", str(self.count_mortality[0]).rjust(3),
                  "\t Immortality Count -", str(self.count_alive[0]).rjust(4))

            print("Validation Count -", str(self.count_all[1]).rjust(4),
                  "\t Mortality Count -", str(self.count_mortality[1]).rjust(3),
                  "\t Immortality Count -", str(self.count_alive[1]).rjust(4))

            print("Test       Count -", str(self.count_all[2]).rjust(4),
                  "\t Mortality Count -", str(self.count_mortality[2]).rjust(3),
                  "\t Immortality Count -", str(self.count_alive[2]).rjust(4))

            print("\n\n======== DataSet Shape ========")
            x_train_np = np.array([np.array(j) for j in self.x_train])
            y_train_np = np.array([np.array(j) for j in self.y_train])
            print("Training   Set :", np.shape(x_train_np), np.shape(y_train_np))

            x_valid_np = np.array([np.array(j) for j in self.x_valid])
            y_valid_np = np.array([np.array(j) for j in self.y_valid])
            print("Validation Set :", np.shape(x_valid_np), np.shape(y_valid_np))

            y_test_np = np.array([np.array(j) for j in self.y_test])
            x_test_np = np.array([np.array(j) for j in self.x_test])
            print("Test       Set :", np.shape(x_test_np), np.shape(y_test_np), "\n\n")

    @staticmethod
    def expand4square_matrix(*vector_set_list, use_origin=False):
        # origin data set       = [ [ v_1,      v_2, ... ,      v_d ],                       .... , [ ... ] ]
        # expand data set       = [ [ v_1,      v_2, ... ,      v_d,        0.0, ..., 0.0 ], .... , [ ... ] ]
        # gray scale data set   = [ [ v_1*255,  v_2*255, ... ,  v_d*255,    0.0, ..., 0.0 ], .... , [ ... ] ]
        size_of_1d = len(vector_set_list[0][0])

        if INITIAL_IMAGE_SIZE and not use_origin:
            size_of_2d = INITIAL_IMAGE_SIZE ** 2
        else:
            size_of_2d = pow(math.ceil(math.sqrt(size_of_1d)), 2)

        print("\n\nThe matrix size of vector - %d by %d" % (math.sqrt(size_of_2d), math.sqrt(size_of_2d)))

        for vector_set in vector_set_list:
            for i, vector in enumerate(vector_set):
                # expand data for 2d matrix
                for _ in range(size_of_1d, size_of_2d):
                    vector.append(0.0)

                vector_set[i] = [v * GRAY_SCALE for v in vector]

    def set_image_path(self, vector_set_list, y_set_list, key_list):
        for key, vector_set, y_list in zip(key_list, vector_set_list, y_set_list):
            for enumerate_i, d in enumerate(zip(vector_set, y_list)):
                y_label = d[1]

                img_name = self.__get_name_of_image_from_index(key, enumerate_i, d[0])

                # img_name_list = [
                #     img_name + EXTENSION_OF_IMAGE,
                #     img_name + '_FLIP_LR' + EXTENSION_OF_IMAGE,
                #     img_name + '_FLIP_TB' + EXTENSION_OF_IMAGE,
                #     img_name + '_FLIP_LR_TB' + EXTENSION_OF_IMAGE,
                #     img_name + '_ROTATE' + EXTENSION_OF_IMAGE,
                #     img_name + '_ROTATE_FLIP_LR' + EXTENSION_OF_IMAGE,
                #     img_name + '_ROTATE_FLIP_TB' + EXTENSION_OF_IMAGE,
                #     img_name + '_ROTATE_FLIP_LR_TB' + EXTENSION_OF_IMAGE
                # ]

                if y_label == [1]:
                    vector_set[enumerate_i] = self.__get_image_from_path(IMAGE_PATH + deathPath + img_name)
                else:
                    vector_set[enumerate_i] = self.__get_image_from_path(IMAGE_PATH + alivePath + img_name)

    def get_image_vector(self, x_data):
        """

        :param x_data:
        :return:
        [
            [img_1_vector_1, img_1_vector_2, ... , img_1_vector_i]
            [img_2_vector_1, img_2_vector_2, ... , img_2_vector_i]
            ...
            [img_N_vector_1, img_N_vector_2, ... , img_N_vector_i]
        ]
        """
        return [[self.__get_image_from_path(path, to_img=True) for path in paths[1]] for paths in x_data]

    def __get_name_of_image_from_index(self, key, enumerate_i, x_index):
        # k cross validation (version 1)
        if VERSION == 1:
            length_of_train = len(self.y_train)
            length_of_valid = length_of_train + len(self.y_valid)

            if x_index < length_of_train:
                return "train_" + str(x_index + 1) + EXTENSION_OF_IMAGE
            elif length_of_train <= x_index < length_of_valid:
                x_index -= length_of_train
                return "valid_" + str(x_index + 1) + EXTENSION_OF_IMAGE
            else:
                x_index -= length_of_valid
                return "test_" + str(x_index + 1) + EXTENSION_OF_IMAGE

        # optimize hyper-parameters (version 2)
        elif VERSION == 2:
            return key + "_" + str(enumerate_i + 1) + EXTENSION_OF_IMAGE
        else:
            return None

    @staticmethod
    def __get_image_from_path(path, to_img=False):
        # normalize image of gray scale
        def __normalize_image(_img):
            gray_value = 255.0

            return np.array([[[k / gray_value for k in j] for j in i] for i in _img])

        img = Image.open(path)
        img.load()

        if to_img:
            if IMAGE_RESIZE:
                img = img.resize((IMAGE_RESIZE, IMAGE_RESIZE))

            new_img = np.asarray(img, dtype='int32')

            if DO_NORMALIZE:
                new_img = __normalize_image(new_img)
        else:
            # img = img.resize((INITIAL_IMAGE_SIZE, INITIAL_IMAGE_SIZE))
            new_img = np.asarray(img, dtype='int32')
            new_img = new_img.transpose([2, 0, 1]).reshape(3, -1)
            new_img = new_img[0]

        return new_img

    @staticmethod
    def reshape_image_for_cnn(x_data):
        def __change_gray_scale(_img):
            return np.array([[[j[0]] for j in i] for i in _img])

        cnt_data = len(x_data)

        x_data = [__change_gray_scale(img) for img in x_data]
        x_data = np.array(x_data)

        return list(x_data.reshape((cnt_data, -1)))

    def __random_forest(self):
        rf = RandomForestClassifier(n_estimators=NUM_OF_TREE, n_jobs=4, max_features='auto', random_state=0)
        return rf.fit(self.x_train, self.y_train)
        # return rf.fit(self.x_train + self.x_valid + self.x_test, self.y_train + self.y_valid + self.y_test)

    def __get_importance_features(self, feature, reverse=False):
        # reverse == T
        # --> get a not important features
        model = self.__random_forest()
        values = sorted(zip(feature.keys(), model.feature_importances_), key=lambda x: x[1] * -1)

        if reverse:
            return [(f[0], feature[f[0]], f[1]) for f in values if f[1] <= 0]
        else:
            return [(f[0], feature[f[0]], f[1]) for f in values if f[1] > 0]

    def show_importance_feature(self, reverse=False):
        feature_importance = self.__get_importance_features(self.feature, reverse=reverse)

        if reverse:
            if DO_SHOW:
                print("\n\nThere is not important feature")
                print("# of count -", len(feature_importance), "\n\n\n")
        else:
            for f in feature_importance:
                self.importance[f[1][0]] = [f[1][1], f[2]]

            if DO_SHOW:
                plot = MyPlot()
                plot.show_importance(feature_importance)

        if DO_SHOW:
            num_of_split_feature = 20
            for i, f in enumerate(feature_importance):
                print("%s (%s)\t %0.5f" % (str(f[1]).ljust(25), f[0], float(f[2])))
                if (i + 1) % num_of_split_feature == 0:
                        print("\n=======================================\n")

    def dump(self):
        with open(SAVE_LOG_NAME, 'w') as outfile:
            json.dump(self.importance, outfile, indent=4)
            print("\n=========================================================\n\n")
            print("success make dump file! - file name is", SAVE_LOG_NAME, "\n\n")
