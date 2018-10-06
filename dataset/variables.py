# -*- coding: utf-8 -*-
from collections import OrderedDict

DATA_PATH = "dataset/"

POSITION_OF_ROW = 2
ID_COLUMN = "C"
RR_COLUMN = "K"
TEMP_COLUMN = "L"

default_info_columns = {
    "scalar": ['E'],
    "class": ['F'],
    "id": [ID_COLUMN]
}

initial_info_columns = {
    "scalar": ["H", 'I', 'J', RR_COLUMN, TEMP_COLUMN, 'M', 'N', 'O', 'P'],
    "class": ['Q', 'R', 'S', 'T', 'U'],
    "symptom": ['G']
}

past_history_columns = {
    "class": ['V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD'],
    "mal_type": ['AE']
}

blood_count_columns = {
    "scalar": ['AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AO']
}

blood_chemistry_columns = {
    "scalar": ['AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ',
               'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK',
               'BL', 'BM', 'BN', 'BO', 'BP', 'BQ', 'BR', 'BS', 'BT'],
    "class": ['BU', 'BV', 'BW']
}

abga_columns = {
    "scalar": ['BX', 'BY', 'BZ', 'CA', 'CB', 'CC', 'CD']
}

culture_columns = {
    "class": ['CE', 'CF', 'CH', 'CI', 'CK', 'CL'],
    "word": ['CG', 'CJ', 'CM']
}

influenza_columns = {
    "class": ['CN', 'CO']
}

ct_columns = {
    "class": ['CP', 'CQ']
}

final_diagnosis_columns = {
    "class": ['CR', 'CS', 'CT', 'CU', 'CV', 'CW', 'CX', 'CY']
    # "diagnosis": ['CZ']
}

columns_dict = OrderedDict()
columns_dict["default"] = default_info_columns
columns_dict["initial"] = initial_info_columns
columns_dict["history"] = past_history_columns
columns_dict["b_count"] = blood_count_columns
columns_dict["b_chemistry"] = blood_chemistry_columns
columns_dict["abga"] = abga_columns
columns_dict["culture"] = culture_columns
columns_dict["influenza"] = influenza_columns
columns_dict["ct"] = ct_columns
columns_dict["f_diagnosis"] = final_diagnosis_columns
