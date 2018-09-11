from collections import OrderedDict

DUMP_PATH = "modeling/vectors/"
DUMP_FILE = "vectors"

KEY_NAME_OF_MERGE_VECTOR = "merge"
LOAD_WORD2VEC = "GoogleNews-vectors-negative300.bin"

default_info_columns = {
    "scalar": {
        "0": ['E']
    },
    "class": ['F'],
    "id": ["C"]
}

initial_info_columns = {
    "scalar": {
        "0": ["H", 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    },
    "class": ['Q', 'R', 'S', 'T', 'U'],
    "symptom": ['G']
}

past_history_columns = {
    "class": ['V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD'],
    "mal_type": ['AE']
}

blood_count_columns = {
    "scalar": {
        "0": ['AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AO']
    }
}

blood_chemistry_columns = {
    "scalar": {
        "0": ['AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ',
              'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BL', 'BM', 'BN', 'BO', 'BP',
              'BQ', 'BR', 'BS', 'BT']
    },
    "class": ['BU', 'BV', 'BW']
}

abga_columns = {
    "scalar": {
        "0": ['BX', 'BY', 'BZ', 'CA', 'CB', 'CC', 'CD']
    }
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
    "class": ['CR', 'CS', 'CT', 'CU', 'CV', 'CW', 'CX', 'CY'],
    "diagnosis": ['CZ']
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
