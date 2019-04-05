# -*- coding: utf-8 -*-
from collections import OrderedDict

DATA_PATH = "dataset/"
ORIGIN_PATH = "origin/"
PARSING_PATH = "parsing/"

POSITION_OF_ROW = 2

COLUMN_NUMBER = "A"
COLUMN_HOSPITAL = "B"
ID_COLUMN = "C"
RR_COLUMN = "K"
TEMP_COLUMN = "L"
Y_COLUMN = "DA"

USE_SELECTED_FEATURES = False

if not USE_SELECTED_FEATURES:
    # All of features
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
        # vector_origin is not using diagnosis (this is a paper option)
        # vector is using diagnosis
        # "diagnosis": ['CZ']
    }
else:
    # # feature selection using vector_one_hot
    # default_info_columns = {
    #     "class": ['F'],
    #     "id": [ID_COLUMN]
    # }
    #
    # initial_info_columns = {
    #     "class": ['R', 'S', 'T', 'U']
    # }
    #
    # past_history_columns = {
    #     "class": ['V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD']
    # }
    #
    # blood_count_columns = {
    #     "scalar": ['AF', 'AG', 'AH', 'AK', 'AL', 'AM', 'AN', 'AO']
    # }
    #
    # blood_chemistry_columns = {
    #     "scalar": ['AP', 'AQ', 'AR', 'AS', 'AT', 'AV', 'AW', 'AX',
    #                'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK',
    #                'BL', 'BM', 'BN', 'BO', 'BP', 'BQ', 'BR', 'BS', 'BT']
    # }
    #
    # abga_columns = {
    #     "scalar": ['BX', 'BY', 'BZ', 'CA', 'CB', 'CC', 'CD']
    # }
    #
    # culture_columns = {
    #     "class": ['CE', 'CF', 'CH', 'CI', 'CK', 'CL']
    # }
    #
    # influenza_columns = {
    #     "class": ['CN', 'CO']
    # }
    #
    # ct_columns = {
    #     "class": ['CP', 'CQ']
    # }
    #
    # final_diagnosis_columns = {
    #     "class": ['CR', 'CS', 'CT', 'CU', 'CV', 'CW', 'CX', 'CY']
    # }

    # feature selection using vector_origin_one_hot
    default_info_columns = {
        "class": ['F'],
        "id": [ID_COLUMN]
    }

    initial_info_columns = {
        "class": ['R', 'S', 'T', 'U']
    }

    past_history_columns = {
        "class": ['V', 'W', 'X', 'Y', 'Z', 'AA', 'AC', 'AD']
    }

    blood_count_columns = {
        "scalar": ['AG', 'AH', 'AI', 'AK', 'AL', 'AM']
    }

    blood_chemistry_columns = {
        "scalar": ['AP', 'AQ', 'AR', 'AS', 'AV', 'AW', 'AY',
                   'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK',
                   'BL', 'BM', 'BN', 'BO', 'BP', 'BQ', 'BR', 'BS', 'BT']
    }

    abga_columns = {
        "scalar": ['BX', 'BY', 'BZ', 'CA', 'CB', 'CC', 'CD']
    }

    culture_columns = {
        "class": ['CE', 'CF', 'CI', 'CK', 'CL']
    }

    influenza_columns = {
        "class": ['CN', 'CO']
    }

    ct_columns = {
        "class": ['CP', 'CQ']
    }

    final_diagnosis_columns = {
        "class": ['CR', 'CS', 'CT', 'CU', 'CV', 'CW', 'CX', 'CY']
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
