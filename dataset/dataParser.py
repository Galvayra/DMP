from DMP.dataset.dataHandler import DataHandler
import math
import re
from DMP.utils.arg_parsing import COLUMN_TARGET, COLUMN_TARGET_NAME, DO_SAMPLING


class DataParser(DataHandler):
    def __init__(self, read_csv):
        if COLUMN_TARGET_NAME:
            print("The Target is", COLUMN_TARGET_NAME, "\n\n")
        else:
            print("The Target is None\n\n")

        super().__init__(read_csv, do_parsing=True, do_sampling=DO_SAMPLING, column_target=COLUMN_TARGET)

    def parsing(self):

        # {
        #   column: [ data_1, ... , data_n ]
        #   C: [ C_1, C_2, ... , C_n ]          ## ID
        #   E: [ .... ]                         ## Age
        #   ...
        #   CZ: [ .... ]                        ## Final Diagnosis
        # }
        #

        for header in self.header_list:
            type_of_column = self.get_type_of_column(header)
            data_dict = self.__init_data_dict(self.x_data_dict[header])

            if type_of_column == "scalar":
                self.__parsing_scalar(header, data_dict)
            elif type_of_column == "class":
                self.__parsing_class(header, data_dict)
            elif type_of_column == "symptom":
                self.__parsing_symptom(header, data_dict)
            elif type_of_column == "mal_type":
                self.__parsing_mal_type(header, data_dict)
            elif type_of_column == "word":
                self.__parsing_word(header, data_dict)
            elif type_of_column == "diagnosis":
                self.__parsing_diagnosis(header, data_dict)

        super().parsing()

    def save(self):
        super().save()

    @staticmethod
    def __init_data_dict(data_lines):
        data_dict = dict()

        for k, v in data_lines.items():
            # key = str(k) + "@" + str(v)
            if v not in data_dict:
                data_dict[v] = [1, [k]]
            else:
                data_dict[v][0] += 1
                data_dict[v][1].append(k)

        return data_dict

    def __modify_dict(self, header, positions, value):
        for position in positions:
            self.x_data_dict[header][position] = value

    def __parsing_gcs_total(self):
        for position in list(self.x_data_dict["P"].keys()):
            self.x_data_dict["P"][position] = 0.0
            for gcs in ["M", "N", "O"]:
                self.x_data_dict["P"][position] += float(self.x_data_dict[gcs][position])

    def __parsing_scalar(self, header, data_dict):
        if header == "P":
            self.__parsing_gcs_total()
        else:
            for data, positions in data_dict.items():
                # Regular Expression for finding float in the data
                re_data = re.findall(r"[-+]?\d*\.\d+|\d+", data)

                # if data is existed
                if len(re_data) == 1:
                    self.__modify_dict(header, positions[1], float(re_data[0]))
                # if data is Null
                elif len(re_data) == 0:
                    self.__modify_dict(header, positions[1], "nan")
                # something error   ex) 9..9
                # have to parsing manually
                else:
                    print("Exception in the parsing scalar process !!")
                    print(header, re_data, positions)
                    exit(-1)

    def __parsing_class(self, header, data_dict):
        for data, positions in data_dict.items():
            try:
                data = float(data)
            except ValueError:
                # data is Null, even if data is existed
                if data == "." or data == "-" or data == "..":
                    self.__modify_dict(header, positions[1], "nan")
                # parsing error    ex) "+++    5000" -> "+++5000"
                else:
                    self.__modify_dict(header, positions[1], "".join(data.split()))
            else:
                # data is Null
                if math.isnan(data):
                    self.__modify_dict(header, positions[1], "nan")
                else:
                    self.__modify_dict(header, positions[1], str(data))

    def __parsing_symptom(self, header, data_dict):
        def __parsing(_w):
            _w = _w.strip().lower()
            _w = _w.replace('.', '. ')
            _w = _w.replace('(', ' ')
            _w = _w.replace(')', ' ')
            _w = "_".join(_w.split())
            _w = "_" + _w + "_"
            _w = _w.replace('_abd._', '_abdominal_')
            _w = _w.replace('_lt._', '_left_')
            _w = _w.replace('_rt._', '_right_')
            _w = _w.replace('_avf_', '_angioplasty_fails_')
            _w = _w.replace('_ptbd_', '_percutaneous_transhepatic_biliary_drainage_')
            _w = _w.replace('_bp_', '_blood_pressure_')
            _w = _w.replace('_cbc_', '_complete_blood_count_')
            _w = _w.replace('_ct_', '_computed_tomography_')
            _w = _w.replace('_lft_', '_liver_function_tests_')
            _w = _w.replace('_wbc_', '_white_blood_cell_')
            _w = _w.replace('_llq_', '_left_lower_quadrant_')
            _w = _w.replace('_luq_', '_left_upper_quadrant_')
            _w = _w.replace('_rlq_', '_right_lower_quadrant_')
            _w = _w.replace('_ruq_', '_right_upper_quadrant_')
            _w = _w.replace('_ugi_', '_upper_gastrointestinal_')
            _w = _w.replace('_hd_cath._', '_hemodialysis_catheter_')
            _w = _w.replace('_cath._', '_catheter_')
            _w = _w.replace('_exam._', '_examination_')
            _w = _w.replace('_t-tube_', '_tracheostomy_tube_')
            _w = _w.replace('_l-tube_', '_levin_tube_')
            _w = _w.replace('_peg_tube_', '_percutaneous_endoscopic_gastrostomy_tube_')
            _w = _w.replace('_op_', '_postoperative_')
            _w = _w.replace('_op._', '_postoperative_')
            _w = _w.replace('_lac._', '_laceration_')
            _w = _w.replace('_vesicles_', '_vesicle_')
            _w = re.sub('_(with|without|&|\+|in|-|for)_', '_', _w)
            _w = _w.replace('n/v', '')
            _w = _w.replace(',_', '_')
            _w = _w.replace('._', '_')
            _w = _w.replace('-', '_')

            # concat token '_'
            _w = re.sub('[_]+', '_', _w)
            _w = _w[1:-1]

            if _w:
                return _w
            else:
                return "nan"

        for data in sorted(data_dict):
            positions = data_dict[data]
            self.__modify_dict(header, positions[1], __parsing(data))

    def __parsing_mal_type(self, header, data_dict):
        def __parsing(_w):

            # process "_a_colon_" -> "_a-colon_"
            def __process_under_bar(__w, colon_or_cell):
                f = re.findall('_[a-z]_' + colon_or_cell + '_', __w)

                if f:
                    f = f[0].split('_')
                    f = "_" + f[1] + "-" + f[2] + "_"
                    return re.sub('_[a-z]_' + colon_or_cell + '_', f, __w)
                else:
                    return __w

            # process "symptom1/symptom2/..../symptomN" -> "symptom1_symptom2_...._symptomN"
            def __process_slash(__w):

                while True:
                    f = re.findall('[a-z]{2,}/[a-z]{2,}', __w)

                    if f:
                        __w = re.sub('[a-z]{2,}/[a-z]{2,}', '_'.join(f[0].split('/')), __w)
                    else:
                        return __w

            if _w == "nan" or len(_w) <= 1:
                return "nan"

            _w = _w.strip().lower()
            _w = _w.replace('.', '. ')

            _w = __process_slash(_w)

            # replace '(A)', '(B)' -> ''
            _w = re.sub('\([a-z]\)', '', _w)
            _w = _w.replace('(', ' ')
            _w = _w.replace(')', ' ')
            _w = "_".join(_w.split())
            _w = "_" + _w + "_"
            _w = re.sub('[.,:;]', '_', _w)
            _w = _w.replace('_ca_', '_cancer_')
            _w = _w.replace('_adenoca_', '_adeno_carcinoma_')
            _w = _w.replace('_adenocarcinoma_', '_adeno_carcinoma_')
            _w = _w.replace('_agc_', '_advanced_gastric_cancer_')
            _w = _w.replace('_agca_', '_advanced_gastric_cancer_')
            _w = _w.replace('_egc_', '_early_gastric_cancer_')
            _w = _w.replace('_egca_', '_early_gastric_cancer_')
            _w = _w.replace('_sqcc_', '_squamous_cell_carcinoma_')
            _w = _w.replace('_cll_', '_chronic_lymphocytic_leukemia_')
            _w = _w.replace('_cml_', '_chronic_myelomonocytic_leukemia_')
            _w = _w.replace('_all_', '_acute_lymphocytic_leukemia_')
            _w = _w.replace('_aml_', '_acute_myelomonocytic_leukemia_')
            _w = _w.replace('_cmmol_', '_chronic_myelomonocytic_leukemia_')
            _w = _w.replace('_dlbcl_', '_diffuse_large_b_cell_lymphoma_')
            _w = _w.replace('_dlbl_', '_diffuse_large_b_cell_lymphoma_')
            _w = _w.replace('_hcc_', '_hepatocellular_carcinoma_')
            _w = _w.replace('_nsclc_', '_non_small_cell_lung_cancer_')
            _w = _w.replace('_nsclca_', '_non_small_cell_lung_cancer_')
            _w = _w.replace('_rcc_', '_renal_cell_carcinoma_')
            _w = _w.replace('_rll_', '_right_lower_lobe_')
            _w = _w.replace('_gb_', '_gallbladder_')
            _w = _w.replace('_lt_', '_left_')
            _w = _w.replace('_rt_', '_right_')
            _w = _w.replace('_cervic_', '_cervical_')
            _w = _w.replace('_gist_', '_gastro_intestinal_stromal_tumors_')
            _w = _w.replace('_cholangiocarcinoma_', '_cholangio_carcinoma_')
            _w = _w.replace('/nsclc', 'nsclc')
            _w = _w.replace('/bladder', 'bladder')
            _w = re.sub('_(with|without|&|of|or|and|from|for)_', '_', _w)
            _w = __process_under_bar(_w, "colon")
            _w = __process_under_bar(_w, "cell")

            # erase hangle
            _w = re.sub('[가-힣]+', '', _w)

            # erase complex word    ex) m34tr3k3
            _w = re.sub('(([0-9]+[a-z]+)|([a-z]+[0-9]+))', '#', _w)
            _w = re.sub('([a-z]*[#]+[a-z]*)', '_', _w)

            # erase noise           ex) number, special words
            _w = re.sub('_[0-9+_/]+_', '_', _w)
            _w = re.sub('_[\W_]+_', '_', _w)
            _w = re.sub("['`]s", '', _w)

            # concat token '_'
            _w = re.sub('[_]+', '_', _w)
            _w = _w[1:-1]

            if _w:
                return _w
            else:
                return "nan"

        for data in sorted(data_dict):
            positions = data_dict[data]
            self.__modify_dict(header, positions[1], __parsing(data))

    def __parsing_word(self, header, data_dict):
        def __parsing(_w):

            # process "blood_1" -> "blood1"
            def __process_blood(__w):
                f = re.findall('blood_[0-9]', __w)

                if f:
                    __w = re.sub('blood_[0-9]', ''.join(f[0].split('_')), __w)

                f = re.findall('blood1,2', __w)

                if f:
                    f = f[0].split(',')
                    f = f[0] + '_blood' + f[1] + '_'
                    return re.sub('blood1,2', f, __w)
                else:
                    return __w

            _w = _w.strip().lower()
            _w = _w.replace('.', '. ')
            _w = "_".join(_w.split())
            _w = "_" + _w + "_"

            _w = __process_blood(_w)

            _w = re.sub("\([\d\D]{2,}\)", '', _w)
            _w = re.sub('[&,:]', '', _w)
            _w = _w.replace('->', '_')
            _w = _w.replace('/', '_')
            _w = _w.replace('_-_', '_')
            _w = _w.replace('-.', '')

            # concat token '_'
            _w = re.sub('[_]+', '_', _w)
            _w = _w[1:-1]

            if len(_w) <= 1:
                return "nan"
            elif _w:
                return _w
            else:
                return "nan"

        for data in sorted(data_dict):
            positions = data_dict[data]
            self.__modify_dict(header, positions[1], __parsing(data))

    def __parsing_diagnosis(self, header, data_dict):
        def __parsing(_w):

            # process "stage_1" -> "stage1", "type_1" -> "type1"
            def __process_stage_and_type(__w, keyword):
                f = re.findall('_' + keyword + '_[0-9]_', __w)

                # print(f, __w)
                if f:
                    f = f[0].split('_')
                    f = '_' + keyword + f[2] + '_'
                    return re.sub('_' + keyword + '_[0-9]_', f, __w)
                else:
                    return __w

            # process "symptom1/symptom2/..../symptomN" -> "symptom1_symptom2_...._symptomN"
            def __process_slash(__w):

                while True:
                    f = re.findall('[a-z]{2,}/[a-z]{2,}', __w)

                    if f:
                        __w = re.sub('[a-z]{2,}/[a-z]{2,}', '_'.join(f[0].split('/')), __w)
                    else:
                        return __w

            _w = _w.strip().lower()
            _w = _w.replace('.', '. ')
            _w = _w.replace(',', '_')
            _w = "_".join(_w.split())
            _w = "_" + _w + "_"

            # process '(' and ')'
            _w = re.sub("\([\d\D]{2,}\)", '', _w)
            _w = _w.replace('(', '_')
            _w = _w.replace(')', '_')

            # erase hangle
            _w = re.sub('[가-힣]+', '', _w)

            _w = _w.replace("_a_fib_", '_a-fib_')
            _w = _w.replace("_a._fib_", '_a-fib_')
            _w = _w.replace("_a_colon_", '_a-colon_')
            _w = _w.replace("_a._colon_", '_a-colon_')
            _w = _w.replace("_s_colon_", '_s-colon_')
            _w = _w.replace("_s._colon_", '_s-colon_')
            _w = _w.replace("_e_coli_", '_e-coli_')
            _w = _w.replace("_e._coli_", '_e-coli_')
            _w = _w.replace("_e._varix_", '_varix_')
            _w = _w.replace("_b_cell_", '_b-cell_')
            _w = _w.replace("_ca._", '_cancer_')
            _w = _w.replace("_ca_", '_cancer_')
            _w = _w.replace('_lt._', '_left_')
            _w = _w.replace('_rt._', '_right_')
            _w = _w.replace('_lt_', '_left_')
            _w = _w.replace('_rt_', '_right_')

            _w = __process_stage_and_type(_w, 'stage')
            _w = __process_stage_and_type(_w, 'type')
            _w = __process_slash(_w)

            _w = re.sub('[&.>?]', '', _w)
            _w = re.sub('_(with|without|of|or|from|and|for)_', '_', _w)
            _w = _w.replace('-_', '_')
            _w = _w.replace(';', '_')

            # concat token '_'
            _w = re.sub('[_]+', '_', _w)
            _w = _w[1:-1]

            if len(_w) <= 1:
                return "nan"
            elif _w:
                return _w
            else:
                return "nan"

        for data in sorted(data_dict):
            positions = data_dict[data]
            self.__modify_dict(header, positions[1], __parsing(data))

            # print(data.ljust(70), __parsing(data))
