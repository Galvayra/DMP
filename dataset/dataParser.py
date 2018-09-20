from DMP.dataset.dataHandler import DataHandler
import math
import re


class DataParser(DataHandler):
    def __init__(self, data_file, is_reverse=False):
        super().__init__(data_file, is_reverse)

    def parsing(self):

        # {
        #   column: [ data_1, ... , data_n ]
        #   C: [ C_1, C_2, ... , C_n ]          ## ID
        #   E: [ .... ]                         ## Age
        #   ...
        #   CZ: [ .... ]                        ## Final Diagnosis
        # }
        #

        for header in list(self.x_data_dict.keys()):
            column_of_type = self.get_type_of_column(header)
            data_dict = self.__init_data_dict(self.x_data_dict[header])

            if column_of_type == "scalar":
                self.__parsing_scalar(header, data_dict)
            elif column_of_type == "class":
                self.__parsing_class(header, data_dict)
            elif column_of_type == "symptom":
                self.__parsing_symptom(header, data_dict)
            elif column_of_type == "mal_type":
                self.__parsing_mal_type(header, data_dict)

        # super().parsing()

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
                if data == "." or data == "-":
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
            _w = _w.replace('_-_', '_')
            _w = _w.replace('n/v', '')
            _w = _w.replace('_with_', '_')
            _w = _w.replace('_&_', '_')
            _w = _w.replace('_+_', '_')
            _w = _w.replace('_in_', '_')
            _w = _w.replace(',_', '_')
            _w = _w.replace('._', '_')
            _w = _w.replace('-', '_')

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

            if _w == "nan" or len(_w) <= 1:
                return "nan"

            _w = _w.strip().lower()
            _w = _w.replace('.', '. ')
            _w = _w.replace('(', ' ')
            _w = _w.replace(')', ' ')
            _w = "_".join(_w.split())
            _w = "_" + _w + "_"
            _w = re.sub('[.,:;]', '_', _w)
            _w = _w.replace('_ca_', '_cancer_')
            _w = _w.replace('_adenoca_', '_adeno_carcinoma_')
            _w = _w.replace('_adenocarcinoma_', '_adeno_carcinoma_')
            _w = _w.replace('_sqcc_', '_squamos_cell_carcinoma_')
            _w = _w.replace('_cll_', '_chronic_lymphocytic_leukemia_')
            _w = _w.replace('_cml_', '_chronic_myelomonocytic_leukemia_')
            _w = _w.replace('_all_', '_acute_lymphocytic_leukemia_')
            _w = _w.replace('_aml_', '_acute_myelomonocytic_leukemia_')
            _w = _w.replace('_cmmol_', '_chronic_myelomonocytic_leukemia_')
            _w = _w.replace('_lt_', '_left_')
            _w = _w.replace('_rt_', '_right_')
            _w = _w.replace('_cervic_', '_cervical_')
            _w = _w.replace('_cholangiocarcinoma_', '_cholangio_carcinoma_')
            _w = _w.replace('_with_', '_')
            _w = _w.replace('_&_', '_')

            _w = re.sub('[가-힣]+', '', _w)
            _w = re.sub('(([0-9]+[a-z]+)|([a-z]+[0-9]+))', '#', _w)
            _w = re.sub('([a-z]*[#]+[a-z]*)', '?', _w)
            _w = re.sub('[_]+', '_', _w)

            if _w:
                return _w
            else:
                return "nan"

        for data in sorted(data_dict):
            positions = data_dict[data]

            # print(data)

            print(data.ljust(80), __parsing(data))
            print()
            # self.__modify_dict(header, positions[1], __parsing(data))
