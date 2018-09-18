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

        # self.show_type_of_columns()

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
