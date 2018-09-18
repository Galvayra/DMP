from DMP.dataset.dataHandler import DataHandler
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

            # self.__inspect_columns(data_lines)
            # print(header, len(data_lines))

            if column_of_type == "scalar":
                self.__parsing_scalar(header, data_dict)

            # for data, count in data_dict.items():
            #     if column_of_type == "scalar":
            #         print(data, count)
            #

            # print()
            # print()

        self.show_type_of_columns()

        # super().parsing()

    def save(self):
        print("save csv file after parsing!!")

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

    def __parsing_scalar(self, header, data_dict):
        for data, positions in data_dict.items():

            # Regular Expression for finding float in the data
            re_data = re.findall(r"[-+]?\d*\.\d+|\d+", data)

            if len(re_data) == 1:
                for position in positions[1]:
                    self.x_data_dict[header][position] = float(re_data[0])

            elif len(re_data) == 0:
                for position in positions[1]:
                    self.x_data_dict[header][position] = "nan"

            else:
                print("Exception in the parsing scalar process !!")
                print(header, re_data, positions)
                exit(-1)