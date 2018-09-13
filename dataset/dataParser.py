from DMP.dataset.dataHandler import DataHandler
import math


class DataParser(DataHandler):
    def __init__(self, is_reverse=False):
        super().__init__(is_reverse)

    def parsing(self):

        # {
        #   column: [ data_1, ... , data_n ]
        #   C: [ C_1, C_2, ... , C_n ]          ## ID
        #   E: [ .... ]                         ## Age
        #   ...
        #   CZ: [ .... ]                        ## Final Diagnosis
        # }
        #

        for header, data_dict in self.x_data_dict.items():
            print(header)

            self.__inspect_columns(data_dict)
            # for data, count in self.__get_data_dict(data_dict).items():
            #     print(data, count[0])
            print()
            print()

        self.free()

    @staticmethod
    def __inspect_columns(data_dict):
        # show result of columns inspecting

        type_dict = {"total": 0}
        for _k, v in data_dict.items():

            key = 0
            if type(v) is float:
                if math.isnan(v):
                    key = "nan"
                else:
                    key = "float"
            elif type(v) is str:
                key = "str"
            elif type(v) is int:
                key = "int"

            if key not in type_dict:
                type_dict[key] = 1
            else:
                type_dict[key] += 1
            type_dict["total"] += 1

        print(type_dict)

    @staticmethod
    def __get_data_dict(data_dict):
        new_data_dict = dict()

        for k, v in data_dict.items():
            # key = str(k) + "@" + str(v)
            if v not in new_data_dict:
                new_data_dict[v] = [1, [k]]
            else:
                new_data_dict[v][0] += 1
                new_data_dict[v][1].append(k)

        return new_data_dict
