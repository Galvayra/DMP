from DMP.dataset.dataHandler import DataHandler


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
            for k, v in self.__get_data_dict(data_dict).items():
                print(k, v)
            print()
            print()

        self.free()

    @staticmethod
    def __get_data_dict(data_dict):
        new_data_dict = dict()

        for k, v in data_dict.items():
            # key = str(k) + "@" + str(v)
            if v not in new_data_dict:
                new_data_dict[v] = 1
            else:
                new_data_dict[v] += 1

        return new_data_dict
