from .variables import W2V_FILE, W2V_PATH
from DMP.utils.arg_encoding import USE_W2V
from os import path


class W2vReader:
    def __init__(self):
        self.w2v_dict = dict()
        self.dimension = 300
        self.__load_model()
        self.__vector_w2v_dict = dict()
        self.__pos_tag = "_noun"

    @property
    def vector_w2v_dict(self):
        return self.__vector_w2v_dict

    @property
    def pos_tag(self):
        return self.__pos_tag

    def __load_model(self):
        if USE_W2V:
            w2v_file = path.dirname(path.abspath(__file__)) + '/' + W2V_PATH + W2V_FILE

            try:
                with open(w2v_file, "r") as r_file:
                    print("Load word2vec model -", w2v_file)

                    # add embed dictionary
                    for line in r_file:
                        line = line.split()
                        self.w2v_dict[line[0].lower()] = [float(v) for v in line[1:]]
            except FileNotFoundError:
                print("There is no File -", w2v_file, "\n")
                exit(-1)
            else:
                print("Complete Loading!!\n\n")

    def has_key_in_w2v_dict(self, key):
        if key + self.pos_tag in self.w2v_dict:
            return True
        else:
            return False

    def get_w2v_vector(self, word, column):
        w2v_vector = [float(0) for _ in range(self.dimension)]
        cnt = 0

        # sum into w2v vector
        for w in word:
            if w in self.vector_w2v_dict[column]:
                for i, v in enumerate(self.w2v_dict[w + self.pos_tag]):
                    w2v_vector[i] += v
                cnt += 1

        # divide vector using total count which is existed in dictionary
        w2v_vector = self.__normalize_vector(w2v_vector, cnt)

        # return w2v_vector
        # scaling values in vector from [-1, 1] to [0, 1]
        return self.__scaling_vector(w2v_vector)

    @staticmethod
    def __normalize_vector(w2v_vector, total_cnt):
        if total_cnt:
            return [v/total_cnt for v in w2v_vector]
        else:
            return w2v_vector

    @staticmethod
    def __scaling_vector(w2v_vector):
        return [(v + 1)/float(2) for v in w2v_vector]

