from .variables import W2V_FILE, W2V_PATH
from DMP.utils.arg_encoding import USE_W2V



class W2vReader:
    def __init__(self):
        self.w2v_dict = dict()
        self.__load_model()
        self.__pos_tag = "_noun"
        self.__dimension = 300

    @property
    def pos_tag(self):
        return self.__pos_tag

    @property
    def dimension(self):
        return self.__dimension

    def __load_model(self):
        if USE_W2V:
            w2v_file = W2V_PATH + W2V_FILE

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

            print("Complete Loading!!\n\n")

    def has_key_in_w2v_dict(self, key):
        if key + self.pos_tag in self.w2v_dict:
            return True
        else:
            return False

    def get_w2v_vector(self, word, vector_dict):
        w2v_vector = [float(0) for _ in range(self.dimension)]
        cnt = 0

        # sum of vectors
        for w in word:
            if w in vector_dict:
                for i, v in enumerate(self.w2v_dict[w + self.pos_tag]):
                    w2v_vector[i] += v
                cnt += 1

        # divide into vector
        if cnt:
            return [v/cnt for v in w2v_vector]
        else:
            return w2v_vector

    def __scaling_vector(self, w2v_vector):
        #### pass

