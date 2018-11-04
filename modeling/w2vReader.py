from .variables import W2V_FILE, W2V_PATH
from DMP.utils.arg_encoding import USE_W2V


class W2vReader:
    def __init__(self):
        self.embed_dict = dict()
        self.load_model()

    def load_model(self):
        if USE_W2V:
            w2v_file = W2V_PATH + W2V_FILE

            try:
                with open(w2v_file, "r") as r_file:
                    print("\n\nLoad word2vec model -", w2v_file, "\n\n")

                    # add embed dictionary
                    for line in r_file:
                        line = line.split()
                        self.embed_dict[line[0].lower()] = [float(v) for v in line[1:]]

            except FileNotFoundError:
                print("There is no File -", w2v_file, "\n")
                exit(-1)
