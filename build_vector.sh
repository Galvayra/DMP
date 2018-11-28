#/bin/bash

python encoding.py -input parsing_1 -output vector_1_w2v_scaling -w2v 1
python encoding.py -input parsing_2 -output vector_2 -w2v 0
python encoding.py -input parsing_2 -output vector_2_w2v_scaling -w2v 1
python encoding.py -input parsing_3 -output vector_3 -w2v 0
python encoding.py -input parsing_3 -output vector_3_w2v_scaling -w2v 1
python encoding.py -input parsing_4 -output vector_4 -w2v 0
python encoding.py -input parsing_4 -output vector_4_w2v_scaling -w2v 1
python encoding.py -input parsing_5 -output vector_5 -w2v 0
python encoding.py -input parsing_5 -output vector_5_w2v_scaling -w2v 1

