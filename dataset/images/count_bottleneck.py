import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-input", "--input", help="set a bottleneck file path\n\n")
    return parser.parse_args()


args = get_arguments()


if __name__ == '__main__':
    try:
        with open(args.input, 'r') as r_file:
            bottleneck = r_file.read()
    except:
        print("File Not Found!\n\n")
        exit(-1)
    else:
        bottleneck = bottleneck.split(',')

        print(len(bottleneck))
