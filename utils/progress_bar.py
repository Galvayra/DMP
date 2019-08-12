import sys


def show_progress_bar(iteration, total, prefix='', suffix='', decimals=1, bar_length=50):
    if iteration == 1:
        sys.stdout.write('\n')

    format_str = "{0:." + str(decimals) + "f}"
    percent = format_str.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix.ljust(7), bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n\n')
    sys.stdout.flush()
