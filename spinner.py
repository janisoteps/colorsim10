import sys
import time

spinner_obj = [
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', '8', ' ', ' ', ' ', '8', ' ', ' ', ' ', '8', ' ', ' ', '8', '8', ' ', ' ', '8', ' ', '8', ' ', ' ', ' ', '8', ' ', ' ', '8', ' '],
    [' ', ' ', '8', ' ', ' ', '8', ' ', '8', ' ', '8', ' ', '8', ' ', '8', ' ', '8', ' ', '8', ' ', '8', '8', ' ', ' ', '8', ' ', '8', ' ', '8'],
    [' ', ' ', '8', ' ', ' ', '8', ' ', '8', ' ', '8', ' ', '8', ' ', '8', ' ', '8', ' ', '8', ' ', '8', ' ', '8', ' ', '8', ' ', '8', ' ', ' '],
    [' ', ' ', '8', ' ', ' ', '8', ' ', '8', ' ', '8', '8', '8', ' ', '8', ' ', '8', ' ', '8', ' ', '8', ' ', ' ', '8', '8', ' ', '8', '8', '8'],
    [' ', ' ', '8', '8', '8', ' ', '8', ' ', ' ', '8', ' ', '8', ' ', '8', '8', ' ', ' ', '8', ' ', '8', ' ', ' ', ' ', '8', ' ', ' ', '8', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '8', ' ', '8', '8', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '8', ' ', '8', ' ', '8', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '8', ' ', '8', ' ', '8', ' ', '8', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '8', ' ', ' ', '8', '8', ' ', ' ', ' '],
]


def print_spinner(spinner_obj):
    for row in spinner_obj:
        str1 = ' '.join(row)
        print(str1)


for i in range(0, 26, 1):
    spinner_obj[11][2 + i] = '>'
    print_spinner(spinner_obj)
    time.sleep(0.1)
    spinner_obj[11][2 + i] = '='
    for n in range(0, 16, 1):
        # sys.stdout.write("\033[K")
        sys.stdout.write("\033[F")
    # if
