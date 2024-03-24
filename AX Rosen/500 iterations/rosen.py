import sys


def function(args):
    x1 = args[0]
    x2 = args[1]
    a = args[2]
    b = args[3]

    f = (a - x1)**2 + b * (x2 - x1**2)**2
    return f

def main(argv):

    x1 = float(argv[1])
    x2 = float(argv[2])
    a = float(argv[3])
    b = float(argv[4])

    return function([x1, x2, a, b])

if __name__ == '__main__':
    evaluated_function = main(sys.argv)
    sys.exit(evaluated_function)