import os, sys

if len(sys.argv) != 3:
    print("python {} [label file] [dir path]".format(sys.argv[0]))
    exit(0)

with open(sys.argv[1]) as label_file:
    labels = label_file.read().split()[1:]
    for label in labels:
        img, direct = [os.path.join(sys.argv[2], f)
                            for f in label.split(',')]
        if not os.path.isdir(direct):
            os.system('mkdir {}'.format(direct))
        os.system('mv {} {}'.format(img, direct))
