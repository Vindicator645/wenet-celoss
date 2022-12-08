import sys, os
from collections import Counter

def dodir(path):
    global h

    for name in os.listdir(path):
        p = os.path.join(path, name)

        if os.path.islink(p):
            pass
        elif os.path.isfile(p):
            x=os.stat(p).st_size
            h[os.stat(p).st_size] += 1
        elif os.path.isdir(p):
            dodir(p)
        else:
            pass

def main(arg):
    global h
    h = {}
    file_size=1
    for i in range (100)
        h[file_size]=0
        file_size=file_size*2
    for dir in arg:
        dodir(dir)

    s = n = 0
    for k, v in sorted(h.items()):
        print("Size %d -> %d file(s)" % (k, v))
        n += v
        s += k * v
    print("Total %d bytes for %d files" % (s, n))

main(sys.argv[1:])