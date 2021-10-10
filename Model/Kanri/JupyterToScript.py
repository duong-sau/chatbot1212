import sys, json

def JTS():
    f = open("..\\.\\", 'r')  # input.ipynb
    j = json.load(f)
    of = open(sys.argv[2], 'w')  # output.py
    if j["nbformat"] >= 4:
        for i, cell in enumerate(j["cells"]):
            of.write("#cell " + str(i) + "\n")
            for line in cell["source"]:
                of.write(line)
            of.write('\n\n')
    else:
        for i, cell in enumerate(j["worksheets"][0]["cells"]):
            of.write("#cell " + str(i) + "\n")
            for line in cell["input"]:
                of.write(line)
            of.write('\n\n')

    of.close()


def STJ():
    from nbformat import current as nbf
    nb = nbf.read(open('test.py', 'r'), 'py')
    nbf.write(nb, open('test.ipynb', 'w'), 'ipynb')
STJ()