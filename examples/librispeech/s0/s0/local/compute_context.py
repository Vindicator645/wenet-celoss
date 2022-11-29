import sys

wer_file_list = sys.argv[1]
context_file = sys.argv[2]
wer_list = wer_file_list.split()
c_lines = open(context_file).readlines()
c_list = []
for c_line in c_lines:
    c_list.append(c_line.strip())
for file in wer_list:
    rec = []
    lab = []
    cnt = 0
    alarm = 0
    with open(file) as f:
        f_lines = f.readlines()
        for f_line in f_lines:
            if 'lab:' in f_line:
                lab.append(f_line)
            if 'rec:' in f_line:
                rec.append(f_line)
    for i in range(len(lab)):
        for c in c_list:
            if c in lab[i]:
                cnt += 1
                print(c)
                print(lab[i])
                print(rec[i])
                if c in rec[i]:
                    alarm += 1
    print(file,str(alarm*1.0/cnt))
        
