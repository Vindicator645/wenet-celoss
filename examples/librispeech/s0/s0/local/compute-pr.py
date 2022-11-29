#!/usr/bin/env python3

import sys
from xml.etree.ElementPath import prepare_predicate

# sys.argv[1]: 抄本
# sys.argv[2]: 模型预测结果
# sys.argv[3]: 热词列表

ft = open(sys.argv[1], "r")
fp = open(sys.argv[2], "r")
fc = open(sys.argv[3], "r")

#精确率 预测为热词的有多少是真热词
#召回率 真热词有多少被预测出来

hotwords = []

mp_true = {}
mp_false = {}
mp_miss = {}

mp_text = {}

line = fc.readline()
while line:
    line = line.strip().upper()
    if " " + line + " " not in mp_true:
        hotwords.append(" " + line + " ")
    mp_true[" " + line + " "] = 0
    mp_false[" " + line + " "] = 0
    mp_miss[" " + line + " "] = 0
    line = fc.readline()

line = ft.readline()
while line:
    key = line.strip().split()[0]
    line = " ".join(line.strip().split()[1:])
    mp_text[key] = line
    line = ft.readline()

count_true = 0
count_false = 0
count_miss = 0

cnt = 0

line_p = fp.readline()
while line_p:
    if line_p.strip().split()[0] not in mp_text:
        line_p = fp.readline()
        continue
    line_t = mp_text[line_p.strip().split()[0]]

    line_t = line_t.strip().upper()
    line_p = line_p.strip().upper()
    line_p = line_p.replace('<CONTEXT>', ' ')
    line_p = line_p.replace('</CONTEXT>', ' ')
    line_p = line_p.replace('  ', ' ')
    line_p = line_p.replace('  ', ' ')

    line_p = " ".join(line_p.strip().split()[1:])
    line_p = " " + line_p + " "
    line_t = " " + line_t + " "

    # cnt += 1
    # print("")
    # print(line_t)
    # print(line_p)
    # if cnt == 10:
    #     break

    for hotword in hotwords:
        real = line_t.count(hotword)
        predict = line_p.count(hotword)
        
        if real > predict:
            count_true += predict
            count_miss += real - predict
            mp_true[hotword] += predict
            mp_miss[hotword] += real - predict
        else :
            count_true += real
            count_false += predict - real
            mp_true[hotword] += real
            mp_false[hotword] += predict - real
    # print("true:",count_true, "  false:", count_false, "  miss:", count_miss)
    line_p = fp.readline()

if ((count_true + count_false) != 0):
    print("precision: ", 1.0 * count_true / (count_true + count_false) * 100, '%')
if ((count_true + count_miss) != 0):
    print("recall: ", 1.0 * count_true / (count_true + count_miss) * 100, '%')
print("true:",count_true, "  false:", count_false, "  miss:", count_miss)

ff = open("/home/work_nfs5_ssd/kxhuang/res.txt", "w")
for key in mp_true:
    cweight = 0.5 / ((mp_true[key] + 2.0 * mp_false[key] + 10) / (mp_true[key] + 2.0 * mp_miss[key] + 10))
    # ff.write(key + "\t" + str(mp_true[key] + mp_false[key]) + "   " + str(mp_true[key] + mp_miss[key]) + "\n")
    ff.write(key + " " + str(cweight) + "\n")




