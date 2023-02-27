import sys
from Bio import SeqIO

FastaFile = open(sys.argv[1], 'r')

total = 0
len_smaller_100 = 0
len_smaller_150 = 0
len_500_150 = 0
len_500 = 0
count_total = 0
count_500 = 0
count_150_500 = 0
count_150 = 0
count_100 = 0
for rec in SeqIO.parse(FastaFile, 'fasta'):
    name = rec.id
    seq = rec.seq
    # print(rec.description)
    seqLen = len(rec)
    if seqLen > 0:
        total = total + 1
        count_total = count_total + seqLen
    if seqLen > 500:
        len_500 = len_500 + 1
        count_500 = count_500 + seqLen
    elif 150 <= seqLen <= 500:
        len_500_150 = len_500_150 + 1
        count_150_500 = count_150_500 + seqLen
    elif 100<= seqLen < 150:
        len_smaller_150 = len_smaller_150 + 1
        count_150 = count_150 + seqLen
    elif seqLen < 100:
        len_smaller_100 = len_smaller_100 + 1
        count_100 = count_100 + seqLen        
    # print(name,":",seqLen)
print("less than 100: ",len_smaller_100,"smaller than :",len_smaller_150, ", 150-500: ", len_500_150, ",larger than 500 : ",len_500, ", total: ",total)
print("less than 100: ",count_100, "smaller than :",count_150, ", 150-500: ", count_150_500, ",larger than 500 : ",count_500, ", total: ",count_total)
FastaFile.close()