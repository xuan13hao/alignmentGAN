from Bio import SeqIO
import matplotlib.pyplot as plt
import sys
filename = sys.argv[1]

def count_length_distribution(filename):
    lengths = []
    for record in SeqIO.parse(filename, "fasta"):
        lengths.append(len(record.seq))
    return lengths

def count_length(filename):
    count_N = 0
    min_length = float('inf')
    max_length = 0
    for record in SeqIO.parse(filename, "fasta"):
        if 'N' in str(record.seq) or 'n' in str(record.seq):
            count_N = count_N + 1
        length = len(record.seq)
        if length < min_length:
            min_length = length
        max_length = max(max_length,length)
    return min_length,max_length,count_N


lengths = count_length_distribution(filename)

plt.hist(lengths, bins=50)
plt.title("Length Distribution")
plt.xlabel("Sequence Length")
plt.ylabel("Count")
plt.show()
plt.savefig("Length Distribution.png")


min_length,max_l, n_count  = count_length(filename)

print("Minimum sequence length:", min_length,", Max sequence length = ",max_l,", number of N : ",n_count)