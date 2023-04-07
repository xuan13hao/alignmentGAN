from Bio import SeqIO
from Bio.Seq import Seq
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

# read the sequences from the two FASTA files
sequences2 = SeqIO.index("gen.fa", "fasta")
sequences1 = SeqIO.index("real.fa", "fasta")

# create a list to store the similarity percentages for each sequence pair
similarity_list = []

# iterate over the two sequence lists and compare each sequence in file1 to each sequence in file2
for seq1_id in sequences1:
    for seq2_id in sequences2:
        seq1 = sequences1[seq1_id].seq
        seq2 = sequences2[seq2_id].seq
        # calculate the Hamming distance between the two sequences
        distance = sum(1 for a, b in zip(seq1, seq2) if a != b)
        # calculate the similarity percentage and store it in the list
        similarity_percentage = (len(seq1) - distance) / len(seq1) * 100
        similarity_list.append((seq1_id, seq2_id, similarity_percentage))

# sort the sequence pairs by their similarity percentage in descending order
sorted_list = sorted(similarity_list, key=lambda x: x[0], reverse=True)
seq_len = len(sequences1)
max_list = []
# for pair in sorted_list:
#     print(f"{pair[0]} \t {pair[1]} \t {pair[2]:.2f}")     
def chunks(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]   
chunks = chunks(sorted_list,seq_len)
for c in chunks:
    max_ = max(c, key=lambda x: x[2])
    max_list.append(max_[2])
print(max_list)
print("Average Accuracy = ",sum(max_list)/len(max_list))
# plot distribution

import matplotlib.pyplot as plt
import numpy as np

bins = np.linspace(0, 1, 11)

plt.hist(max_list, bins=bins, edgecolor='black', density=True)
plt.xlabel('Percentage Ranges')
plt.ylabel('Frequency')
plt.title('Histogram of Data with Percentage Ranges')

plt.legend(['Data'])
plt.show()
plt.savefig("distribution.png")


# dic = {}
# for pair in sorted_list:
#     # check if the key already exists in the dictionary
#     if pair[0] in dic:
#         # check if the existing value is already a tuple
#         if isinstance(dic[pair[0]], tuple):
#             # concatenate the new elements to the existing tuple
#             dic[pair[0]] = dic[pair[0]] + (pair[1], pair[2])
#         else:
#             # if the existing value is not a tuple, convert it to a tuple and concatenate
#             dic[pair[0]] = tuple([dic[pair[0]]]) + (pair[1], pair[2])
#     else:
#         # if the key does not exist in the dictionary, create a new tuple with the new elements
#         dic[pair[0]] = (pair[1], pair[2])

# for id, tp in dic.items():
#     print(id,":",tp)
