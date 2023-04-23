from Bio import SeqIO
from Bio.Seq import Seq
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

# read the sequences from the two FASTA files
sequences2 = SeqIO.index("gen.fa", "fasta")
sequences1 = SeqIO.index("real_1000.fa", "fasta")

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
three_ = []
four_ = []
five_ = []
six_ = []
seve_ = []
eight_ = []
nine_ = []
one_hun = []
for i in max_list:
    i = int(i)
    if 30 <= i < 40:
        three_.append(i)
    elif 40 <= i < 50:
        four_.append(i)
    elif 50 <= i < 60:
        five_.append(i)
    elif 60 <= i < 70:
        six_.append(i)
    elif 70 <= i < 80:
        seve_.append(i)
    elif 80 <= i < 90:
        eight_.append(i)
    elif 90 <= i < 100:
        nine_.append(i)
    elif 99 < i <= 101:
        one_hun.append(i)
print("100% = ",len(one_hun))
print("90-100% = ",len(nine_))
print("80-90% = ",len(eight_))
print("70-80% = ",len(seve_))
print("60-70% = ",len(six_))
print("50-60% = ",len(five_))
print("40-50% = ",len(four_))
print("30-40% = ",len(three_))