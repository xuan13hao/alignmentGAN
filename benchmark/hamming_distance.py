from Bio import SeqIO
from Bio.Seq import Seq
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

# read the sequences from the two FASTA files
sequences1 = SeqIO.index("gen.fa", "fasta")
sequences2 = SeqIO.index("real.fa", "fasta")

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
sorted_list = sorted(similarity_list, key=lambda x: x[2], reverse=True)

# print the sorted sequence pairs
for pair in sorted_list:
    print(f"{pair[0]} vs {pair[1]}: {pair[2]:.2f}% similarity")


