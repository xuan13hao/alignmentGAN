from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# First, read the two fasta files
fasta_file1 = "gen.fa"
fasta_file2 = "real.fa"


records1 = SeqIO.parse(fasta_file1, "fasta")
records2 = SeqIO.parse(fasta_file2, "fasta")

# Define a function to compute Hamming distance between two sequences
def hamming_distance(seq1, seq2):
    return sum(1 for a, b in zip(seq1, seq2) if a != b)

# Loop over the sequences in fasta file 1 and compare them to fasta file 2
for r1 in records1:
    for r2 in records2:
        similarity = float(len(r1.seq) - hamming_distance(r1.seq, r2.seq))/len(r1.seq)*100
        print(f"Similarity between {r1.id} and {r2.id} is {similarity}")
