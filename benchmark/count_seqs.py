from Bio import SeqIO
import sys


fasta = sys.argv[1]
def count_sequences(fasta_file):
    """
    This function takes a path to a FASTA file and returns the number of sequences in the file.
    """
    with open(fasta_file, "r") as handle:
        records = SeqIO.parse(handle, "fasta")
        num_sequences = sum(1 for record in records)
    return num_sequences

fasta_file = fasta
num_sequences = count_sequences(fasta_file)
print("Number of sequences:", num_sequences)