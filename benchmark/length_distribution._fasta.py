from Bio import SeqIO
import matplotlib.pyplot as plt

def count_length_distribution(filename):
    lengths = []
    for record in SeqIO.parse(filename, "fasta"):
        lengths.append(len(record.seq))
    return lengths
def count_min_length(filename):
    min_length = float('inf')
    for record in SeqIO.parse(filename, "fasta"):
        length = len(record.seq)
        if length < min_length:
            min_length = length
    return min_length

filename = "human_transcriptome.fa"
lengths = count_length_distribution(filename)

plt.hist(lengths, bins=50)
plt.title("Length Distribution")
plt.xlabel("Sequence Length")
plt.ylabel("Count")
plt.show()
plt.savefig("Length Distribution.png")


min_length = count_min_length(filename)

print("Minimum sequence length:", min_length)