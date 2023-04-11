import matplotlib.pyplot as plt
from collections import Counter
from Bio import SeqIO

def count_kmers(fasta_file, k):
    # initialize empty list to store k-mers
    kmers = []
    # open fasta file and iterate over sequences
    with open(fasta_file) as f:
        for record in SeqIO.parse(f, "fasta"):
            # extract sequence and iterate over k-mers
            sequence = str(record.seq)
            for i in range(len(sequence)-k+1):
                kmer = sequence[i:i+k]
                kmers.append(kmer)
    # count frequency of each k-mer
    kmer_counts = Counter(kmers)
    return kmer_counts

# example usage
fasta_file = 'gen.fa'
k = 3
kmer_counts = count_kmers(fasta_file, k)

# create bar chart of k-mer frequency
plt.bar(kmer_counts.keys(), kmer_counts.values())
plt.xlabel('K-mer sequence')
plt.ylabel('Frequency')
plt.title('K-mer distribution (k={})'.format(k))
# plt.show()
plt.savefig('gen_kmer_dis.png')