import matplotlib.pyplot as plt
from collections import Counter
from itertools import product
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
                k_dic = generate_all_kmers(k)
                kmers.append(k_dic[kmer])
    # count frequency of each k-mer
    kmer_counts = Counter(kmers)
    return kmer_counts

def generate_all_kmers(k):
    alphabet = "ACGT"
    kmers = [''.join(p) for p in product(alphabet, repeat=k)]
    # print(kmers)
    idx = 0
    kmer_dict = {}
    for kmer in kmers:
        idx = idx + 1
        kmer_dict[kmer] = idx
    return kmer_dict
# example usage
fasta_file_gen = 'gen.fa'
k = 3
kmer_counts_gen = count_kmers(fasta_file_gen, k)
kmer_counts_gen = dict(sorted(kmer_counts_gen.items()))
print(kmer_counts_gen)
# create bar chart of k-mer frequency
plt.bar(kmer_counts_gen.keys(), kmer_counts_gen.values())
plt.xlabel('K-mer sequence')
plt.ylabel('Frequency')
plt.title('K-mer Generated distribution (k={})'.format(k))
# plt.show()
plt.savefig('gen_kmer_dis.png')

fasta_file_real = 'real.fa'
k = 3
kmer_counts_real = count_kmers(fasta_file_real, k)
kmer_counts_real = dict(sorted(kmer_counts_real.items()))
print(kmer_counts_real)
# create bar chart of k-mer frequency
plt.bar(kmer_counts_real.keys(), kmer_counts_real.values())
plt.xlabel('K-mer sequence')
plt.ylabel('Frequency')
plt.title('K-mer Real distribution (k={})'.format(k))
# plt.show()
plt.savefig('real_kmer_dis.png')

diff = {k: kmer_counts_gen[k] - kmer_counts_real[k] for k in kmer_counts_gen}
plt.bar(diff.keys(), diff.values())
plt.xlabel('K-mer sequence')
plt.ylabel('Diff Frequency')
plt.title('K-mer Real distribution (k={})'.format(k))
# plt.show()
plt.savefig('diff.png')