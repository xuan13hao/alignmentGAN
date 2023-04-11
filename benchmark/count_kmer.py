import pysam
import itertools

fastafile = pysam.FastaFile("real.fa")  # open FASTA file for reading
kmers = ["".join(x) for x in itertools.product("ATCG", repeat=3)]
counts = {kmer: 0 for kmer in kmers}  # initialize count for each 3-mer to 0
# print(counts)
for name in fastafile.references:
    seq = fastafile.fetch(name)  # get sequence for current reference
    kmer = seq[:3]  # get first 3-mer sequence in read
    if kmer in counts:
        counts[kmer] += 1  # increment count if 3-mer sequence is found

fastafile.close()  # close FASTA file

print(counts)

# check if every possible 3-mer sequence has at least one read
if all(counts.values()):
    print("Every possible 3-mer sequence is present in at least one read.")
else:
    print("Some possible 3-mer sequences are not present in any read.")
