import pysam
import random
from itertools import product

import pysam

samfile = pysam.AlignmentFile("test.sam", "r")  # open SAM file for reading

reads = {}  # dictionary to store reads for each 3-mer sequence

# create list of all possible 3-mers
kmers = ["".join(x) for x in product("ATCG", repeat=3)]

# initialize dictionary with empty lists for each 3-mer sequence
for kmer in kmers:
    reads[kmer] = []

for read in samfile:
    seq = read.seq  # get read sequence
    # loop through all 3-mers in the read sequence
    if len(seq) == 101:
        for i in range(len(seq) - 2):
            kmer = seq[i:i+3]  # get 3-mer sequence
            if kmer in reads:
                reads[kmer].append(read)  # add read to dictionary if 3-mer sequence is found

samfile.close()  # close SAM file

# do something with the matching reads for each 3-mer sequence
print("kmer type num in sam file = ",len(reads))
for kmer, read_list in reads.items():
    print(f"{kmer}: {len(read_list)} reads")
