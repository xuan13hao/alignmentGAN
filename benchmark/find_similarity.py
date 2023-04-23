import itertools
import numpy as np
from Bio import pairwise2, SeqIO
from Bio.SeqRecord import SeqRecord
# Load fasta files
seqs1 = list(SeqIO.parse("gen.fa", "fasta"))
seqs2 = list(SeqIO.parse("real.fa", "fasta"))

# Compute pairwise sequence similarities
similarities = np.zeros((len(seqs1), len(seqs2)))
for i, j in itertools.product(range(len(seqs1)), range(len(seqs2))):
    alignment = pairwise2.align.globalxx(seqs1[i].seq, seqs2[j].seq, one_alignment_only=True)
    similarity = alignment[0].score / max(len(seqs1[i]), len(seqs2[j]))
    if similarity >= 0.9:
        similarities[i, j] = similarity

# Rank similarities in descending order
indices = np.dstack(np.unravel_index(np.argsort(similarities.ravel())[::-1], similarities.shape))[0]

seq_records1 = []
seq_records2 = []
for i, j in indices:
    if similarities[i, j] == 0:
        break
    seq_record1 = SeqRecord(seqs1[i].seq, id=seqs1[i].id, description="")
    seq_record2 = SeqRecord(seqs2[j].seq, id=seqs2[j].id, description="")
    if similarities[i, j] >= 0.9:
        seq_records1.append(seq_record1)
        seq_records2.append(seq_record2)

SeqIO.write(seq_records1, "gen.fa", "fasta")
SeqIO.write(seq_records2, "real_1000.fa", "fasta")
