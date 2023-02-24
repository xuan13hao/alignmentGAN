from Bio import SeqIO
fasta_file = "test.fa"
records = list(SeqIO.parse(fasta_file, "fasta"))

import numpy as np
nucleotides = ['A', 'C', 'G', 'T']
counts = np.zeros((len(records[0].seq), len(nucleotides)))
for record in records:
    for i, base in enumerate(record.seq):
        if base in nucleotides:
            counts[i, nucleotides.index(base)] += 1
freqs = counts / len(records)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for i, nucleotide in enumerate(nucleotides):
    ax.plot(freqs[:, i], label=nucleotide)
ax.legend()
ax.set_xlabel("Position")
ax.set_ylabel("Frequency")
ax.set_title("Per Base Sequence Content")
plt.savefig("squares.png")
plt.show()


