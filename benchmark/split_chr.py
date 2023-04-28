from Bio import SeqIO

# Read the chromosome 1 sequence from the file
with open("chrY.fa", "r") as f:
    record = SeqIO.read(f, "fasta")

# Split the sequence into fragments of length 100 and write them to a new FASTA file
with open("chrY_101.fa", "w") as f:
    for i in range(0, len(record), 101):
        fragment_start = i+1
        fragment_end = min(i+101, len(record))
        fragment_seq = str(record.seq[i:fragment_end]).upper()
        if "N" not in fragment_seq:
            f.write(">chrY:{:d}-{:d}\n".format(fragment_start, fragment_end))
            f.write(fragment_seq + "\n")
