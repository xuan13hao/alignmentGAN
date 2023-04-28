from Bio import SeqIO

# Read the hg19 file and iterate over its records
with open("hg19.fa", "r") as f_in, open("chrY.fa", "w") as f_out:
    for record in SeqIO.parse(f_in, "fasta"):
        if record.id == "chrY":
            # Write the chromosome 1 record to the output file
            SeqIO.write(record, f_out, "fasta")
            break
