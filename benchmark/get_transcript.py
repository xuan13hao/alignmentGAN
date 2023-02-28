import pyfaidx
import pandas as pd

# Open the reference genome file in GTF format and read it into a pandas DataFrame
gtf_file = 'hg19.refGene.gtf'
df = pd.read_csv(gtf_file, sep='\t', comment='#', header=None, 
                 names=['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'])

# Filter the DataFrame to retain only transcripts
df_transcripts = df[df['feature'] == 'transcript']

# Extract the transcript sequences and write them to a FASTA file
with open('human_transcriptome.fa', 'w') as outfile:
    # Open the reference genome file in FASTA format using pyfaidx
    fasta_file = 'hg19.fa'
    genome = pyfaidx.Fasta(fasta_file, sequence_always_upper=True)

    for i, row in df_transcripts.iterrows():
        # Extract the transcript sequence from the reference genome
        if row.end - (row.start-1) >= 100 :
            transcript_seq = genome[row.seqname][row.start-1:row.end].seq
            # Write the transcript sequence to the output FASTA file
            header = f">{row.attribute.split(';')[0].split()[1][1:-1]}|{row.seqname}:{row.start}-{row.end}"
            outfile.write(f"{header}\n{transcript_seq}\n")