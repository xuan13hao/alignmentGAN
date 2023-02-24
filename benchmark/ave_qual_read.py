from Bio import SeqIO
from math import log

def ave_qual(quals):
    """Calculate average basecall quality of a read.
    Receive the integer quality scores of a read and return the average quality for that read
    First convert Phred scores to probabilities,
    calculate average error probability
    convert average back to Phred scale
    """
    if quals:
        return -10 * log(sum([10**(q / -10) for q in quals]) / len(quals), 10)
    else:
        return None

def extract_from_fastq(fq):
    """Extract quality score from a fastq file."""
    ave_qual_reads = []
    for rec in SeqIO.parse(fq, "fastq"):
        ave_qual_reads.append(ave_qual(rec.letter_annotations["phred_quality"]))

qual_score = extract_from_fastq("test.fq")
print(qual_score)