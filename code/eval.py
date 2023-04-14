
import torch
from itertools import product
SEQ_LENGTH = 99
def load_discrinimator():
    dis = torch.load('discriminator.pkl')
    return dis

def eval_discriminator(discriminator, inputs):
    """
    Evaluate discriminator
    """
    # inputs = inputs.type(torch.LongTensor)
    inputs = inputs.cuda()
    inputs = inputs.view(1, -1)
    # print(inputs)
    # Evaluate discriminator
    discriminator.eval()
    correct_predictions = 0
    total_predictions = 0
    scores = discriminator.batchClassify(inputs)
    # predicted_labels = torch.round(scores)
    # print(scores)
    return scores

    # for batch in test_data_loader:
    #     # Get batch inputs and labels
    #     inputs = batch['inputs']
    #     labels = batch['labels']
        
    #     # Make predictions
    #     scores = discriminator.batchClassify(inputs)
    #     predicted_labels = torch.round(scores)
        
    #     # Count correct predictions
    #     correct_predictions += (predicted_labels == labels).sum().item()
    #     total_predictions += len(labels)

    # # Calculate accuracy
    # accuracy = correct_predictions / total_predictions
    # print('Discriminator accuracy:', accuracy)
def DNAToWord(dna, K ,pad_token='PAD',mask_token = 'MASK'):
    kmers = []
    length = len(dna)
    for i in range(length - K + 1):
        # sentence += dna[i: i + K] + " "
        seq = dna[i: i + K].upper()
        # if 'N' in seq:
        #     seq = mask_token
        kmers.append(seq) 
    if len(kmers) <= SEQ_LENGTH-1:
        kmers = kmers + [pad_token] * (SEQ_LENGTH - 1 - len(kmers))
    return kmers
    # sentence = sentence[0 : len(sentence) - 1]
    # kmers = [w for w in sentence]     
    
def decode_all_kmers(k):
    alphabet = "ACGT"
    kmers = [''.join(p) for p in product(alphabet, repeat=k)]
    # print(kmers)
    idx = 0
    kmer_dict = {}
    for kmer in kmers:
        idx = idx + 1
        kmer_dict[idx] = kmer
    kmer_dict[0] = "START"
    # kmer_dict[4097] = "PADDING"
    # print(kmer_dict)
    return kmer_dict
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
def generate_sequence(kmer_list):
    # print(kmer_list)
    sequence = kmer_list[0]
    k = len(kmer_list[0])
    for i in range(1, len(kmer_list)):
        sequence += kmer_list[i][k-1:]
    return sequence

def convertDNAtoTensor(sequence):
    # nt2int = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    nt2int = generate_all_kmers(3)
    seq_as_int = [nt2int[nt] for nt in sequence]
    input = torch.tensor(seq_as_int)
    return input

if __name__ == '__main__':
    target = "GAAAATCTCATCTTGAATTGTAGCTCCCATAATCCCCACATGTTGTCTTTCTTTATAAATTACCCAGTCTCGAGTATGTCTTTCTTAGCAGTGTGTGGCGC"
    ref = "GAAAATCTCATCTTGAATTGTAGCTCCCATAATCCCCACATGTTGTGGGAGGGACCCAGTGGGAGATAATTGAATCATGGTGGTGGGTTTTCCCCATGCTG"
    t = DNAToWord(target,3)
    r = DNAToWord(ref,3)
    t_tensor = convertDNAtoTensor(t)
    ref_tensor = convertDNAtoTensor(r)
    print(t_tensor)
    dis = load_discrinimator()
    out_target = eval_discriminator(dis,t_tensor)
    out_ref = eval_discriminator(dis,ref_tensor)
    print(out_target)
    print(out_ref)