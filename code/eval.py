
import torch

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
def convertDNAtoTensor(sequence):
    nt2int = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    seq_as_int = [nt2int[nt] for nt in sequence]
    input = torch.tensor(seq_as_int)
    return input

if __name__ == '__main__':
    target = "GAAAATCTCTCATGTACTCATTTAATAAAAATCAATACCTAGTACCAGGGAGGGAGATAATTGAATCTCTGTTGTCATTGTAGTCTCATGGTGGGAGGGAC"
    ref = "GAAAATCTCATCTTGAATTGTAGCTCCCATAATCCCCACATGTTGTGGGAGGGACCCAGTGGGAGATAATTGAATCATGGTGGTGGGTTTTCCCCATGCTG"
    t_tensor = convertDNAtoTensor(target)
    ref_tensor = convertDNAtoTensor(ref)
    dis = load_discrinimator()
    out_target = eval_discriminator(dis,t_tensor)
    out_ref = eval_discriminator(dis,ref_tensor)
    print(out_target)
    print(out_ref)