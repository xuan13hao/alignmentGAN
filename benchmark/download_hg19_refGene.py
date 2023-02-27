import urllib.request

url_hg19 = 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz'
filename_hg19 = 'hg19.fa.gz'
url_gtf = 'https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/genes/hg19.refGene.gtf.gz'
filename_gtf = 'hg19.refGene.gtf.gz'

# Download the file from the URL
urllib.request.urlretrieve(url_hg19, filename_hg19)
urllib.request.urlretrieve(url_gtf, filename_gtf)


print('GTF and FASTA FILES Download complete.')