import urllib.request

url = 'https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/genes/hg19.refGene.gtf.gz'
filename = 'hg19.refGene.gtf.gz'

# Download the file from the URL
urllib.request.urlretrieve(url, filename)

print('Download complete.')