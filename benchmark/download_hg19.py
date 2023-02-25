import urllib.request

url = 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz'
filename = 'hg19.fa.gz'

# Download the file from the URL
urllib.request.urlretrieve(url, filename)

print('Download complete.')