from urllib.request import urlopen
from lxml import etree


##########NASZE ZAPYTANIE
ID = '???' # id sekwencji
baseurl = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
query = "db=nucleotide&id="+ID+"&format=xml" 
url = baseurl+query
########################

#########OTWORZ URL I PRZECZYTAJ XML
f = urlopen(url) 
resultxml = f.read() 
xml = etree.XML(resultxml) 
###################################

resultelements = xml.xpath("//GBSeq_sequence") #sekwencja

for element in resultelements:
    print(element.text)