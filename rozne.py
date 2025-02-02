from urllib.request import urlopen
from urllib.parse import urlencode 
from lxml import etree  

term = "???"
base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
url = base_url+ urlencode({"db": "pubmed", "term": term, "retmax": 30}) 

response = urlopen(url) #otwieramy polączenie URL
resultxml = response.read() #czytam zawartosc
xml = etree.XML(resultxml)  #strukturyzacja

resultelements = xml.xpath("//Id")  #czego chce sie dowiedziec? Tutaj ID prac

IDs = []
for elem in resultelements:
    IDs.append(elem.text)

print(IDs)

for i in IDs:
    baseurl = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
    query = "db=pubmed&id="+i+"&format=xml"    
    url = baseurl+query

    f = urlopen(url) #otwieram połączenie URL
    resultxml = f.read() #czytam zawartość
    xml = etree.XML(resultxml) #strukturyzacja

    resultelements= xml.xpath("//ArticleTitle")   #Inne: ArticleTitle, LastName, Keyword, AbstractText, DescriptorName

    for element in resultelements:
        print(element.text)
