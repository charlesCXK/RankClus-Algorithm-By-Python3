# --*-- encoding: utf-8 --*--

import sys
from lxml import etree
from lxml.etree import XMLSyntaxError

xmlfile = 'dblp.xml'
dtd = etree.DTD(file='dblp.dtd')

records = list()
for event, element in etree.iterparse(xmlfile, load_dtd=True):
  if element.tag == 'article':
  	try:
  		print(element.find('year').text)

  		break
  		year = int(element.find('year').text)
  		if 2010 < year <= 2018:
  			record = dict()
  			record['year'] = year
  			record['authors'] = [author.text for author in element.findall("author")]
  			record['journal'] = element.find('journal').text
  			records.append(record)
  	except AttributeError:
  		continue