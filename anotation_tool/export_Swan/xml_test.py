## test xml


import xml.etree.ElementTree as ET

#xmlデータを読み込みます
tree = ET.parse('datest2_debates_anna@swan.de.xml')
#一番上の階層の要素を取り出します
root = tree.getroot()


for child_1 in root:
    if child_1.tag == "annotations":
        for child_2 in child_1:
            print(child_2.tag)
            print(child_2.attrib)
            print("ID : {}\tStart : {}\tEnd : {}".format(child_2.find("id").text, child_2.find("start").text, child_2.find("end").text))
    else:
        for child_3 in child_1:
            print(child_3.tag)
            print(child_3.attrib)
            print("From : {}\tTo : {}\tLabel : {}".format(child_3.find("from").text, child_3.find("to").text, child_3.find("labels").text))

print("-----")
# print(child.attrib["name"],child.find("rank").text)
#print(root[0][0].text)


for id in root.iter('id'):
    print(id.text)

for links in root.iter("links"):
    for lb in links.iter("label"):
        print(lb.text)
