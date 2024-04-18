from attacut import Attacut

text1 = "ร้อนมาก อยากตากลม"
text2 = "อากาศร้อน ส่งผลให้มนุษย์อวกาศไม่มีไอศกรีมรับประทาน"

# attacut = Attacut()
# attacut.from_checkpoint("attacut/models/attacut_sc/model.pth")
attacut = Attacut.from_checkpoint("attacut/models/attacut_sc/model.pth")
print(attacut.tokenizer(text1))
print(attacut.tokenizer(text2))




