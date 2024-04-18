from attacut import Attacut

text1 = "ร้อนจัด อยากตากลม"
text2 = "แมร้อน"

# attacut = Attacut()
# attacut.from_checkpoint("attacut/models/attacut_sc/model.pth")

attacut = Attacut.from_checkpoint("attacut/models/attacut_sc/model.pth")
print(attacut.tokenizer(text1))
print(attacut.tokenizer(text2))




