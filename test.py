from attacut import Attacut

text1 = "ร้อนจัด อยากตากลม"

attacut = Attacut()
attacut.from_checkpoint("attacut/artifacts/attacut-sc/model.pth")
print(attacut.tokenizer(text1))


