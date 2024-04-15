from attacut import Attacut

text1 = "ร้อนจัด อยากตากลม"
text2 = "มันเย็นสบาย"
text3 = "ปอปลาตากลม"
attacut = Attacut()
print(attacut.from_checkpoint(text1))
print(attacut.from_checkpoint(text2))
print(attacut.from_checkpoint(text3))

