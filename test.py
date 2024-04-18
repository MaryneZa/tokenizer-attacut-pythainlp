from attacut.models import Attacut

text1 ="ทดสอบภาษาอังกฤษ"
text2 = "อากาศร้อน ส่งผลให้มนุษย์อวกาศไม่มีไอศกรีมรับประทาน"

attacut = Attacut.from_checkpoint("attacut/statics/model.pth")
print(attacut.inferrence(text1))
print(attacut.inferrence(text2))




