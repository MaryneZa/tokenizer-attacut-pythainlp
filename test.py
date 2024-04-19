from attacut.models import Attacut

attacut = Attacut.from_checkpoint("attacut/statics/model.pth")
text1 = "ทดสอบภาษาอังกฤษ"
print(attacut.inferrence(text1))
