import easyocr
reader = easyocr.Reader(['en', 'ko'], gpu=False) # this needs to run only once to load the model into memory
result = reader.readtext('test_image2.png', detail=0, paragraph=True)
print(result)