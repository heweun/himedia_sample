from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")

classifier = pipeline("ner", model=model, tokenizer=tokenizer)

result = classifier("Alya told Jasmine that Andrew could pay with cash..")
print(result)