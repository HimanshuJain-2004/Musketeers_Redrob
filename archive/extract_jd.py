import docx

doc = docx.Document('job_description.docx')
text = '\n'.join([p.text for p in doc.paragraphs])
with open('JAIN/prompts/job_description.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print("Extracted JD to JAIN/prompts/job_description.txt")
