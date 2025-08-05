import spacy

#not performing very well

nlp = spacy.load("en_core_web_sm")

# Sample text
text = """xX @ EduPlusCampus ra n

learner.vierp.in

Bansilal Ramnath Agarwal Charitable Trust's

AY; Vishwakarma Institute of
a) Technology

(An Autonomous Institute affiliated to Savitribai Phule Pune University)
666, Upper Indiranagar, Bibwewadi, Pune- 411 037.
Website: www.vit.edu

12110729
DAVARI VAISHNAVI RAMCHANDRA
BACHELOR OF TECHNOLOGY

or

DIRECTOR

Permanent Address: Vitthaldeo housing society
malkapur karad Karad, Karad, PIN: 415539

Mobile No:7350283973

Branch: E&TC

Birth Date: 28/09/ 2003

Blood Group: O Positive

Admission Year: 2021-22

Emergency Contact Person: Ramchandra davari
Emeraencv.Contact.Person.Number:.9766436216

= O <"""

# Process the text
doc = nlp(text)

student_name = []
college_name = []

for ent in doc.ents:
    if ent.label_ == "PERSON":
        student_name.append(ent.text)
    elif ent.label_ in ["ORG", "GPE", "FAC"]:  # ORG usually covers college/university names
        college_name.append(ent.text)

print("Student Name(s):", student_name)
print("College Name(s):", college_name)
