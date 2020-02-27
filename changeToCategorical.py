"""A simple file to change the input data for ANNs so that each categorical variable
        now has one column for each possible value"""


# we assume that categorical features are contiguous and start at index 1
# note that the values in SEX, RACE, and other such dicts must be contiguous integers starting with 1
ID = 'person_id'
FEATURES_TO_USE = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'is_recid', 'is_violent_recid\n']
SEX = {"Male":1, "Female":2}
RACE = {"African-American":3, "Other":4, "Caucasian":5, "Hispanic":6, "Native American":7, "Asian":8}
CATEGORICAL = {'sex':SEX, 'race':RACE}
FIRST_NUMERIC = 3
LAST_CATEGORICAL = 8
PERSON_LENGTH = 16
READ_FILE = "ANAMergedTestFeatures.csv"
WRITE_FILE = "Test.csv"
NEW_FEATS = ['person_id', 'Male', 'Female', "African-American", "Other", "Caucasian", "Hispanic", "Native American",
             "Asian", 'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
             'is_recid', 'is_violent_recid\n']


features = []
all_people_labels = []
with open(READ_FILE) as file:
    i = 0
    for line in file:
        person = [0]*PERSON_LENGTH
        split_line = line.split(",")
        if i == 0:
            features = split_line
            i += 1
        else:
            for j, feat in enumerate(split_line):
                if features[j] in CATEGORICAL:
                    person[CATEGORICAL[features[j]][feat]] = 1
                elif features[j] in FEATURES_TO_USE:
                    if "\n" in feat:
                        feat = feat[:-1]
                    person[FEATURES_TO_USE.index(features[j]) + LAST_CATEGORICAL + 1] = int(float(feat))
                elif features[j] == ID:
                    person[0] = int(float(feat))
            all_people_labels.append(person)

string = ''
for feat in NEW_FEATS:
    string += feat
    string += ','
string = string[:-1]

#print(string)


for person in all_people_labels:
    for feat in person:
        string += str(feat)
        string += ','
    string = string[:-1]
    string += "\n"
#print(string)

with open(WRITE_FILE, 'w') as file:
    file.write(string)
