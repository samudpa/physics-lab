import json
import string

# Download a copy of Divina Commedia from Wikisource:
# https://ws-export.wmcloud.org/?lang=it&title=Divina_Commedia

SOURCE_PATH = "data/Divina_Commedia.txt"
OUTPUT_PATH = "data/Divina_Commedia.json"

# initialize data dictionary
divina_commedia = {}
cantiche = ["Inferno", "Purgatorio", "Paradiso"]
for cantica in cantiche:
    divina_commedia[cantica] = {}
lengths = [34, 33, 33]  # number of cantos in each cantica

canto = None
cantica = -1

# open source file, and iterate over each line
f = open(SOURCE_PATH, "r", encoding="utf-8")

for line in f:

    # strip newline character
    line = line.strip("\n")

    # "Altri progetti" serves as a separator between cantos
    if line == "Altri progetti":

        # save verses into main dictionary
        if canto != 0 and canto != None:
            divina_commedia[cantiche[cantica]][canto] = verses

        # reached end of cantica
        if canto == None or canto == lengths[cantica]:
            canto = 0
            cantica += 1
            if cantica >= len(cantiche):
                break

        else:

            # increase count
            canto += 1
            verses = []
            skipped_summary = False

    elif line != "":

        if canto != 0 and canto != None:

            # append line to list of verses, but only
            # if summary has already been skipped
            if skipped_summary:

                # strip digits at the end of the verse
                #   from https://stackoverflow.com/a/40691501
                line = line.rstrip(string.digits)
                verses.append(line)

            else:
                skipped_summary = True

# close source file
f.close()

# dump data into .json file
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(json.dumps(divina_commedia))
