import os
import requests

file = open("./multimedia.txt", "r")
os.makedirs("./images", exist_ok=True)

image_count = int(input("Enter number of images to download: "))

id = 0
line = file.readline()
for count in range(image_count):
    line = file.readline()
    if line == "":
        break
    col = 0
    start = 0
    end = 0
    link = ""

    for i in range(len(line)):
        if line[i] == '\t':
            col += 1
        
            if col == 3:
                start = i+1
            elif col == 4:
                end = i
                break
    
    link = line[start : end]
    ext_pos = 0
    for i in range(len(link) - 1, 0, -1):
        if link[i] == '.':
            ext_pos = i
            break
    ext = link[ext_pos:]

    data = requests.get(link).content
    image = open(f"./images/{id}{ext}", "wb")

    image.write(data)
    id += 1