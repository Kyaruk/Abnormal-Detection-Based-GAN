

with open("bibtex.txt", 'r', encoding='utf-8') as file:
    save_list = []
    dict = {}
    for line in file:
        line = line.strip('\n')
        save_list.append(line)
    del save_list[0]
    del save_list[-1]
    for line in save_list:
        temp = line.split(' = ')
        index = temp[0]
        if (temp[1][-1] == ','):
            content = temp[1][1:-2]
        else:
            content = temp[1][1:-1]
        dict[index] = content
    if dict['publisher'] == "Association for Computing Machinery":
        dict['publisher'] = "ACM"
print(dict['title'] + "[A]//" + dict['booktitle'] + "[C]. " + dict['location'] +": " + dict['publisher'] + ", " + dict['year'] + ": " + dict['pages'] + ".")