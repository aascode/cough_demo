with open("data/result1.txt", "r") as f:
    data = f.readlines()

result = []
for line in data:
    utt, acc = line.split()
    acc = float(acc)
    if acc >= 0.9:
        result.append("{}".format(utt.split("/")[-1]))

with open("data/file_filter1.txt", "w") as f:
    f.write("\n".join(result))