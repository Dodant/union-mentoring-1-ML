import matplotlib.pyplot as plt
from glob import glob
import json 

json_list = glob('./*.json')
json_list.sort()

x = []
train = []
test = []

for file_path in json_list:
    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file)
        x.append(json_data['result'][0]['p'])
        train.append(json_data['result'][0]['trainset_acc'])
        test.append(json_data['result'][0]['testset_acc'])


plt.plot(list(map(str, x)), train, list(map(str, x)), test)
plt.xlabel('Prune Rate')
plt.ylim([0, 100])
plt.legend(['Trainset Acc', 'Testset Acc'])
plt.gcf().set_size_inches(10, 5)
plt.savefig('final_result.png', facecolor='#eeeeee', pad_inches=0.3)