
import json
ENCODING = 'utf-8'
DATABASE = 'data/fever/fever.db'
#import pandas

def detect_multihop(input):
    fin = open(input, 'rb')
    instances = []
    index = 0
    index_list = []
    for line in fin:
        object = json.loads(line.decode(ENCODING).strip('\r\n'))
        #if 'label' in object:
        #    label = ''.join(object['label'].split(' '))
        #else:
        #    label = 'REFUTES'
        evidences = object['evidence']
        store = False
        if len(evidences) > 1:
            store = True
            for e in evidences:
                if len(e) == 1:
                    store = False

        if store:
            print(index)
            #print(evidences)
            index_list.append(index)

        #claim = object['claim']
        #instances.append([index, label, claim, evidences])

        index += 1
    fin.close()
    return index_list

index_list = detect_multihop('data/retrieved/dev.ensembles.s10.jsonl')

#df = pandas.DataFrame(data={"col1": index_list})
#df.to_csv("./file.csv", sep=',',index=False)