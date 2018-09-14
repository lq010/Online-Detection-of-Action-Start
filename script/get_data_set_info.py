import os

val_annotations_path = os.path.join(os.pardir,'data/TH14_Temporal_annotations_validation/annotation')

files = os.listdir(val_annotations_path)
files.remove('Ambiguous_val.txt')

videos = dict()
actions = dict()
files.sort()
for actionFile in files:
    filepath = os.path.join(val_annotations_path, actionFile)
    #print('Reading file: %s from %s'%(actionFile, val_annotations_path))
    file = open(filepath, 'r')
    actions_one_class = file.readlines() # temporal locations of instances of one class
    
    for line in actions_one_class:
        items = line.strip().split()
        assert len(items) == 3
        vid = items[0]  # video name
        if vid not in videos.keys():
            videos[vid] = {}
        action_count = videos[vid].get(actionFile, 0)
        videos[vid][actionFile] = action_count+1
        if actionFile not in actions:
            actions[actionFile] = {}
        actions[actionFile][vid] = 1
    #print(actionFile +" - "+str(len(actions[actionFile])))
    
    file.close()
print('######################')
print('# actions: ' +str(len(actions)))
print('------------------------')
print ('fileNAme'.ljust(30) + '#videos')
temp = 0
for action in actions:
    print(action.ljust(30)+ str(len(actions[action])))
    temp += len(actions[action])
print('------------------------')
print('SUM of videos: ' +str(temp))

print('------------------------')
print('real num of videos: '+ str(len(videos)))

print('------------------------')
print('video name'.ljust(30) + '# action classes')
for key, value in videos.items():
    print(key.ljust(30) + str(len(value)) +' '+ str(value))