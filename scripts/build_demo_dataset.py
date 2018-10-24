import json

def build_demo_dataset(source_file, target_file, item_num):
    captioning = json.load(open(source_file, 'r'))
    target_keys = captioning.keys()[:item_num]
    target = {key:captioning[key] for key in target_keys}
    json.dump(target, open(target_file, 'w'))

if __name__ == '__main__':
    build_demo_dataset('./data/densecap/train.json', './data/demo_densecap/train.json', 120)
    build_demo_dataset('./data/densecap/val_1.json', './data/demo_densecap/val_1.json', 60)
    build_demo_dataset('./data/densecap/val_2.json', './data/demo_densecap/val_2.json', 60)

