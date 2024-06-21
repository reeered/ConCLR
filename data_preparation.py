import os
import random
import lmdb

def parse_gt_file(gt_path):
    with open(gt_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            parts = line.strip().split(',')
            if len(parts) > 8:
                return parts[8]
            else:
                raise Exception("Wrong format")

def extract_labels(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    label_file = os.path.join(output_dir, 'labels.txt')
    with open(label_file, 'w', encoding='utf8') as f:
        for img in os.listdir(data_dir):
            if img.endswith('.jpg'):
                gt_path = os.path.join(data_dir, 'gt_' + img.replace('.jpg', '.txt'))
                text = parse_gt_file(gt_path)
                f.write(f"{img}\t{text}\n")
    print(f"Labels extracted to {label_file}")

def split_data(data_dir, output_dir, train_ratio=0.8, max_length=None):
    images = [img for img in os.listdir(data_dir) if img.endswith('.jpg')]
    random.shuffle(images)
    train_size = int(len(images) * train_ratio)
    train_images = images[:train_size]
    test_images = images[train_size:]

    n_train = 0
    n_test = 0

    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for img in train_images:
            gt_path = os.path.join(data_dir, 'gt_' + img.replace('.jpg', '.txt'))
            text = parse_gt_file(gt_path)
            if max_length and len(text) > max_length:
                continue
            n_train += 1
            f.write(f"{img}\t{text}\n")

    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        for img in test_images:
            gt_path = os.path.join(data_dir, 'gt_' + img.replace('.jpg', '.txt'))
            text = parse_gt_file(gt_path)
            if max_length and len(text) > max_length:
                continue
            n_test += 1
            f.write(f"{img}\t{text}\n")
    print(f"Data split: {len(train_images)} training and {len(test_images)} testing")
    print(f"After filtering with max_length: {n_train} training and {n_test} testing")

def save_to_lmdb(data_dir, output_dir, split='train'):
    lmdb_path = os.path.join(output_dir, f'{split}.lmdb')
    os.makedirs(output_dir, exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=1099511627776)  # 1TB

    with env.begin(write=True) as txn:
        with open(os.path.join(data_dir, f'{split}.txt'), 'r') as f:
            for line in f:
                img_name = line.strip()
                img_path = os.path.join(data_dir, img_name)
                gt_path = os.path.join(data_dir, 'gt_' + img_name.replace('.jpg', '.txt'))

                text = parse_gt_file(gt_path)

                with open(img_path, 'rb') as img_file:
                    img_data = img_file.read()

                txn.put(img_name.encode('utf8'), img_data)
                txn.put(f"{img_name}_label".encode('utf8'), text.encode('utf8'))
    print(f"Data saved to {lmdb_path}")

if __name__ == '__main__':
    data_dir = './data/ICDAR2013+2015/train_data'
    output_dir = './data/ICDAR2013+2015'
    extract_labels(data_dir, output_dir)
    split_data(data_dir, output_dir, max_length=13)
    # save_to_lmdb(args.data_dir, args.output_dir, split='train')
    # save_to_lmdb(args.data_dir, args.output_dir, split='test')