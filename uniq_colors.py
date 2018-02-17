import jsonlines
import json


with jsonlines.open('/Users/jdo/dev/scrapers/scraper10/scraper10/spiders/items.jsonl', 'r') as infile:
    uniq_set = set()
    feature_dict = {}
    for item in infile:
        # print(item)
        img_hash = item['image_hash'][0]
        img_features = item['color1']
        uniq_set.add(img_hash)
        feature_dict[img_hash] = img_features

    uniq_feature_list = []
    with jsonlines.open('uniq_items1.jsonl', 'w') as outfile:
        hash_list = list(uniq_set)
        for img_hash in hash_list:
            features = feature_dict[img_hash]
            feat_obj = {'img_hash': img_hash, 'img_features': features}
            # write_dict = json.dumps(feat_obj)
            print('unique features written for: ', img_hash)
            outfile.write(feat_obj)
