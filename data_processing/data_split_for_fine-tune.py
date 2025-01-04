import json
import pandas as pd

dataset_name = 'Books'  # TripAdvisor
if dataset_name=='Books':
    rec_item_path = f'data/data_sort_by_time/{dataset_name}/rec_item_{dataset_name}.json'
    data_for_ft_path = f'data/data_for_fine-tune/{dataset_name}/fine-tune.json'
    save_path = f'data/data_for_fine-tune/{dataset_name}/data_wo_ft.json'

    with open(rec_item_path, 'r') as f:
        rec_items = json.load(f)

    with open(data_for_ft_path, 'r') as f:
        fine_tune = json.load(f)

    result = {}
    uids=list(fine_tune.keys())
    for item in rec_items.items():
        uid = item[0]
        if uid in uids:
            continue
        result[item[0]] = item[1]

    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4)
else:
    rec_item_path = f'data/data_sort_by_time/{dataset_name}/rec_item_{dataset_name}.json'
    data_for_ft_path = f'data/data_for_fine-tune/{dataset_name}/fine-tune.xlsx'
    save_path = f'data/data_for_fine-tune/{dataset_name}/data_wo_ft.json'

    with open(rec_item_path, 'r') as f:
        rec_items = json.load(f)

    df = pd.read_excel(data_for_ft_path)
    uids = df['uid'].tolist()

    result = {}
    for item in rec_items.items():
        uid = item[0]
        if uid in uids:
            continue
        result[item[0]] = item[1]

    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4)