import pandas as pd
import json

from collections import defaultdict
pd.options.display.float_format = '{:,.3g}'.format

def get_rows_from_json(result_dict):
	metrics = ['IS', 'IS_std', 'KID', 'KID_std', 'LPIPS', 'LPIPS_std']
	rows = []
	for k, v in result_dict.items():
		row = []
		for m in metrics:
			row.append(v['EWC'][m])
			row.append(v['No EWC'][m])
		rows.append(row)
	return rows

dataset_short = ['Bald', 'Eyeglasses', 'Bangs', 'Obama', 'Cat']
metrics = ['IS', 'IS_std', 'KID', 'KID_std', 'LPIPS', 'LPIPS_std']
ewc_pair = ['EWC', 'No EWC']
### dataset experiment
experiment_name = 'dataset'
outpath = f'results/{experiment_name}_experiment_results.json'
with open(outpath, 'r') as f:
	result_dict = json.load(f)

rows = get_rows_from_json(result_dict)
midx = pd.MultiIndex.from_product([metrics , ewc_pair])
df = pd.DataFrame(rows, columns=midx, index = dataset_short)
print(df[['KID', 'LPIPS']].to_latex(multirow=True))

### nshot experiment
experiment_name = 'n_shot'
outpath = f'results/{experiment_name}_experiment_results.json'
with open(outpath, 'r') as f:
	result_dict = json.load(f)

rows = get_rows_from_json(result_dict)

n_shot_runs = [100, 50, 10, 3]
midx = pd.MultiIndex.from_product([metrics , ewc_pair])
df = pd.DataFrame(rows, columns=midx, index = n_shot_runs)
print(df[['KID', 'LPIPS']].to_latex(multirow=True))

### method experiment
experiment_name = 'method_compare'
outpath = f'results/{experiment_name}_experiment_results.json'
with open(outpath, 'r') as f:
	result_dict = json.load(f)

rows = get_rows_from_json(result_dict)

method_runs = ['LS', 'IN', 'LS+IN', 'D_EWC']
midx = pd.MultiIndex.from_product([metrics , ewc_pair])
df = pd.DataFrame(rows, columns=midx, index = method_runs)
print(df[['KID', 'LPIPS']].to_latex(multirow=True))
