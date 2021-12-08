import pandas as pd
import json

from collections import defaultdict
pd.options.display.float_format = '{:,.3g}'.format

def get_rows_from_json(result_dicts, metrics = ['IS', 'IS_std', 'KID', 'KID_std', 'LPIPS', 'LPIPS_std']):
	rows = []
	if isinstance(result_dicts, dict):
		for k, v in result_dict.items():
			row = []
			for m in metrics:
				row.append(v['EWC'][m])
				row.append(v['No EWC'][m])
			rows.append(row)
	else:
		for k, v in result_dicts[0].items():
			row = []
			for d in result_dicts:
				for m in metrics:
					row.append(d[k]['EWC'][m])
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
isinstance(result_dict,list)
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
## for poster (bald only, w no EWC)
experiment_name = 'method_compare'
outpath = f'results/{experiment_name}_Bald_experiment_results.json'
with open(outpath, 'r') as f:
	result_dict = json.load(f)

rows = get_rows_from_json(result_dict)

method_runs = ['LS', 'IN', 'LS+IN', 'D_EWC']
midx = pd.MultiIndex.from_product([metrics , ewc_pair])
df = pd.DataFrame(rows, columns=midx, index = method_runs)
print(df[['KID', 'LPIPS']].to_latex(multirow=True))

## for paper (all 3 subsets with EWC only)
experiment_name = 'method_compare'
datasets = ['Eyeglasses','Bangs']

result_dict_list = []
for dset in datasets:
	with open(f'results/{experiment_name}_{dset}_experiment_results.json', 'r') as f:
		result_dict_list.append(json.load(f))

rows = get_rows_from_json(result_dict_list, metrics = ['KID', 'LPIPS'])
midx = pd.MultiIndex.from_product([datasets,['KID', 'LPIPS']])
df = pd.DataFrame(rows, columns=midx, index = method_runs)
print(df.to_latex(multirow=True))
