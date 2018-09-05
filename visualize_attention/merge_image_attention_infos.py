import os
import json

ATT_FILES_DIR = 'debug_att_dir'
RESULT_FILE = 'merged_att_result.json'

def merge():
	generator = os.walk(ATT_FILES_DIR)
	_, _, files = next(generator)
	result = {}
	for fname in files:
		try:
			suffix_idx = fname.index('.json')
			image_id = fname[:suffix_idx]
			with open(os.path.join(ATT_FILES_DIR, fname), 'r') as f:
				data = json.load(f)
				result[image_id] = data

		except:
			print("ignore file: {}".format(fname))

	with open(RESULT_FILE, 'w') as f:
		json.dump(result, f)


if __name__ == '__main__':
	merge()
