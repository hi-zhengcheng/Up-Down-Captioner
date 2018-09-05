import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import json
import argparse
import os

def show_step_attentions(image_path, attention_info, top_n, result_dir):
	# Avoid creating same file repeatedly
	image_name = os.path.basename(image_path)
	_name, file_extensiion = os.path.splitext(image_name)
	new_name = "{}.png".format(_name)
	new_path = os.path.join(result_dir, new_name)
	if os.path.isfile(new_path):
		return

	img = mpimg.imread(image_path)
	fig = plt.figure()

	# Figure title
	fig_title = 'Blue 1~4: *{:.2f}  *{:.2f}  *{:.2f}  *{:.2f}'.format(
		attention_info['score']['Bleu_1'],
		attention_info['score']['Bleu_2'],
		attention_info['score']['Bleu_3'],
		attention_info['score']['Bleu_4'])
	fig.suptitle(fig_title)
	
	step_num = len(attention_info['steps'])
	box_num = len(attention_info['boxes'])
	col_num = 4
	row_num = (step_num + col_num - 1) / col_num
	boxes = attention_info['boxes']

	# Attention box colors ordered by prob.
	colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

	for step_idx in range(step_num):
		row_idx = step_idx / col_num
		col_idx = step_idx % col_num

		axes = fig.add_subplot(row_num, col_num, row_idx * col_num + col_idx + 1)
		axes.set_xticks([])
		axes.set_yticks([])
		axes.imshow(img)

		step_info = attention_info['steps'][step_idx]
		atts = step_info['attentions']

		top_n_idxes = np.argsort(-np.array(atts))[:top_n]

		title = step_info['caption'].split()[-1] if step_info['caption'] else ""
		title = '.' if title.endswith('.') else title
		axes.set_title(title, 
			fontdict={'fontsize': 12},
			loc='center')
		
		for i, att_idx in enumerate(top_n_idxes):
			# draw attention rect
			left = int(boxes[att_idx][0])
			top = int(boxes[att_idx][1])
			width = int(boxes[att_idx][2]) - left
			height = int(boxes[att_idx][3]) - top
			color = colors[i] if i < len(colors) else colors[-1]
			rect = patches.Rectangle((left, top), width, height, linewidth=1, edgecolor=color, facecolor='None')
			axes.add_patch(rect)

			# draw attention value text
			text_pad = 1
			axes.text(left + text_pad + 1, top - text_pad - 1, 
				str(int(atts[att_idx] * 100)), 
				color='b', 
				fontsize=8, 
				bbox=dict(pad=1, edgecolor='None', facecolor='white', alpha=.5))

	plt.savefig(new_path, dpi=600)
	plt.close()


def create_attention_images(id2info_json_path, image_dir, top_n, result_dir):
	"""Show attention infos in every caption step.

	Args:
		id2info_json_path: json file containing image id to attention info.
		image_dir: image dir.
		top_n: draw top n attention bbox.
		result_dir: save attentioned png image into this dir.

	Returns:
		Create images with attention rectangles in result_dir.
	"""
	assert os.path.isfile(id2info_json_path), "Error! file not exist: {}".format(id2info_json_path)

	with open(id2info_json_path, 'r') as f:
		data = json.load(f)

	count = 0
	for image_id, attention_info in data.items():
		count += 1

		image_path = None
		for split in ['train', 'val']:
			tmp_name = "COCO_{}2014_{:012}.jpg".format(split, int(image_id))
			test_path = os.path.join(image_dir, "{}2014/{}".format(split, tmp_name))
			if os.path.isfile(test_path):
				image_path = test_path
				break
	
		if image_path is None:
			print("{:5d}/{} : Image file with id {} not found.".format(count, len(data), image_id))
			continue
	
		print("{:5d}/{} : {}".format(count, len(data), test_path))
		show_step_attentions(image_path, attention_info, top_n, result_dir)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--id2info_json', default='merged_att_result.json.new', help='json file containing image id to attention info.')
	parser.add_argument('--image_dir', default='.', help='image dir')
	parser.add_argument('--top_n', type=int, default=3, help='top n attention areas')
	parser.add_argument('--result_dir', required=True, help='dir to save attention images')
	args = parser.parse_args()

	create_attention_images(
		args.id2info_json, 
		args.image_dir, 
		args.top_n, 
		args.result_dir)
	
