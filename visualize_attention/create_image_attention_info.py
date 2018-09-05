import json
import csv
import os
import sys
import numpy as np
import caffe


class AttentionDebugger(object):
	
	def __init__(self):
		self.id2score = self._load_id2score()
		self.id2pre_caption = self._load_img_id2pre_caption()
		self.vocab = self._load_vocab()
		self.precessed_image_dict = {}

		self.project_root = os.path.join(os.path.dirname(__file__), '..')

		self.target_dir = os.path.join(os.path.dirname(__file__), 'debug_att_dir')
		if not os.path.isdir(self.target_dir):
			os.mkdir(self.target_dir)

		self.batch_size = 12
		self.beam_size = 5

	def _translate(self, blob):
		caption = "";
		w = 0;
		while True:
			next_word = self.vocab[int(blob[w])]
			if w == 0:
				next_word = next_word.title()
			if w > 0 and next_word != "." and next_word != ",":
				caption += " ";
			if next_word == "\"" or next_word[0] == '"':
				caption += "\\"; # Escape
			caption += next_word;
			w += 1
			if caption[-1] == '.' or w == len(blob):
				break
		return caption


	def _load_vocab(self):
		vocab_file = os.path.join(self.project_root, 'data/coco_splits/train_vocab.txt')
		assert os.path.isfile(vocab_file), "Error: {} not exist.".format(vocab_file)
		print("loading vocab file: {}".format(vocab_file))
		vocab = []
		with open(vocab_file) as f:
			for word in f:
				vocab.append(word.strip())
		return vocab
 

	def _load_id2score(self):
		images_score_file = os.path.join(self.project_root, 'scores/caption_lstm/scst_iter_1000_scores.json')
		assert os.path.isfile(images_score_file), "Error: {} not exist.".format(images_score_file)
		print("loading score file: {}".format(images_score_file))
		with open(images_score_file, 'r') as f:
			id2score = json.load(f)

		print("scored images number: {}".format(len(id2score)))
		return id2score

	
	def _load_img_id2pre_caption(self):
		caption_file = os.path.join(self.project_root, 'outputs/caption_lstm/scst_iter_1000.json')
		assert os.path.isfile(caption_file), "Error: {} not exist.".format(caption_file)
		print("loading caption file: {}".format(caption_file))
		with open(caption_file, 'r') as f:
			captions = json.load(f)

		print("number : {}".format(len(captions)))

		id2pre_caption = {}
		for item in captions:
			id2pre_caption[item['image_id']] = item['caption']

		print("id2pre_caption len: {}".format(len(id2pre_caption)))
		return id2pre_caption


	def _debug_one_batch(self, net, batch_image_ids):

		for idx in range(self.batch_size):
			debug_result = {}

			# image_id
			image_id = batch_image_ids[idx]
			debug_result['image_id'] = image_id
			print("\n === Start debugging image: {} ===".format(image_id))
			print("You Can search this image id from : http://cocodataset.org/#explore ")

			# caption		
			print("\n--- predicted caption: ---")
			print(json.dumps(self.id2pre_caption[image_id], indent=4))
			debug_result['caption'] = self.id2pre_caption[image_id]

			# score
			print("\n--- scores: ---")
			print(json.dumps(self.id2score[str(image_id)], indent=4))
			debug_result['score'] = self.id2score[str(image_id)]

			# split caption into list
			print("\n--- create every step attention data ---")
			caption = self.id2pre_caption[image_id]
			caption_str = caption
			if caption[:-1] == '.':
				caption = caption[:-1]
			caption = caption.split()
			caption.append('.')
			print(caption)
			print("caption length: {}".format(len(caption)))
			
			# box number
			assert net.blobs['num_boxes'].data.shape[0] == self.batch_size, "num_boxes shape error."
			num_boxes = int(net.blobs['num_boxes'].data[idx][0])
			print("\nbox number: {}".format(num_boxes))
		
			# box info
			assert net.blobs['boxes'].data.shape[0] == self.batch_size, "boxes shape error."
			print("\nboxes info:")
			print(net.blobs['boxes'].data[idx][1:num_boxes + 1])
			box_list = []
			for i in range(num_boxes):
				box_list.append(net.blobs['boxes'].data[idx][1 + i].tolist())
			debug_result['boxes'] = box_list

			print("\nimage id, height, weight:")
			assert net.blobs['image_id'].data.shape[0] == self.batch_size, "image_id shape error."
			print(net.blobs['image_id'].data[idx])

			print("\n--- compute boxes and attention for each step: ---")
			# Which beam we should pay attention to. At first, all beams are same.
			track_flag = [True for i in range(self.beam_size)]

			# each step contains infos:
			# step, attentions, caption 
			_step_infos = []
			for step in range(len(caption)):
				# step 
				step_info = {}
				step_info['step'] = step
				
				# attention values for each box
				attentions_info = None
				blob_name = 'att_weight_{}'.format(step)
				assert net.blobs[blob_name].data.shape[0] == self.batch_size * self.beam_size, "{} shape error.".format(blob_name)
				if step == 0:
					attentions_info = net.blobs[blob_name].data[idx * self.beam_size, :num_boxes].tolist()
				else:
					for i in range(self.beam_size):
						if track_flag[i]:
							attentions_info = net.blobs[blob_name].data[idx * self.beam_size + i, :num_boxes].tolist()
							break
				step_info['attentions'] = attentions_info
	
				# partial caption
				caption_info = ''
				blob_name = 'bs_sentence_{}'.format(step)
				assert net.blobs[blob_name].data.shape[0] == self.batch_size * self.beam_size, "{} shape error.".format(blob_name)

				# clear track_flag
				for i in range(self.beam_size):
					track_flag[i] = False

				for i in range(self.beam_size):
					partial_caption = self._translate(net.blobs[blob_name].data[idx * self.beam_size + i])
					if caption_str.lower().startswith(partial_caption.lower()):
						caption_info = partial_caption
						# set flag to track which attention should be save
						track_flag[i] = True
						break
				step_info['caption'] = caption_info
	
				_step_infos.append(step_info)
	
	
			debug_result['steps'] = _step_infos
	
			target_file_name = "{}.json".format(debug_result['image_id'])
			target_path = os.path.join(self.target_dir, target_file_name)
			print("Saving: {}".format(target_path))
			with open(target_path, 'w') as f:
				json.dump(debug_result, f)

		
	
	def _load_model(self):
		model_file = '/data/orcs/bing/git/Up-Down-Captioner/experiments/test_attention/decoder.prototxt'
		weights_file = '/data/orcs/bing/git/Up-Down-Captioner/snapshots/caption_lstm/lstm_scst_iter_1000.caffemodel.h5'
		caffe.init_log(0, 1)
		caffe.set_device(0)
		caffe.set_mode_gpu()

		net = caffe.Net(model_file, weights_file, caffe.TEST)
		net.layers[0].load_dataset()

		return net
		
	def debug(self):
		self._pause('Make sure you execute this command in shell: \nsource source_me.sh')
		self._pause('Make sure you execute this command in shell: \nexport CUDA_VISIBLE_DEVICES=1')

		net = self._load_model()

		id_to_caption = {}
		iteration = 0

		while True:
			ending = False
			out = net.forward()
			image_ids = net.blobs['image_id'].data
			captions = net.blobs['caption'].data
			
			assert captions.shape[0] == self.beam_size * self.batch_size, "shape is not OK"

			batch_image_ids = []
			for n in range(self.batch_size):
				image_id = str(int(image_ids[n][0]))
				if image_id in id_to_caption:
					ending = True
					break
				else:
					id_to_caption[image_id] = True
				batch_image_ids.append(int(image_id))

			if ending:
				break
			else:
				self._debug_one_batch(net, batch_image_ids)

			print("Iteration: {}".format(iteration))
			iteration += 1


if __name__ == '__main__':
	debugger = AttentionDebugger()
	debugger.debug()
