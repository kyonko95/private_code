import json

def main():
	#f = open('/home/pcd002/yoshida/pytorch-tutorial/tutorials/03-advanced/image_captioning/data/mscoco_char1_caption.json','r')
	f = open('/home/pcd002/two_stream_cnn/data/annotations/day1-3_rgb.json','r')
	json_data = json.load(f)

	print("{}".format(json.dumps(json_data,indent=4)))
	print(type(json_data))

if __name__=='__main__':
	main()