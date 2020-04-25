import os
import re
from os.path import join as pjoin

def read_SDN():
	cnt = 0
	not_save_cnt = 0
	sents = []
	for f in os.listdir("/data/lxk/DependencyParsing/SDN"):
		with open(pjoin("/data/lxk/DependencyParsing/SDN", f), "r", encoding="utf-8", errors="ignore") as fin:
			line = fin.readline().strip()
			while line:
				print(line)
				if line == "<CONLL>":
					sent = []
					while line:
						if not line or not line == "</CONLL>":
							save = True
							head_cnt = 0
							for i in sent:
								l = i.split("\t")
								if l[6] == "-1":
									save = False
								elif l[6] == "0":
									head_cnt += 1
							if save and head_cnt == 1:
								sents += [sent]
								cnt += 1
							else:
								not_save_cnt += 1
							sent = []
						else:
							sent += [line]
						line = fin.readline().strip()
				line = fin.readline().strip()
	print("Done loading {} sents, with failed {} sents.".format(cnt, not_save_cnt))
	return sents

def write_file(filename, samples, id_num):
	start = id_num
	with open(filename, "w", encoding="utf-8", errors="ignore") as fout:
		for sent in samples:
			fout.write("# sent_id = {}\n".format(id_num))
			for i in sent:
				fout.write(i + "\n")
			id_num += 1
	print("Done dumping {} data to {}.".format(id_num - start, filename))
	return id_num

def preprocess():
	sents = read_SDN()
	num = len(sents)
	train_samples = sents[: int(float(num) * 0.8)]
	dev_samples = sents[int(float(num) * 0.8): int(float(num) * 0.9)]
	test_samples = sents[int(float(num) * 0.9):]

	cnt = 1
	cnt = write_file("/data/lxk/DependencyParsing/train.txt", train_samples, cnt)
	cnt = write_file("/data/lxk/DependencyParsing/dev.txt", dev_samples, cnt)
	cnt = write_file("/data/lxk/DependencyParsing/test.txt", test_samples, cnt)
	

if __name__ == "__main__":
	preprocess()