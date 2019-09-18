with open('cites.wiki') as f, open('cites.ga', 'w') as f1:
	for line in f:
		line = line.strip().split('\t')
		print(line[0])
		f1.write('\t'.join((line[0], 'ldcOnt:GPE.UrbanArea.City')) + '\n')