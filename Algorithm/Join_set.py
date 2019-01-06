a = [1,3,4,6,7,8,10]
b = [2,4,5,6,8,9]

def join_set(a,b):
	res = []
	i_a, i_b = 0,0
	while (i_a < len(a) and i_b < len(b)):
		if (a[i_a] == b[i_b]):
			res.append(a[i_a])
			i_a += 1
			i_b += 1
		elif (a[i_a]>b[i_b]):
			i_b += 1
		elif (b[i_b]>a[i_a]):
			i_a += 1
	return res

print(join_set(a,b))