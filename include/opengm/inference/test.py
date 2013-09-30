order=3
nAssigments = 2**3
print "nAssigments",nAssigments

for a in range(nAssigments):
	print "		asssigment",a

	for v in range(order):
		print "			v",v

		if( a & (1 << v)):
			print "alpha"
		else :
			print "not alpha"