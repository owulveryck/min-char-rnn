aa=11
b = 0
for a, b, c in zip([aa,12,13,14,15],
                   [21,22,23,24,25],
                   [31,32,33,34,35]):
  a += 1
  print "a: %s" % a
  print "b: %s" % b
  print "c: %s" % c

print aa
