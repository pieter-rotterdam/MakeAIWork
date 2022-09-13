benelux = {"Belgium", "The Netherlands", "Luxembourg"}

my_list = benelux {}
my_order = [3, 1, 2]
my_list = [my_list[i] for i in my_order]

print (my_list)