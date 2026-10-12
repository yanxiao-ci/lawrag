list1 = [0.3, 0.9, 0.2]
list2 = ["A", "B", "C"]

# print(list(zip(list1, list2)))

result = sorted(zip(list1, list2), reverse=True)
print(result)
