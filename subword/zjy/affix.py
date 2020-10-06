import re
f = open("C:/Users/张金元/Desktop/a.txt", "r", encoding='utf-8')     #打开test.txt文件，以只读得方式，注意编码格式，含中文
data = f.readlines()                            #循环文本中得每一行，得到得是一个列表的格式<class 'list'>
f.close()                                       #关闭test.txt文件
for line in data:
    result = re.findall('-(\w*\w)\s+',line)     #使用正则表达式筛选每一行的数据,自行查找正则表达式
    result1 = list(set(result))
    result1.sort(key=result.index)
    print("res1:%s" % result1)
for i in result1:
    f1 = open("C:/Users/张金元/Desktop/c.txt", "a+", encoding='utf-8')
    f1.write(i + '\n')  # 将每一行打印进test1.txt文件并换行
    f1.close()  # 关闭test1.txt文件**
    print(i)


