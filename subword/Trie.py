# coding:utf-8
import shutil

class Trie:
    def __init__(self):
        self.dic = {}  # 定义一种基于字典的树结构{key:{val:0,...}}},val代表是否具有此单词,{}里面是子字典元素集合

    def insert(self, word: str) -> None:
        root = self.dic
        for i in word:
            if i not in root:
                root[i] = {}
            root = root[i]
        root["end"] = True

    def search(self, word:str) -> bool:
        root = self.dic
        for i in word:
            if i not in root:
                return False
            else:
                root = root[i]
        return 'end' in root

    def startsWith(self, prefix: str):
        '''

        :param prefix: word
        :return: 单词的前缀，如果是后缀需要逆序处理
        '''
        self.z = 0
        root = self.dic
        flag = -1
        for index in range(len(prefix)):
            if prefix[index] in root:
                if 'end' in root[prefix[index]]:
                    flag = index
                root = root[prefix[index]]
            else:
                break
        self.z = flag
        if flag == -1:
            # print(None)
            return None
        else:
            return prefix[0:flag + 1]
