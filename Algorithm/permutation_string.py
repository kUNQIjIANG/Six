链接：https://www.nowcoder.com/questionTerminal/fe6b651b66ae47d7acce78ffdd9a96c7?toCommentId=1451709
来源：牛客网

输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。 
输入描述:
输入一个字符串,长度不超过9(可能有字符重复),字符只包括大小写字母


## total number of permutation of string with repitition 
## swap first char with chars later which is not same,
## every swap add all permutation starting from the second char
## <recursion>

class Solution:
    def __init__(self):
        self.res = []
        
    def permutate(self,s,begin):
        if begin == len(s) - 1:
            self.res.append(s)
        else:
            for i in range(begin,len(s)):
                if begin != i and s[begin] == s[i]:
                    continue
                swap_s = self.swap(s,begin,i)
                self.permutate(swap_s,begin+1)
                
    def Permutation(self, ss):
        # write code here
        if len(ss) != 0:
            self.permutate(ss,0)
        return sorted(self.res)
    
    def swap(self,c, i, j):
        c = list(c)
        c[i], c[j] = c[j], c[i]
        return ''.join(c)
