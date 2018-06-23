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
