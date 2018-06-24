# -*- coding:utf-8 -*-

题目描述
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

# 1,序列最后一位为根，前面有左右子树两部分
# 2,左子树所有小于根，右子树大于根
# 3,移动光标找到右子树起始位
# 4,检查前面左子树小于根
# 5,递归分别检查左右子树序列
class Solution:
    
    def verify(self,seq,start,end):
        
        if start >= end:
            return True
        cursor = end
        while(cursor >= start and seq[cursor-1] > seq[end]):
            cursor -= 1
        for i in range(start,cursor):
            if seq[i] > seq[end]:
                return False
        return self.verify(seq,start,cursor-1) and self.verify(seq,cursor,end-1)
    
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if len(sequence) == 0:
            return False
        else:
            return self.verify(sequence,0,len(sequence)-1)
