# there is only one memory address of dic 
# where get updated in recursions

def fast_fibbo(n,dic):
    if n not in dic.keys():
        dic[n] = fast_fibbo(n-1,dic) + fast_fibbo(n-2,dic)
    return dic[n]
    
fast_fibbo(100,{0:0,1:1})
