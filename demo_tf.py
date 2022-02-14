import random
import numpy as np
import time
start_time = time.time()

def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 += str(ele)  
    
    # return string  
    return str1 

b=[]
buffer = range(-10,10)
for i in np.linspace(10,100,300):
    a = i + random.choice(buffer)
    if a != 0:
        print(1/(a))
        val = 1/(a)
        b.append(val)
        print("cnvrg_linechart_{} value: '{}'".format('loss', val))
        #time.sleep(1)
print("cnvrg_tag_train_runtime" , (time.time() - start_time))        
print('cnvrg_tag_accuracy', random.choice(np.arange(93,98,0.1)))
str_out = listToString(b)
file = open('weights.h5', 'w')
file.write(str_out)
file.close()
        


