from PIL import Image, ImageFont, ImageDraw
import numpy as np
import requests

def one_hot_decode(data):
    data = np.squeeze(data)
    list_max=data.tolist()
    list_max.sort(key=lambda x: float(x), reverse = True)
    n = np.argwhere(data==list_max[0])
    return int(n), list_max[0]

def translate_Y(yi , show = False):
    
    num = {
        0 : "fall", 1 : "normal", 2 : "unknow" } 
    
    yi_new = num.get( yi ) 
            
    if show:
        print(yi)
        print(yi_new)            
        
    return yi_new

def lineNotifyMessage():
    # 傳送訊息字串
    message = "Some one fall!!"    
    # 修改成你的Token字串
    token = 'MxdCPSadoPmMOZtydEp8ctLPHcAyqsJ7TH0bBYzTE5a'
    
    headers = { "Authorization": "Bearer " + token,"Content-Type" : "application/x-www-form-urlencoded" }
    payload = {'message': message}
    r = requests.post("https://notify-api.line.me/api/notify", headers = headers, params = payload)
    return r.status_code    

