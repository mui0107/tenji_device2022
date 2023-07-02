from itertools import zip_longest
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import pygame
import time

camera_id = 0
delay = 1
window_name = 'frame'
#リスト関係
a=0
b=0
#情報を分ける用
inf=0
#座標計算用
a0=float()
b0=float()
a1=float()
b1=float()
#傾き、切片格納リスト
ab_list=[]
ab1_list=[]

first_list=[]
second_list=[]
third_list=[]
last_list=[]
#列のリスト
a_list=[]
b_list=[]
c_list=[]
d_list=[]
e_list=[]

cap = cv2.VideoCapture(camera_id)
img2 = np.ones((640,480,3)) * 255 

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

if not cap.isOpened():
    print("映像が取得できません。")
 
    import sys
    sys.exit()
def sound1():
    pygame.mixer.init() #初期化
 
    pygame.mixer.music.load("警告音1.mp3") #読み込み
 
    pygame.mixer.music.play(1) #再生
 
    time.sleep(1)

def sound2():
    pygame.mixer.init() #初期化
 
    pygame.mixer.music.load("女性_図書館右.mp3") #読み込み
 
    pygame.mixer.music.play(1) #再生
 
    time.sleep(8)

def sound3():
    pygame.mixer.init() #初期化
 
    pygame.mixer.music.load("女性_図書館左.mp3") #読み込み
 
    pygame.mixer.music.play(1) #再生
 
    time.sleep(8)

def getXY(xy,XY):
    return (xy+XY)/2

def getInf(X,r,c):
    if X[1]>ab0_list[r-1]:
        if X[0]>=ab1_list[c-1]:
            return 0#右下
        if X[0]<ab1_list[c-1]:
            return 1#左下
    if X[1]<=ab0_list[r-1]:
        if X[0]>=ab1_list[c-1]:
            return 2#右上
        if X[0]<ab1_list[c-1]:
            return 3#左上
while True:
    #初期化用
    a=0
    b=0
    #情報を分ける用
    inf0=0
    inf_list=[]
    #座標計算用
    a0=float()
    b0=float()
    a1=float()
    b1=float()
    #傾き、切片格納リスト
    ab0_list=[]
    ab1_list=[]

    first_list=[]
    second_list=[]
    last_list=[]
    error_list=[]
    read_list=[]
    #列のリスト
    a_list=[]
    b_list=[]
    c_list=[]
    d_list=[]
    e_list=[]

    ret, frame = cap.read()
    if frame is None:
        break
 
    # 画像の読み込み + グレースケール化
    img = frame
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    temp = cv2.imread('tenji2_2syukusyou3.jpg')
    template = cv2.resize(temp, dsize=(25,25))
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    template2= cv2.resize(temp, dsize=(20,20))
    template_gray2 = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)
# 処理対象画像に対して、テンプレート画像との類似度を算出する
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    res2 = cv2.matchTemplate(img_gray, template_gray2, cv2.TM_CCOEFF_NORMED)
    # 類似度の高い部分を検出する
    threshold = 0.8
    loc = np.where(res >= threshold)
    loc2 =np.where(res2 >= threshold)
    # テンプレートマッチング画像の高さ、幅を取得する
    h, w = template_gray.shape 
    # 検出した部分に赤枠をつける    
    first_list=list(zip(*loc[::-1]))+list(zip(*loc2[::-1]))         
    for h in  range (len(first_list)):
        for hh in range (len(first_list)):
            if h>=hh:
                continue
            if abs(first_list[h][0]-first_list[hh][0])>=10 or abs(first_list[h][1]-first_list[hh][1])>=10:
                continue
            if first_list[hh] in error_list:
                continue
            error_list.append(first_list[hh])
  
    for g in error_list:
        first_list.remove(g)
    first_list=sorted(first_list)
    for c in range (len(first_list)):
        if c>=0 and c<5:
            a_list.append(first_list[c][::-1])
        if c>=5 and c<10:
            b_list.append(first_list[c][::-1]) 
        if c>=10 and c<15:
            c_list.append(first_list[c][::-1])   
        if c>=15 and c<20:
            d_list.append(first_list[c][::-1])
        if c>=20 and c<25:
            e_list.append(first_list[c][::-1])
    a_list=sorted(a_list)
    b_list=sorted(b_list)
    c_list=sorted(c_list)
    d_list=sorted(d_list)
    e_list=sorted(e_list)
    second_list.clear()
    second_list=a_list+b_list+c_list+d_list+e_list
    for hhh in second_list: 
        last_list.append(hhh[::-1])
    #認識部分
    
    for g in  range (len(last_list)):
        cv2.circle(img,(last_list[g][0],last_list[g][1]),20,(0,0,255),-1)
        cv2.putText(img, str(g), (last_list[g][0], last_list[g][1]),cv2.FONT_HERSHEY_PLAIN,4, (0, 0, 0), 3, cv2.LINE_AA)
        if len(last_list)==25:
            if g==1 or g==2 or g==3:
                cv2.line(img, (last_list[g][0], last_list[g][1]), (last_list[g+20][0], last_list[g+20][1]), (0, 0, 255))
                ab0_list.append(getXY(last_list[g][1],last_list[g+20][1]))
            if g==5 or g==10 or g==15:
                cv2.line(img, (last_list[g][0], last_list[g][1]), (last_list[g+4][0], last_list[g+4][1]), (0, 0, 255))
                ab1_list.append(getXY(last_list[g][0],last_list[g+4][0]))

    if len(last_list)==25:
        read_list.append(getInf(last_list[6],1,1))
        read_list.append(getInf(last_list[7],2,1))
        read_list.append(getInf(last_list[8],3,1))
        read_list.append(getInf(last_list[11],1,2))
        read_list.append(getInf(last_list[12],2,2))
        read_list.append(getInf(last_list[13],3,2))
        read_list.append(getInf(last_list[16],1,3))
        read_list.append(getInf(last_list[17],2,3))
        read_list.append(getInf(last_list[18],3,3))
       
        
    
    if read_list==[1,1,1,1,1,1,1,1,1]:
        sound1()
        sound2()
        continue
    if read_list==[2,2,2,2,2,2,2,2,2]:
        sound1()
        sound3()
        continue
    # 画像の保存・表示
    cv2.imshow(window_name, frame)
    #cv2.imshow("img",img2)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
 
cv2.destroyWindow(window_name)
