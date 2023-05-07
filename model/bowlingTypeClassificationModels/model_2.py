# import re
import numpy as np
import cv2
import numpy as np
# from google.colab.patches import cv2_imshow
from ultralytics.yolo.engine.model import YOLO
# import time
# import imageio
import os






def coordinates(path):
  xmin, ymin, xmax, ymax, w, h=0.00,0.00,0.00,0.00,0.00, 0.00
  max_ci=0

  with open(path, "r") as file:
    for line in file:
        if line.startswith(" + tensor"):
            line = line.strip()
            line=line[line.index("[") + 1 : line.index("]")]
            abc= line.startswith("[")
            if abc:
              detections=line.split("]")
            else:
              xmin, ymin, xmax, ymax =0,0,0,0
              break
            for detection in detections:
              if detection.startswith("["):
                  xmin, ymin, xmax, ymax, ci,  _ = map(float, detection[detection.index("[") + 1 : ].split(","))
                  ci=float(ci)
                  if ci>max_ci:
                    continue
                  else:
                    xmin, ymin, xmax, ymax=0
            w = xmax- xmin
            h = ymax-ymin               
  return  int(xmin), int(ymin), int(w), int(h)


def coordinates_player(file):
    # coordinates={}
    batsman_xmin, batsman_ymin, batsman_xmax, batsman_ymax,bt_ci=0,0,0,0,0
    bowler_xmin, bowler_ymin, bowler_xmax, bowler_ymax,bl_ci=0,0,0,0,0
    with open(file, "r") as file:
      tensor=False
      tensor_line=""
      for line in file:
        if line.startswith(" + tensor"):
            tensor=True
        if(tensor):
          tensor_line=tensor_line+line
      
      tensor = tensor_line.split(" + tensor")[-1].strip()
     
      tensor = tensor.replace("(","").replace("device='cuda:0'","").replace(")]","")
      tensor = tensor.replace(", ,",",")
      tensor=tensor.replace("size=0, 6","")
      tensor=tensor.replace(")","")
      
      tensor=tensor.replace("[],","  ")
      # print("") 
      # print("Batsman Bowler Tensor:"," "+tensor+" ") 
      # print("")
      tensor=tensor.strip()
    #   print(tensor)
    
      if tensor=="":
        coordinates={"batsman":{"xmin":0,"ymin":0,"w": 0,"h": 0,"u":0},
                   "bowler":{"xmin":0,"ymin": 0,"w": 0,"h":0,"u":0}}
        return coordinates
    
      tensor = eval(tensor)
      # tensor = tensor[0]
   
    #   print(tensor)
      tensor=np.array(tensor)
      # print("------------tensor------------",tensor)
      # print(len(tensor))
    #   print(tensor)
      if(len(tensor)==1):
        tensor_element = tensor[0]
        if(tensor_element!=[]):

          for i in range(len(tensor)):
            if tensor[i][5]==1 and bl_ci==0:
                bowler_xmin, bowler_ymin, bowler_xmax, bowler_ymax,bl_ci,bl_cl = tensor[i][0], tensor[i][1], tensor[i][2], tensor[i][3],tensor[i][4],tensor[i][5]
            elif tensor[i][5]==0 and bt_ci==0:
        
              batsman_xmin, batsman_ymin, batsman_xmax, batsman_ymax,bt_ci,bt_cl = tensor[i][0], tensor[i][1], tensor[i][2], tensor[i][3],tensor[i][4],tensor[i][5]
            elif tensor[i][5]==0 and bt_ci!=0:
          
              if ((bt_ci<tensor[i][4])):
                  batsman_xmin, batsman_ymin, batsman_xmax, batsman_ymax,bt_ci,bt_cl = tensor[i][0], tensor[i][1], tensor[i][2], tensor[i][3],tensor[i][4],tensor[i][5]
            elif tensor[i][5]==1 and bl_ci!=0:

              if bl_ci<tensor[i][4]:
                  bowler_xmin, bowler_ymin, bowler_xmax, bowler_ymax,bl_ci,bl_cl = tensor[i][0], tensor[i][1], tensor[i][2], tensor[i][3],tensor[i][4],tensor[i][5]
      else:
       
        for i in range(len(tensor)):
 
          if tensor[i][5]==1 and bl_ci==0:
             
             bowler_xmin, bowler_ymin, bowler_xmax, bowler_ymax,bl_ci,bl_cl = tensor[i][0], tensor[i][1], tensor[i][2], tensor[i][3],tensor[i][4],tensor[i][5]
          elif tensor[i][5]==0 and bt_ci==0:
            batsman_xmin, batsman_ymin, batsman_xmax, batsman_ymax,bt_ci,bt_cl = tensor[i][0], tensor[i][1], tensor[i][2], tensor[i][3],tensor[i][4],tensor[i][5]
          elif tensor[i][5]==0 and bt_ci!=0:
            if  bt_ci<tensor[i][4]:
                batsman_xmin, batsman_ymin, batsman_xmax, batsman_ymax,bt_ci,bt_cl = tensor[i][0], tensor[i][1], tensor[i][2], tensor[i][3],tensor[i][4],tensor[i][5]
            # if  bt_ci<tensor[0][i][4]:
            #     batsman_xmin, batsman_ymin, batsman_xmax, batsman_ymax,bt_ci,bt_cl = tensor[i][0], tensor[i][1], tensor[i][2], tensor[i][3],tensor[i][4],tensor[i][5]
          elif tensor[i][5]==1 and bl_ci!=0:
            if  bl_ci<tensor[i][4]:
                 bowler_xmin, bowler_ymin, bowler_xmax, bowler_ymax,bl_ci,bl_cl = tensor[i][0], tensor[i][1], tensor[i][2], tensor[i][3],tensor[i][4],tensor[i][5]
            # if bl_ci<tensor[0][i][4]:
                
            #     bowler_xmin, bowler_ymin, bowler_xmax, bowler_ymax,bl_ci,bl_cl = tensor[i][0], tensor[i][1], tensor[i][2], tensor[i][3],tensor[i][4],tensor[i][5]

      batsman_w=batsman_xmax-batsman_xmin
      batsman_h=batsman_ymax-batsman_ymin
   
      bowler_w= bowler_xmax- bowler_xmin
      bowler_h= bowler_ymax- bowler_ymin

      
      coordinates={"batsman":{"xmin":int(batsman_xmin),"ymin":int(batsman_ymin),"w": int(batsman_w),"h": int(batsman_h),"u":0},
                   "bowler":{"xmin":int(bowler_xmin),"ymin": int(bowler_ymin),"w": int(bowler_w),"h":int(bowler_h),"u":0}}
      return coordinates


# frame_coordinates={}
def ball_tracking(cap,model_ball,player_det_model):
 
  frame_coordinates={}
  # min_distance=1000
  # distance=100000
  distance={}
  frame_coordinates_player={}
  # Create a VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')

  #Load the video file
#   cap = cv2.VideoCapture(path)

  #Get the frame dimensions
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  
  #Video Writer
  out = cv2.VideoWriter('model/bowlingTypeClassificationModels/content/output.mp4', fourcc, 30.0, (frame_width, frame_height))

  #Defining the Model
  model = YOLO(model_ball)
  model_player = YOLO(player_det_model)

  frames=[]
  count=1;
  
  while True:

    ret, frame = cap.read()
    if not ret:
      break 
    
    cv2.imwrite('model/bowlingTypeClassificationModels/content/pic.jpg',frame)

    results = model.predict(source='model/bowlingTypeClassificationModels/content/pic.jpg')

    # For writting the predicted result
    with open("model/bowlingTypeClassificationModels/content/output.txt","w") as f:
      f.write(str(results))

    # Get the bounding box around the ball        
    path='model/bowlingTypeClassificationModels/content/output.txt'
    
    #Getting the exact coordinates      
    x,y,w,h =coordinates(path)

    frame_coordinates[count]={"xmin":x,"ymin":y,"w":w,"h":h}
    
    # Draw a bounding box around the ball
    cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 0, 255), 2)
    
    results = model_player.predict(source=frame)
    with open("model/bowlingTypeClassificationModels/content/output2.txt","w") as f:
        f.write(str(results))
    
    player_coordinates=coordinates_player("model/bowlingTypeClassificationModels/content/output2.txt")
    frame_coordinates_player[count]=player_coordinates


    cv2.rectangle(frame, (player_coordinates["batsman"]["xmin"],player_coordinates["batsman"]["ymin"]),(player_coordinates["batsman"]["xmin"]+player_coordinates["batsman"]["w"],player_coordinates["batsman"]["ymin"]+player_coordinates["batsman"]["h"]), (0, 255, 0), 2)
    cv2.rectangle(frame, (player_coordinates["bowler"]["xmin"],player_coordinates["bowler"]["ymin"]),(player_coordinates["bowler"]["xmin"]+player_coordinates["bowler"]["w"],player_coordinates["bowler"]["ymin"]+player_coordinates["bowler"]["h"]), (0, 255, 0), 2)
    
    if y!=0 and player_coordinates["batsman"]["ymin"]!=0:
      distance_ball_and_batsman=y-player_coordinates["batsman"]["ymin"]
      distance[count]=distance_ball_and_batsman
   
   
    # Display the current frame
    frames.append(frame)

    # print("---------frame------------",count)
    # print("---------distance------------",distance)

    count=count+1
    # cv2.imshow(frame)

  # Writting the frames in output video
  for frame in frames:
      out.write(frame)
  out.release()

  return frame_coordinates,frame_coordinates_player,distance



def finding_player_in_consectiveframe(frame_number,frame_coordinates_player,player):
    print("finding in consecutive")
    last_key = list(frame_coordinates_player.keys())[-1]
    for i in [1,-1,2,-2,3,-3]:
      key=frame_number+i
    #   print("consecutive",key)
      if key>0 and key<last_key: 
        # print(key)
        if frame_coordinates_player[key][player]["xmin"]!=0 and frame_coordinates_player[key][player]["ymin"]!=0:
          print("Found from consecutive")
          frame_coordinates_player[frame_number][player]["xmin"]=frame_coordinates_player[key][player]["xmin"]
          frame_coordinates_player[frame_number][player]["ymin"]=frame_coordinates_player[key][player]["ymin"]
          frame_coordinates_player[frame_number][player]["u"]=1
          return frame_coordinates_player
    return frame_coordinates_player


def extracting_first_and_bounce(frame_coordinates,frame_coordinates_player,min_distance_frame):
   
   print("Extracting first and bounce frame")
      
   bounce_frame=0
   first_frame=0
   sorted_dict = dict(sorted(frame_coordinates.items(), key=lambda x: x[1]["ymin"], reverse=True))
   
   for key,value in frame_coordinates.items():
    # print("key ,value :",key,value)
    
    if frame_coordinates[key]["ymin"]==0 and frame_coordinates[key]["xmin"]==0:
      continue
    else:
      first_frame=key
      if frame_coordinates_player[key]["batsman"]["ymin"]==0 and frame_coordinates_player[key]["batsman"]["xmin"]==0:
       frame_coordinates_player = finding_player_in_consectiveframe(key,frame_coordinates_player,"batsman")
      if frame_coordinates_player[key]["bowler"]["ymin"]==0 and frame_coordinates_player[key]["bowler"]["xmin"]==0:
        frame_coordinates_player = finding_player_in_consectiveframe(key,frame_coordinates_player,"bowler")
      break
#    print("First Frame Detected",first_frame)
  
   for key,value in sorted_dict.items():

      # if key>min_distance_frame:
      #   continue
      print(key,value)
      if frame_coordinates_player[key]["batsman"]["ymin"]==0 and frame_coordinates_player[key]["batsman"]["xmin"]==0:
       print("batsman not detected")
       frame_coordinates_player = finding_player_in_consectiveframe(key,frame_coordinates_player,"batsman")
      
      if frame_coordinates_player[key]["bowler"]["ymin"]==0 and frame_coordinates_player[key]["bowler"]["xmin"]==0:
        print("bowler not detected")
        frame_coordinates_player = finding_player_in_consectiveframe(key,frame_coordinates_player,"bowler")

      
      bounce_frame=key
      
      prev_value = None
      first_peak= False
      condition=False
      
      for key2, value2 in min_distance_frame.items():
        if prev_value is None:
            prev_value = value2
            continue
        
        if value2 < prev_value and not first_peak:
            prev_value = value2
            continue
        elif value2>prev_value and not first_peak:
            prev_value = value2
            first_peak = True

        elif value2>prev_value and  first_peak:
            prev_value = value2
        
        elif value2 < prev_value and  first_peak:
            prev_value = value2
            if key<key2:
              condition=True
              break
      if (condition):
        # print("condition is met")
        break
        

   
  #  print("bounce",bounce_frame)
   return first_frame,bounce_frame 


def ball_classification(cap, player_det_model, ball_det_model):
     
    count=1
    H1=0
    H2=0
    height_of_pitch=0
    zoom=0
    model_player = YOLO(player_det_model)
    model_ball = YOLO(ball_det_model)
    frame_coordinates,frame_coordinates_player,min_distance_frame=ball_tracking(cap,ball_det_model,player_det_model)
    
    first_frame,bounce_frame=extracting_first_and_bounce(frame_coordinates,frame_coordinates_player,min_distance_frame)
    
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        if(count==first_frame):
            cv2.imwrite('model/bowlingTypeClassificationModels/content/pic.jpg',frame)
            results = model_player.predict(source=frame)
            with open("model/bowlingTypeClassificationModels/content/output.txt","w") as f:
                f.write(str(results))
            coordinates_bt=coordinates_player("model/bowlingTypeClassificationModels/content/output.txt")


            H1=coordinates_bt["batsman"]["h"]

            bottom_batsman=coordinates_bt["batsman"]["ymin"]+coordinates_bt["batsman"]["h"]
            bottom_bowler=coordinates_bt["bowler"]["ymin"]+coordinates_bt["bowler"]["h"]
            for i in [1,-1,2,-2,3,-3]:
                if H1==0:
                    for key,value in frame_coordinates_player.items():
                        if key==count+i:
                            coordinates_bt=frame_coordinates_player[key]
                H1=coordinates_bt["batsman"]["h"]  

            cv2.rectangle(frame, (coordinates_bt["batsman"]["xmin"],coordinates_bt["batsman"]["ymin"]),(coordinates_bt["batsman"]["xmin"]+coordinates_bt["batsman"]["w"],coordinates_bt["batsman"]["ymin"]+coordinates_bt["batsman"]["h"]), (0, 0, 255), 2)
            cv2.rectangle(frame, (coordinates_bt["bowler"]["xmin"],coordinates_bt["bowler"]["ymin"]),(coordinates_bt["bowler"]["xmin"]+coordinates_bt["bowler"]["w"],coordinates_bt["bowler"]["ymin"]+coordinates_bt["bowler"]["h"]), (255,0, 0), 2)
            height_of_pitch=bottom_bowler-bottom_batsman


            results = model_ball.predict(source="model/bowlingTypeClassificationModels/content/pic.jpg")
            with open("model/bowlingTypeClassificationModels/content/output.txt","w") as f:
              f.write(str(results))
            xmin,ymin,h,w=coordinates("model/bowlingTypeClassificationModels/content/output.txt")
            cv2.rectangle(frame, (xmin,ymin),(xmin+w,ymin+h), (0, 255,0 ), 1)
            # cv2_imshow(frame)

        if(count==bounce_frame):
          
            cv2.imwrite('model/bowlingTypeClassificationModels/content/pic.jpg',frame)
            results = model_player.predict(source="model/bowlingTypeClassificationModels/content/pic.jpg")
            with open("model/bowlingTypeClassificationModels/content/output.txt","w") as f:
              f.write(str(results))

            coordinates_bt=coordinates_player("model/bowlingTypeClassificationModels/content/output.txt")
            H2=coordinates_bt["batsman"]["h"]

            if H2==0:
              for i in [1,-1,2,-2,3,-3]:
                if H2==0:
                  for key,value in frame_coordinates_player.items():
                    if key==count+i:
                      coordinates_bt=frame_coordinates_player[key]
                H2=coordinates_bt["batsman"]["h"]   
            cv2.rectangle(frame, (coordinates_bt["batsman"]["xmin"],coordinates_bt["batsman"]["ymin"]),(coordinates_bt["batsman"]["xmin"]+coordinates_bt["batsman"]["w"],coordinates_bt["batsman"]["ymin"]+coordinates_bt["batsman"]["h"]), (0, 0, 255), 2)
            cv2.rectangle(frame, (coordinates_bt["bowler"]["xmin"],coordinates_bt["bowler"]["ymin"]),(coordinates_bt["bowler"]["xmin"]+coordinates_bt["bowler"]["w"],coordinates_bt["bowler"]["ymin"]+coordinates_bt["bowler"]["h"]), (255, 0, 0), 2)

            results = model_ball.predict(source="model/bowlingTypeClassificationModels/content/pic.jpg")
            with open("model/bowlingTypeClassificationModels/content/output.txt","w") as f:
              f.write(str(results))
            xmin,ymin,h,w=coordinates("model/bowlingTypeClassificationModels/content/output.txt")
            ball_coordinates=[xmin,ymin,h,w]
            cv2.rectangle(frame, (xmin,ymin),(xmin+w,ymin+h), (0, 0, 255), 2)

            ymin=ymin-(coordinates_bt["batsman"]["ymin"]+H2)+2  
         
          # writer2.writerow([filename, coordinates_bt["batsman"], ball_coordinates])
        count+=1
      
    if H1 != 0:
        zoom=H2/H1
        print("zoom factor:",zoom)

    height_of_pitch=height_of_pitch*zoom
      # print("height_of_pitch after including zoom factor:",height_of_pitch)


    scaled_height=height_of_pitch/20
    if ymin<=(scaled_height*2):
        ball_type="Yorker"
    elif ymin<=(scaled_height*6) and ymin>(scaled_height*2):
        ball_type="Full"
    elif ymin>(scaled_height*6) and ymin<=(scaled_height*8):
        ball_type="Good"
    else:
        ball_type="Short"
    return ball_type



# def classifyBowlingType(video: cv2.VideoCapture):

if __name__ == "__main__":
    player_det_model="model/bowlingTypeClassificationModels/playerDetMod.pt"
    ball_det_model="model/bowlingTypeClassificationModels/ballDetMod.pt"

    videofile = "C:/Users/ahmed/Development/FYP/LahoreVsMultan.13.2.mp4"
    video = cv2.VideoCapture(videofile)
    # video_path = "C:/Users/ahmed/Development/FYP/Module 2/Dataset/train/Pull Shot/Match 13 - Cropped 2, Islamabad, 16.2, Pull Shot, Lofted.mp4"
    output = ball_classification(video, player_det_model,ball_det_model)
    # os.remove("model/bowlingTypeClassificationModels/content/output.mp4")
    # os.remove("model/bowlingTypeClassificationModels/content/output.txt")
    # os.remove("model/bowlingTypeClassificationModels/content/output2.txt")
    # os.remove("model/bowlingTypeClassificationModels/content/pic.jpg")

    # return {"BOWLING TYPE": output}
    print(output)
