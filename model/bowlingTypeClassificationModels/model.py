import cv2
from ultralytics.yolo.engine.model import YOLO
import numpy as np
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

"""### FUNCTION: coordinates_player()"""


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
      print(tensor)
    
      if tensor=="":
        coordinates={"batsman":{"xmin":0,"ymin":0,"w": 0,"h": 0,"u":0},
                   "bowler":{"xmin":0,"ymin": 0,"w": 0,"h":0,"u":0}}
        return coordinates
    
      tensor = eval(tensor)
      # tensor = tensor[0]
   
      print(tensor)
      tensor=np.array(tensor)
      # print("------------tensor------------",tensor)
      # print(len(tensor))
      print(tensor)
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
def ball_tracking(path,model_ball,player_det_model):
 
  frame_coordinates={}
  # min_distance=1000
  # distance=100000
  distance={}
  frame_coordinates_player={}
  # Create a VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')

  #Load the video file
  cap = cv2.VideoCapture(path)

  #Get the frame dimensions
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  
  #Video Writer
  out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
  # out = cv2.VideoWriter('model/bowlingTypeClassificationModels/content/output.mp4', fourcc, 30.0, (frame_width, frame_height))

  #Defining the Model
  model = YOLO(model_ball)
  model_player = YOLO(player_det_model)

  frames=[]
  count=1
  
  while True:

    ret, frame = cap.read()
    if not ret:
      break 
    
    cv2.imwrite('pic.jpg',frame)
    # cv2.imwrite('model/bowlingTypeClassificationModels/content/pic.jpg',frame)

    results = model.predict(source='pic.jpg')
    # results = model.predict(source='model/bowlingTypeClassificationModels/content/pic.jpg')

    # For writting the predicted result
    with open("output.txt","w") as f:
      f.write(str(results))

    # Get the bounding box around the ball        
    path='output.txt'
    # path='model/bowlingTypeClassificationModels/content/output.txt'
    
    #Getting the exact coordinates      
    x,y,w,h =coordinates(path)

    frame_coordinates[count]={"xmin":x,"ymin":y,"w":w,"h":h}
    
    # Draw a bounding box around the ball
    cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 0, 255), 2)
    
    results = model_player.predict(source=frame)
    with open("output2.txt","w") as f:
        f.write(str(results))
    
    player_coordinates=coordinates_player("output2.txt")
    frame_coordinates_player[count]=player_coordinates


    cv2.rectangle(frame, (player_coordinates["batsman"]["xmin"],player_coordinates["batsman"]["ymin"]),(player_coordinates["batsman"]["xmin"]+player_coordinates["batsman"]["w"],player_coordinates["batsman"]["ymin"]+player_coordinates["batsman"]["h"]), (0, 255, 0), 2)
    cv2.rectangle(frame, (player_coordinates["bowler"]["xmin"],player_coordinates["bowler"]["ymin"]),(player_coordinates["bowler"]["xmin"]+player_coordinates["bowler"]["w"],player_coordinates["bowler"]["ymin"]+player_coordinates["bowler"]["h"]), (0, 255, 0), 2)
    
    if y!=0 and player_coordinates["batsman"]["ymin"]!=0:
      distance_ball_and_batsman=y-player_coordinates["batsman"]["ymin"]
      distance[count]=distance_ball_and_batsman
   
   
    # Display the current frame
    frames.append(frame)

    print("---------frame------------",count)
    print("---------distance------------",distance)

    count=count+1
    # cv2_imshow(frame)

  # Writting the frames in output video
  for frame in frames:
      out.write(frame)
  out.release()

  return frame_coordinates,frame_coordinates_player,distance

"""### FUNCTION: finding_player_in_consectiveframe()"""

def finding_player_in_consectiveframe(frame_number,frame_coordinates_player,player):
    print("finding in consecutive")
    last_key = list(frame_coordinates_player.keys())[-1]
    for i in [1,-1,2,-2,3,-3]:
      key=frame_number+i
      print("consecutive",key)
      if key>0 and key<last_key: 
        # print(key)
        if frame_coordinates_player[key][player]["xmin"]!=0 and frame_coordinates_player[key][player]["ymin"]!=0:
          print("Found from consecutive")
          frame_coordinates_player[frame_number][player]["xmin"]=frame_coordinates_player[key][player]["xmin"]
          frame_coordinates_player[frame_number][player]["ymin"]=frame_coordinates_player[key][player]["ymin"]
          frame_coordinates_player[frame_number][player]["u"]=1
          return frame_coordinates_player
    return frame_coordinates_player

"""### FUNCTION: extracting_first_and_bounce"""

def extracting_first_and_bounce(frame_coordinates,frame_coordinates_player,min_distance_frame):
   
   print("Extracting first and bounce frame")
      
   bounce_frame=0
   first_frame=0
   sorted_dict = dict(sorted(frame_coordinates.items(), key=lambda x: x[1]["ymin"], reverse=True))
   
   for key,value in frame_coordinates.items():
    print("key ,value :",key,value)
    
    if frame_coordinates[key]["ymin"]==0 and frame_coordinates[key]["xmin"]==0:
      continue
    else:
      first_frame=key
      if frame_coordinates_player[key]["batsman"]["ymin"]==0 and frame_coordinates_player[key]["batsman"]["xmin"]==0:
       frame_coordinates_player = finding_player_in_consectiveframe(key,frame_coordinates_player,"batsman")
      if frame_coordinates_player[key]["bowler"]["ymin"]==0 and frame_coordinates_player[key]["bowler"]["xmin"]==0:
        frame_coordinates_player = finding_player_in_consectiveframe(key,frame_coordinates_player,"bowler")
      break
   print("First Frame Detected",first_frame)
  
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
        print("condition is met")
        break
        

   
  #  print("bounce",bounce_frame)
   return first_frame,bounce_frame

"""### FUNCTION: ball_classification()"""

def ball_classification(player_name,filename,folder_path,player_det_model,ball_det_model):
     
      cap = cv2.VideoCapture(folder_path)
      count=1
      H1=0
      H2=0
      height_of_pitch=0
      zoom=0
      model_player = YOLO(player_det_model)
      model_ball = YOLO(ball_det_model)
      frame_coordinates,frame_coordinates_player,min_distance_frame=ball_tracking(folder_path,ball_det_model,player_det_model)
      print("-------------------Ball Coordinates----------------------")
      print(frame_coordinates)
    
      first_frame,bounce_frame=extracting_first_and_bounce(frame_coordinates,frame_coordinates_player,min_distance_frame)
      print(first_frame,bounce_frame)
      # # Loop through the frames
      while True:
        # # Read the next frame    
        ret, frame = cap.read()
        if not ret:
          print("file ended")
          break
        if(count==first_frame):
          print("first frame",count)
          cv2.imwrite('pic.jpg',frame)
          # cv2.imwrite('model/bowlingTypeClassificationModels/content/pic.jpg',frame)
          results = model_player.predict(source=frame)
          # with open("model/bowlingTypeClassificationModels/content/output.txt","w") as f:
          with open("output.txt","w") as f:
            f.write(str(results))
          coordinates_bt=coordinates_player("output.txt")
          # coordinates_bt=coordinates_player("model/bowlingTypeClassificationModels/content/output.txt")
          
          
          H1=coordinates_bt["batsman"]["h"]
      
          bottom_batsman=coordinates_bt["batsman"]["ymin"]+coordinates_bt["batsman"]["h"]
          bottom_bowler=coordinates_bt["bowler"]["ymin"]+coordinates_bt["bowler"]["h"]
          for i in [1,-1,2,-2,3,-3]:
            if H1==0:
             for key,value in frame_coordinates_player.items():
              if key==count+i:
                coordinates_bt=frame_coordinates_player[key]
            H1=coordinates_bt["batsman"]["h"]  
          
          # print("----------------------------------------------------------------")
          # print("height of batsman in first frame (H1):",H1)
          cv2.rectangle(frame, (coordinates_bt["batsman"]["xmin"],coordinates_bt["batsman"]["ymin"]),(coordinates_bt["batsman"]["xmin"]+coordinates_bt["batsman"]["w"],coordinates_bt["batsman"]["ymin"]+coordinates_bt["batsman"]["h"]), (0, 0, 255), 2)
          cv2.rectangle(frame, (coordinates_bt["bowler"]["xmin"],coordinates_bt["bowler"]["ymin"]),(coordinates_bt["bowler"]["xmin"]+coordinates_bt["bowler"]["w"],coordinates_bt["bowler"]["ymin"]+coordinates_bt["bowler"]["h"]), (255,0, 0), 2)
          height_of_pitch=bottom_bowler-bottom_batsman
          
          # print("height of pitch in first frame :",height_of_pitch)
          
          results = model_ball.predict(source="pic.jpg")
          with open("output.txt","w") as f:
            f.write(str(results))
          xmin,ymin,h,w=coordinates("output.txt")
          cv2.rectangle(frame, (xmin,ymin),(xmin+w,ymin+h), (0, 255,0 ), 1)
          # cv2_imshow(frame)

        if(count==bounce_frame):
          
          print("bounce frame",bounce_frame)
          cv2.imwrite('pic.jpg',frame)
          # cv2.imwrite('model/bowlingTypeClassificationModels/content/pic.jpg',frame)
          results = model_player.predict(source="pic.jpg")
          # results = model_player.predict(source="model/bowlingTypeClassificationModels/content/pic.jpg")
          with open("output.txt","w") as f:
            f.write(str(results))
         
          coordinates_bt=coordinates_player("output.txt")
          H2=coordinates_bt["batsman"]["h"]
         
          if H2==0:
            for i in [1,-1,2,-2,3,-3]:
              if H2==0:
                for key,value in frame_coordinates_player.items():
                  if key==count+i:
                    coordinates_bt=frame_coordinates_player[key]
              H2=coordinates_bt["batsman"]["h"]  

          # cv2.imwrite("/content/drive/MyDrive/PSL 8/Bounce/"+player_name+"/"+filename+".png",frame)
          # cv2_imshow(frame)
          # print("----------------------------------------------------------------")
          # print("batsman height in bounce frame (H2):",H2)


          cv2.rectangle(frame, (coordinates_bt["batsman"]["xmin"],coordinates_bt["batsman"]["ymin"]),(coordinates_bt["batsman"]["xmin"]+coordinates_bt["batsman"]["w"],coordinates_bt["batsman"]["ymin"]+coordinates_bt["batsman"]["h"]), (0, 0, 255), 2)
          cv2.rectangle(frame, (coordinates_bt["bowler"]["xmin"],coordinates_bt["bowler"]["ymin"]),(coordinates_bt["bowler"]["xmin"]+coordinates_bt["bowler"]["w"],coordinates_bt["bowler"]["ymin"]+coordinates_bt["bowler"]["h"]), (255, 0, 0), 2)
          
          results = model_ball.predict(source="pic.jpg")
          with open("output.txt","w") as f:
            f.write(str(results))
          xmin,ymin,h,w=coordinates("output.txt")
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
      print("scaled_height:",scaled_height)
      print("ymin:",ymin)
      print("height_of_pitch",height_of_pitch)
      if ymin<=(scaled_height*2):
        ball_type="Yorker"
      elif ymin<=(scaled_height*6) and ymin>(scaled_height*2):
        ball_type="Full"
      elif ymin>(scaled_height*6) and ymin<=(scaled_height*8):
        ball_type="Good"
      else:
         ball_type="Short"
      return ball_type
      # with open("/content/drive/MyDrive/PSL 8/Mapped_Coord_Ball_Csv"+player_name+".csv","a") as csv_file:
      #   writer2 = csv.writer(csv_file)
      #   writer2.writerow([filename, coordinates_bt["batsman"], ball_coordinates,ball_type])

        # csv_file.close()


"""# EXECUTION CODE"""

# video_path="/content/drive/MyDrive/PSL 8/Lahore/Fakhar/Match 1/LahoreVsMultan.13.2.mp4"
# player_det_model="/content/drive/MyDrive/FYP_Code/Yolo_v8/Batsman Bowler Detection/18 march runs/detect/train5/weights/best.pt"
# ball_det_model="/content/drive/MyDrive/FYP_Code/Yolo_v8/Ball Detection/19 feb runs/detect/train/weights/best.pt"

def classifyBowlingType():
  player_det_model="model/bowlingTypeClassificationModels/playerDetMod.pt"
  ball_det_model="model/bowlingTypeClassificationModels/ballDetMod.pt"
  video_path = "temp.mp4"
  videofile = "temp.mp4"

  player = "test"
  output = ball_classification(player,videofile,video_path,player_det_model,ball_det_model)
  
  # os.remove("model/bowlingTypeClassificationModels/content/output.mp4")
  # os.remove("model/bowlingTypeClassificationModels/content/output.txt") 
  # os.remove("model/bowlingTypeClassificationModels/content/output2.txt")
  # os.remove("model/bowlingTypeClassificationModels/content/pic.jpg")

  # Ideal
  # bowling_type = ball_classification(video_path,player_det_model,ball_det_model)

  return {"Bowling type": output}

