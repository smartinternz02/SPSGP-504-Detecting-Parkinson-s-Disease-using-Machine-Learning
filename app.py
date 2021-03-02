import pickle
import cv2
from skimage import feature
from flask import Flask,request,render_template
import os.path

app=Flask(__name__)

#rendering html pages

@app.route("/")
def about():
    return render_template('about.html')
    
@app.route("/about")
def home():
    return render_template("home.html")
    
@app.route("/info")
def information():
    return render_template("info.html")
    
@app.route("/upload")
def test():
    return render_template("index6.html")
    
@app.route("/predict",methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']  #request the file
        basepath=os.path.dirname(__file__)   #store the file dir
        filepath=os.path.join(basepath,"uploads",f.filename) #stores the the file in uploads folder
        f.save(filepath) #saving the file
        print("[Info] Loading model...")
        model=pickle.loads(open('parkinson.pkl',"rb").read())
        
        # pre-process the image in same manner
        
        image=cv2.imread(filepath)
        output=image.copy()
        
        # load the input image,convert to grayscale and resize
        
        output=cv2.resize(output,(128,128))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image=cv2.resize(image,(200,200))
        image=cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        #feature using last trained random forest
        
        features=feature.hog(image,orientations=9,pixels_per_cell=(10,10),cells_per_block=(2,2),transform_sqrt=True,block_norm="L1")
        preds=model.predict([features])
        print(preds)
        ls=["healthy","parkinson"]
        result=ls[preds[0]]
        
        #the set of output images
        if result=="healthy":
            color=(0,255,0)
        else:
            color=(0,0,255)
            
        cv2.putText(output,result,(3,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
        cv2.imshow("Output",output)
        cv2.waitKey(0)
        return result
        
    return None
    

# MAin Function

if __name__ == "__main__":
    app.run(debug=True)
    
