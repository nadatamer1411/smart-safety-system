smart-safety-system/
│
├── src/
│   ├── detection/
│   ├── alerts/
│   ├── dashboard/
│   └── utils/
│
├── data/
│   ├── logs.csv
│   └── images/
│
├── models/
├── app.py
└── requirements.txt

للتشغيل
venv\Scripts\activate
python src/detection/test_video.py
streamlit run app.py

للتدريب
venv\Scripts\activate
yolo detect train data=helmet_dataset/data.yaml model=yolov8n.pt epochs=3

للتجهيز
pip install -r requirements.txt

للرفع على GIT
git pull                        # sync latest changes from remote first
git add .                       # stage all changed files
git commit -m "your message"    # commit with a description
git push     

Switch to Branch
git fetch
git checkout version-2
git switch version-2                  # push to remote
