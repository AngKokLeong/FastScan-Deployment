from typing import Any


from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements


class StreamlitUserInterface:

    def __init__(self, **kwargs: Any):
        check_requirements("streamlit>=1.29.0")
        import streamlit as st
        from ultralytics import YOLO
        from ultralytics import settings
        import cv2

        self.st = st
        self.yolo = YOLO
        self.cv2 = cv2
        self.yolo_settings = settings

        self.model = None

        self.confidence_threshold = 0.25
        self.iou = 0.45
        self.model_path = None # 


    def setup_web_user_interface(self) -> None:

        '''
            Setup the Streamlit Web interface with custom HTML elements
        '''

        # Refer to https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/streamlit_inference.py

        menu_style_cfg = """<style>MainMenu {visibility: hidden;} </style>"""

        main_title_cfg = """
                        <div>
                            <h1 style="color:#FF64DA; text-align:center; 
                                        font-size:40px; 
                                        margin-top:-50px;
                                        font-family: 'Archivo', san-serif; margin-bottom:20px;"
                            >
                                FastScan
                            </h1>

                        </div>
                        """


        sub_title_cfg = """

            <div>
                <h4 style="color: #042AFF; 
                            text-align: center; 
                            font-family: 'Archivo', sans-serif;
                            margin-top:-15px;
                            margin-bottom: 50px;" >
                
                    Simple Prototype of Shopping Checkout using Object Detection
                </h4>
            </div>

                        """        

        self.st.set_page_config(
            page_title="FastScan Prototype",
            layout="wide"   
        )

        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)
    

    def sidebar(self):
        
        self.st.sidebar.title("Configuration")

        self.source = "webcam"
        self.enable_track = self.st.sidebar.radio("Enable Tracking", ("Yes", "No"))
        self.confidence_threshold = float(self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.confidence_threshold, 0.01))
        self.iou = float(self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))

        self.ann_frame = self.st.empty()

    
    def configure(self):


        with self.st.spinner("Model is loading..."):

            self.model = self.yolo(f"best.pt")
            #class_names = list(self.model.names.values())

        self.yolo_settings.update({"run_dir": "runs"})



    def start_inference(self):

        self.setup_web_user_interface()
        self.sidebar()
        self.configure()


        shopping_item_dict: dict = {
            0: "ECLIPSE PLUS LOQUAT PEAR $3.40",
            1: "HICHEW GRAPE $1.40"
        }

        if self.st.sidebar.button("Start"):
            stop_button = self.st.button("Stop")

            opencv_video_capture_object = self.cv2.VideoCapture(0)

            if not opencv_video_capture_object.isOpened():
                self.st.error("Unable to open webcam")

            while opencv_video_capture_object.isOpened():
                success, frame = opencv_video_capture_object.read()

                if not success:
                    self.st.warning("Failed to read frame from webcam. Please verify the webcam is connected properly.")
                    break
                
                if self.enable_track == "Yes":
                    results = self.model.track(frame, conf=self.confidence_threshold, iou=self.iou, persist=True)
                    results[0].names = shopping_item_dict
                else:
                    results = self.model(frame, conf=self.confidence_threshold, iou=self.iou)
                    results[0].names = shopping_item_dict

                
                annotated_frame = results[0].plot()

                if stop_button:
                    opencv_video_capture_object.release()
                    self.st.stop()

                self.ann_frame.image(annotated_frame, channels="BGR")
        
            opencv_video_capture_object.release()
        self.cv2.destroyAllWindows()
        