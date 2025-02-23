from typing import Any

class StreamlitUserInterface:

    def __init__(self, **kwargs: Any):
        import streamlit as st

        self.st = st
        self.model = None

        self.confidence_threshold = 0.25
        self.iou = 0.45
        self.model_path = None # 


    def setup_web_user_interface(self) -> None:

        '''
            Setup the Streamlit Web interface with custom HTML elements
        '''

        menu_style_cfg = """<style>MainMenu {visibility: hidden;} </style>"""

        main_title_cfg = """
            <div>
                <h1 style="color:#FF64DA; text-align:center; 
                            font-size:40px; 
                            margin-top:-50px;
                            font-family: 'Archivo', san-serif; margin-bottom:20px;"
                >
                    Shopping Checkout System
                </h1>

            </div>
                        """


        sub_title_cfg = """

            <div>
                <h4 style="color:#042AFF; 
                            text-align:center; 
                            font-family: 'Archivo', sans-serif;
                            margin-top:-15px;
                            margin-bottom: 50px;">
                
                    Simple Prototype of Shopping Checkout using Object Detection
                </h4>
            </div>

                        """        

        self.st.set_page_config(
            page_title="Shopping Checkout System prototype"   
        )
    


    def start_inference(self):

        self.setup_web_user_interface()


