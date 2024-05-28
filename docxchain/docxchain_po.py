import sys
from pdf2image import convert_from_path
from PIL import Image
import argparse
import os
import numpy as np
import cv2
import datetime
import random
import time
import pytz
import json
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract
from pipelines.document_structurization import DocumentStructurization
from utilities.visualization import *

class DocxChain_PO():
    def __init__(self):

        self.configs = dict()

        layout_analysis_configs = dict()
        layout_analysis_configs['from_modelscope_flag'] = False
        layout_analysis_configs['model_path'] = 'home/DocXLayout_231012.pth'  # note that: currently the layout analysis model is NOT from modelscope
        self.configs['layout_analysis_configs'] = layout_analysis_configs

        text_detection_configs = dict()
        text_detection_configs['from_modelscope_flag'] = True
        text_detection_configs['model_path'] = 'damo/cv_resnet18_ocr-detection-line-level_damo'
        self.configs['text_detection_configs'] = text_detection_configs

        text_recognition_configs = dict()
        text_recognition_configs['from_modelscope_flag'] = True
        text_recognition_configs['model_path'] = 'damo/cv_convnextTiny_ocr-recognition-document_damo'  # alternatives: 'damo/cv_convnextTiny_ocr-recognition-scene_damo', 'damo/cv_convnextTiny_ocr-recognition-general_damo', 'damo/cv_convnextTiny_ocr-recognition-handwritten_damo'
        self.configs['text_recognition_configs'] = text_recognition_configs

        formula_recognition_configs = dict()
        formula_recognition_configs['from_modelscope_flag'] = False
        formula_recognition_configs['image_resizer_path'] = 'home/LaTeX-OCR_image_resizer.onnx'
        formula_recognition_configs['encoder_path'] = 'home/LaTeX-OCR_encoder.onnx'
        formula_recognition_configs['decoder_path'] = 'home/LaTeX-OCR_decoder.onnx'
        formula_recognition_configs['tokenizer_json'] = 'home/LaTeX-OCR_tokenizer.json'
        self.configs['formula_recognition_configs'] = formula_recognition_configs
        os.environ['TESSDATA_PREFIX'] = 'home/tessdata-main/'
        random.seed(1)
        self.document_structurizer = DocumentStructurization(self.configs)

        if "tables" not in os.listdir("./"):
            os.mkdir("./tables")

    def pdf2image(self, pdf_path: str):
        """
        Convert a PDF file to a list of images.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            List[np.ndarray]: A list of images, where each image is a numpy array.
        """
        images = convert_from_path(pdf_path)

        for i in range(len(images)):

            images[i] = np.array(images[i])

        return images
    
    def extractFooter(self, image: np.ndarray):
        """
        Crops the footer from a given image.
    
        Args:
            image (np.ndarray): The input image.
    
        Returns:
            np.ndarray: The cropped image without the footer.
            np.ndarray: The cropped footer.
        """
        image = image[95:2200, :, :]
        footer = image[2200:, :, :]
        return image, footer

    
    def merge_regions(self, results):
        """
        Merge regions that have the same category on adjacent pages.

        Args:
            results (list): A list of dictionaries, where each dictionary represents
                the output of a single page of layout analysis.

        Returns:
            list: The updated list of dictionaries.
            list: A list of dictionaries, where each dictionary represents a region that
                spans multiple pages. Each dictionary contains the "page" and "region_poly"
                keys.
        """
        regions = []
        for page in range(len(results) - 1):
            if results[page]["information"][-1]["category_name"] == results[page + 1]["information"][0]["category_name"]:
                if results[page]["information"][-1]["category_name"] == "table":
                    regions.append({
                        "page": page,
                        "region_poly1": results[page]["information"][-1]["region_poly"],
                        "region_poly2": results[page + 1]["information"][0]["region_poly"],
                    })
                    results[page]["information"][-1] = {}
                    results[page + 1]["information"][0] = {}

        return results, regions
    
    def order_file(self, results):
        """
        Sort the list of results by their y-coordinate.

        Args:
            results (list): A list of dictionaries, where each dictionary represents
                the output of a single page of layout analysis.

        Returns:
            list: The sorted list of dictionaries.
        """
        results.sort(key=lambda x: (x["region_poly"][1]))
        return results
    
    def structure1image(self, image: np.ndarray) -> list:
        """
        This function takes an image as input and applies the layout analysis and document structurization pipeline to it.

        Args:
            image (np.ndarray): The input image.

        Returns:
            list: A list of dictionaries, where each dictionary represents the output of a single page of layout analysis.
        """
        final_result = self.document_structurizer(image)

        final_result = self.order_file(final_result)

        return final_result
    
    def document_structure(self, path):
        """
        This function takes a pdf as input and applies the layout analysis and document structurization pipeline to it.

        Args:
            path (str): The path to the PDF file.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents the output of a single page of layout analysis.
        """
        images = self.pdf2image(path)
        images = [self.extractFooter(image)[0] for image in images]

        doc = [
            {
                "page": 0,
                "content": self.structure1image(images[0])
            }
        ]

        if len(images) > 1:
            for i in range(1, len(images)):
                tmp = self.structure1image(images[i])
                if doc[i-1]["content"][-1]["category_name"] == tmp[0]["category_name"] and doc[i-1]["content"][-1]["category_name"] == "table":
                    img1 = images[i-1][doc[i-1]["content"][-1]["region_poly"][1]:doc[i-1]["content"][-1]["region_poly"][-1],:,:]
                    img2 = images[i][tmp[0]["region_poly"][1]:tmp[0]["region_poly"][-1],:,:]

                    concatenated_img = np.concatenate((img1, img2), axis=0)

                    plt.imsave(f"tables/table-{random.randint(0,100)}-page{i}.jpg", concatenated_img)

                    doc[i-1]["content"] = doc[i-1]["content"][:-2]



                    doc.append(
                        {
                            "page": i,
                            "content": self.structure1image(images[i])[1:]
                        }
                    )

                else:
                    doc.append({
                        "page": i,
                        "content": self.structure1image(images[i])
                    })

        return(doc)
    
    def json2df(self, jsonData):
        """
        This function takes a list of dictionaries as input and converts it to a pandas dataframe.

        Parameters:
            jsonData (list): A list of dictionaries, where each dictionary represents the output of a single page of layout analysis.

        Returns:
            pd.DataFrame: A pandas dataframe containing the input data.
        """
        data = []
        for i in range(len(jsonData)):
            page = jsonData[i]["content"]
            for j in range(len(page)):
                region_poly = page[j]["region_poly"]
                category_name = page[j]["category_name"]
                content = " ".join([page[j]["text_list"][k]["content"][0] for k in range(len(page[j]["text_list"]))])
                page_number = i
                data.append(
                    {
                        "region": region_poly,
                        "category_name": category_name,
                        "content": content,
                        "page": int(page_number)
                    }
                )
        return pd.DataFrame(data, columns = data[0].keys())
    
    def extractTables(self, pdf, df):
        """
        This function takes a PDF file and a pandas dataframe as input, and extracts all the tables from the PDF and saves them as images in the "tables" directory.

        Args:
            pdf (str): The path to the PDF file.
            df (pd.DataFrame): A pandas dataframe containing the output of the layout analysis pipeline.

        Returns:
            None: This function does not return any additional values.
        """
        images = self.pdf2image(pdf)
        tables = df[df["category_name"] == "table"]
        l = []
        for i in range(tables.shape[0]):
            region = tables.iloc[i]["region"]
            page = int(tables.iloc[i]["page"])
            print(region, page)
            img = images[page]
            img = img[int(region[1])-5:int(region[2])+20, :,:]

            data = pytesseract.image_to_data(
                        img, 
                        lang="eng+fra",
                        output_type='data.frame', 
                        config='--psm 12 --oem 1')
            
            l.append(data)
            plt.imsave(f"tables/table-{region}-page{page}.jpg", img)
            plt.imshow(img)
            plt.show()
        return l