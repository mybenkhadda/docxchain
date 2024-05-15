# Databricks notebook source
# MAGIC %md
# MAGIC ## Deep Learning document parser
# MAGIC using **UNSTRUCTURED** framework

# COMMAND ----------

# MAGIC %md
# MAGIC **Pre-Challenge** : Install UNSTRUCTURED package with all file & connector types

# COMMAND ----------

!pip install "unstructured[all-docs]" pdf2image
!sudo apt-get -y install poppler-utils

# COMMAND ----------

# MAGIC %md
# MAGIC **Challenge #1** : Extract data from tables and images from the PDF document

# COMMAND ----------

# package to extract documents layouts using DL models (title, itemlist, table...)
from unstructured_inference.models.base import get_model
from unstructured_inference.inference.layout import DocumentLayout

# package to extract textual information from layout elements
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.image import partition_image
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import io
import ABC

# COMMAND ----------

# MAGIC %md
# MAGIC **Challenge #2** : Rearrange detected blocks (especially for PDF with double-column) and parse text from them

# COMMAND ----------

MODEL = get_model("yolox")
FILENAME = './documents/purchasing_contract_with_glossary.pdf'
layout = DocumentLayout.from_file(FILENAME, detection_model=model)
total_number_pages = len(layout.pages)
PAGE_NUMBER = 7

#function to reorder the layout extracted
def layout_reordering(document_layout : DocumentLayout = None, page_number : int = 0):
    '''
        Function aims to reorder the layout of a document using
        geometrical property : 1st bbox is the top-left and last bbox is the bottom-right
    inputs:
        document_layout : layout of the document extracted using DL models (yolox, detectron2...)
        page_number : to precise the page to reorder in the document
    output:
        sorted layout of the document page
    '''
        
    x1_min = min([el.bbox.x1 for el in document_layout.pages[page_number].elements])
    x2_max = max([el.bbox.x2 for el in document_layout.pages[page_number].elements])
    mid_line_x_coordinate = (x2_max + x1_min) /  2
    left_column = []
    right_column = []
    for el in document_layout.pages[page_number].elements:
        if el.bbox.x1 < mid_line_x_coordinate:
            left_column.append(el)
        else:
            right_column.append(el)

    left_column.sort(key = lambda z: z.bbox.y1)
    right_column.sort(key = lambda z: z.bbox.y1)
    sorted_layout = left_column + right_column
    return sorted_layout

sorted_page = layout_reordering(document_layout=layout,page_number=PAGE_NUMBER)
# sorted_page

def get_layout_bbox_list(layout_list : list[DocumentLayout] = None):
    '''
        Function aims to extract the bbox coordinates from the layouts list
    inputs:
        layout_list : list of layouts extracted from the document
        page_number : to precify the document page extracted
    output:
        tuple - (list of layout bbox coordinates, page_number)
    '''
    bbox_list = []
    # Get all the bounding box coordinates from the ordered layout list
    for i in range(len(layout_list)):
        box = (*layout_list[i].to_dict()['coordinates'][0],*layout_list[i].to_dict()['coordinates'][2])
        # box_x1, box_y1 = sorted_layout[13].to_dict()['coordinates'][0]
        # box_x2, box_y2 = sorted_layout[13].to_dict()['coordinates'][2]
        bbox_list.append(box)

    return bbox_list

sorted_layout_bbox_list = get_layout_bbox_list(layout_list=sorted_page)

# Store Pdf with convert_from_path function
def visualize_layout(document_path : str = "", bbox_list : list = None, page_number : int = 0):
    '''
        Function aims to visualize the layouts extracted from the document page
    inputs:
        document_path: path toward the document
        bbox_list : list of layouts bbox coordinates
        page_number : to precify the document page to visualize
    output:
        image plots of the extracted elements from the document page
    '''
    images = convert_from_path(document_path)[page_number]
    # Plot all the image extracted with the previous bounding box
    # fig, axs = plt.subplots(len(layout_list))
    for j in range(len(bbox_list)):
        img2 = images.crop(bbox_list[j])
        # axs[j].plot(img2)
        plt.figure(j)
        plt.imshow(img2)

visualize_layout(document_path=FILENAME, bbox_list=sorted_layout_bbox_list, page_number=PAGE_NUMBER)

# Statement to OCRize each image and extract textual information
def img_to_text(document_path : str = "", bbox_list : list = None, page_number : int = 0):
    '''
        Function aims to extract textual information from each
        layout elements from the document page
    inputs:
        document_path: path toward the document
        bbox_list : list of layouts bbox coordinates
        page_number : to precify the document page to visualize
    output:
        image plots of the extracted elements from the document page
    '''
    # convert PDF page into PIL image object
    images = convert_from_path(document_path)[page_number]
    page_textual_elements = []

    # loop through the bbox coordinates list of layout elements
    for idx in range(len(bbox_list)):
        # crop the layout from the image
        cropped_img = images.crop(bbox_list[idx])

        # create a temporary image object from this cropped element
        fp = TemporaryFile()
        cropped_img.save(fp, format="PNG")

        # partition the cropped image to extract textual elements and elements types
        img_elements = partition_image(file=fp, infer_table_structure=True)

        # close the temporary image to free memory storage
        fp.close()

        # add extracted element to the final string list containing the text from the document page
        ## if element is a Table -> transform it into html table to add some structure
        ## else -> join all string elements extracted
        if img_elements[0].category == 'Table':
            page_textual_elements.append(img_elements[0].metadata.text_as_html)
        else:
            full_text = [element.text for element in img_elements]
            page_textual_elements.append(''.join(full_text))
    
    return page_textual_elements

extracted_text_list = img_to_text(document_path=FILENAME, bbox_list=sorted_layout_bbox_list, page_number=PAGE_NUMBER)
extracted_text_list

# COMMAND ----------

# # code snippet to extract text from a specific layout element
# def bbox_to_text(document_path : str = "", bbox_element : tuple = None, page_number : int = 0):
#     images = convert_from_path(document_path)[page_number]
#     cropped_img = images.crop(bbox_element)
#     fp = TemporaryFile()
#     cropped_img.save(fp, format="PNG")
#     bbox_elements = partition_image(file=fp, infer_table_structure=True)
#     fp.close()
#     print(len(bbox_elements))
#     full_text = [element.text for element in bbox_elements]
    
#     return ''.join(full_text)

# bbox_text = bbox_to_text(document_path='./purchasing_contract_example.pdf', bbox_element=sorted_layout_bbox_list[2], page_number=page_number)
# bbox_text
