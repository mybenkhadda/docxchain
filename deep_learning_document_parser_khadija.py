# Databricks notebook source
# MAGIC %md
# MAGIC ## Deep Learning document parser
# MAGIC using **UNSTRUCTURED** framework

# COMMAND ----------

# MAGIC %md
# MAGIC **Pre-Challenge** : Install UNSTRUCTURED package with all file & connector types

# COMMAND ----------

!pip install "unstructured[all-docs]"
!apt-get -y install poppler-utils
!pip install pdf2image

# COMMAND ----------

# MAGIC %md
# MAGIC **Challenge #1** : Extract data from tables and images from the PDF document

# COMMAND ----------

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.image import partition_image

filename = "./purchasing_contract_example.pdf"

#infer_table_structure=True automatically selects hi_res strategy
elements = partition_pdf(filename=filename, infer_table_structure=True, languages=["eng", "fr"])

tables =[]
titles = []
list_items = []
narrative_texts = []

for element in elements:
    if element.category == "Title":
        titles.append(element)
    elif element.category == "Table":
        tables.append(element.metadata.text_as_html)
    elif element.category == "ListItem":
        list_items.append(element)
    elif element.category == "NarrativeText": 
        narrative_texts.append(element)
    

print([el.category for el in elements])

print(titles[0])
print(list_items[0])
#print(tables[2].text)
print("----------------------------")
print(tables[8])


# COMMAND ----------

!pip install tabula-py
!pip install tabulate

# COMMAND ----------

from tabulate import tabulate
import pandas as pd
from bs4 import BeautifulSoup
html_table=tables[3]
print(html_table)
soup = BeautifulSoup(html_table, 'html.parser')
table = soup.find('table')
df = pd.read_html(str(table))[0]
df = df.ffill()

for col in df.columns:
    first_non_null = df[col].first_valid_index()
    if first_non_null is not None:
        first_value = df.at[first_non_null, col]
        df[col] = df[col].fillna(first_value)


display(df)
print(tabulate(df, df.head()))

# COMMAND ----------

#print(tables[0])

print(titles[0])
print('-------------------------')
print(narrative_texts[0])
print('-------------------------')
print(list_items[0])

# COMMAND ----------

types =[el.category for el in elements]
image=[]
header=[]
uncatego=[]
text_dict = {
    'Title': titles,
    'NarrativeText': narrative_texts,
    'ListItem': list_items,
    'Table': tables,
    'UncategorizedText': uncatego,
    'Header': header,
    'Image' : image
}
type_index = {type_name: 0 for type_name in text_dict}

for type_name in types:
    print(f"--- {type_name} ---")
    
    type_list = text_dict.get(type_name, [])
    
    if type_index[type_name] < len(type_list):
        print(type_list[type_index[type_name]])
        type_index[type_name] += 1
    else:
        print("Not available")

# COMMAND ----------

merged_df =  df.ffill()
display(merged_df)
print(tabulate(merged_df))

# COMMAND ----------

# MAGIC %md
# MAGIC **Challenge #2** : Rearrange detected blocks (especially for PDF with double-column)

# COMMAND ----------

#print(elements[0].to_dict())
titre = [t for t in elements if t.category=="Title"]
print(titre[1])

# COMMAND ----------

print(tables[0])

# COMMAND ----------


