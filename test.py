import json
import logging
import matplotlib.pyplot as plt
import networkx as nx
from networkx import connected_components
from openai import OpenAI
import pandas as pd
from sentence_transformers import SentenceTransformer, util


data = pd.read_csv("amazon_products.csv")
data['text'] = data['TITLE'] + data['BULLET_POINTS'] + data['DESCRIPTION']


# ENTITY TYPES:
entity_types = {
  "product": "https://schema.org/Product", 
  "rating": "https://schema.org/AggregateRating",
  "price": "https://schema.org/Offer", 
  "characteristic": "https://schema.org/PropertyValue", 
  "material": "https://schema.org/Text",
  "manufacturer": "https://schema.org/Organization", 
  "brand": "https://schema.org/Brand", 
  "measurement": "https://schema.org/QuantitativeValue", 
  "organization": "https://schema.org/Organization",  
  "color": "https://schema.org/Text",
}

# RELATION TYPES:
relation_types = {
  "hasCharacteristic": "https://schema.org/additionalProperty",
  "hasColor": "https://schema.org/color", 
  "hasBrand": "https://schema.org/brand", 
  "isProducedBy": "https://schema.org/manufacturer", 
  "hasColor": "https://schema.org/color",
  "hasMeasurement": "https://schema.org/hasMeasurement", 
  "isSimilarTo": "https://schema.org/isSimilarTo", 
  "madeOfMaterial": "https://schema.org/material", 
  "hasPrice": "https://schema.org/offers", 
  "hasRating": "https://schema.org/aggregateRating", 
  "relatedTo": "https://schema.org/isRelatedTo"
 }

client = OpenAI(api_key="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")

def extract_information(text, model="gpt-3.5-turbo"):
   completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt.format(
              entity_types=entity_types,
              relation_types=relation_types,
              specification=text
            )
        }
        ]
    )

   return completion.choices[0].message.content

system_prompt = """You are an expert agent specialized in analyzing product specifications in an online retail store.
Your task is to identify the entities and relations requested with the user prompt, from a given product specification.
You must generate the output in a JSON containing a list with JOSN objects having the following keys: "head", "head_type", "relation", "tail", and "tail_type".
The "head" key must contain the text of the extracted entity with one of the types from the provided list in the user prompt, the "head_type"
key must contain the type of the extracted head entity which must be one of the types from the provided user list,
the "relation" key must contain the type of relation between the "head" and the "tail", the "tail" key must represent the text of an
extracted entity which is the tail of the relation, and the "tail_type" key must contain the type of the tail entity. Attempt to extract as
many entities and relations as you can.
"""

user_prompt = """Based on the following example, extract entities and relations from the provided text.
Use the following entity types:

# ENTITY TYPES:
{entity_types}

Use the following relation types:
{relation_types}

--> Beginning of example

# Specification
"YUVORA 3D Brick Wall Stickers | PE Foam Fancy Wallpaper for Walls,
 Waterproof & Self Adhesive, White Color 3D Latest Unique Design Wallpaper for Home (70*70 CMT) -40 Tiles
 [Made of soft PE foam,Anti Children's Collision,take care of your family.Waterproof, moist-proof and sound insulated. Easy clean and maintenance with wet cloth,economic wall covering material.,Self adhesive peel and stick wallpaper,Easy paste And removement .Easy To cut DIY the shape according to your room area,The embossed 3d wall sticker offers stunning visual impact. the tiles are light, water proof, anti-collision, they can be installed in minutes over a clean and sleek surface without any mess or specialized tools, and never crack with time.,Peel and stick 3d wallpaper is also an economic wall covering material, they will remain on your walls for as long as you wish them to be. The tiles can also be easily installed directly over existing panels or smooth surface.,Usage range: Featured walls,Kitchen,bedroom,living room, dinning room,TV walls,sofa background,office wall decoration,etc. Don't use in shower and rugged wall surface]
Provide high quality foam 3D wall panels self adhesive peel and stick wallpaper, made of soft PE foam,children's collision, waterproof, moist-proof and sound insulated,easy cleaning and maintenance with wet cloth,economic wall covering material, the material of 3D foam wallpaper is SAFE, easy to paste and remove . Easy to cut DIY the shape according to your decor area. Offers best quality products. This wallpaper we are is a real wallpaper with factory done self adhesive backing. You would be glad that you it. Product features High-density foaming technology Total Three production processes Can be use of up to 10 years Surface Treatment: 3D Deep Embossing Damask Pattern."

################

# Output
[
  {{
    "head": "YUVORA 3D Brick Wall Stickers",
    "head_type": "product",
    "relation": "isProducedBy",
    "tail": "YUVORA",
    "tail_type": "manufacturer"
  }},
  {{
    "head": "YUVORA 3D Brick Wall Stickers",
    "head_type": "product",
    "relation": "hasCharacteristic",
    "tail": "Waterproof",
    "tail_type": "characteristic"
  }},
  {{
    "head": "YUVORA 3D Brick Wall Stickers",
    "head_type": "product",
    "relation": "hasCharacteristic",
    "tail": "Self Adhesive",
    "tail_type": "characteristic"
  }},
  {{
    "head": "YUVORA 3D Brick Wall Stickers",
    "head_type": "product",
    "relation": "hasColor",
    "tail": "White",
    "tail_type": "color"
  }},
  {{
    "head": "YUVORA 3D Brick Wall Stickers",
    "head_type": "product",
    "relation": "hasMeasurement",
    "tail": "70*70 CMT",
    "tail_type": "measurement"
  }},
  {{
    "head": "YUVORA 3D Brick Wall Stickers",
    "head_type": "product",
    "relation": "hasMeasurement",
    "tail": "40 tiles",
    "tail_type": "measurement"
  }},
  {{
    "head": "YUVORA 3D Brick Wall Stickers",
    "head_type": "product",
    "relation": "hasMeasurement",
    "tail": "40 tiles",
    "tail_type": "measurement"
  }}
]

--> End of example

For the following specification, generate extract entitites and relations as in the provided example.

# Specification
{specification}
################

# Output

"""

kg = []
for content in data['text'].values[:100]:
  try:
    extracted_relations = extract_information(content)
    extracted_relations = json.loads(extracted_relations)
    kg.extend(extracted_relations)
  except Exception as e:
    logging.error(e)

kg_relations = pd.DataFrame(kg)

heads = kg_relations['head'].values
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(heads)
similarity = util.cos_sim(embeddings[0], embeddings[1])

G = nx.Graph()
for _, row in kg_relations.iterrows():
  G.add_edge(row['head'], row['tail'], label=row['relation'])
  
pos = nx.spring_layout(G, seed=47, k=0.9)
labels = nx.get_edge_attributes(G, 'label')
plt.figure(figsize=(15, 15))
nx.draw(G, pos, with_labels=True, font_size=10, node_size=700, node_color='lightblue', edge_color='gray', alpha=0.6)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8, label_pos=0.3, verticalalignment='baseline')
plt.title('Product Knowledge Graph')
plt.show()