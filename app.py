import cv2
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login 
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

login(token = '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')

model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline('text-generation',model = model, tokenizer = tokenizer)
def Load_tumor_classifier ():
  # Load the model
  model = tf.keras.models.load_model('Brain Tumor.h5')
  return model

def annotator(img, class_label):
  pt1 = (50, 45)
  text = class_label
  text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
  text_w, text_h = text_size[0]
 # Draw the rectangle
  cv2.line(img, pt1, (text_w+50, 45), (255,255,255), 25)
  cv2.putText(img, text, (50, 50),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,0,0), 2)


def run_inference(img):
  global class_label
  classes = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']
  model = Load_tumor_classifier()
  # img_bgr = cv2.imread(img)
  # print(img_bgr.shape)
  if img is None:
    print("Error: Unable to load image.")
  else:
      # Convert BGR to RGB
      rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      rgb_image = cv2.resize(rgb_image, (224,224))
      rgb_image = np.expand_dims(rgb_image, axis=0)


  output = model.predict(rgb_image)
  max_index = np.argmax(output)
  class_label = classes[max_index]
  annotator(img, class_label)

  return img
def getResponse (user_message):

  brain_tumor_info = {
    'Glioma': {
        'description': "Gliomas are a type of brain tumor that originate from glial cells in the brain and spinal cord. They can be benign or malignant.",
        'symptoms': "Symptoms of gliomas can vary depending on their location and size, but common symptoms include headaches, seizures, cognitive difficulties, and changes in personality or mood.",
        'prevention': "There are no known ways to prevent gliomas. However, avoiding exposure to radiation and certain chemicals may reduce the risk.",
        'treatment': "Treatment options for gliomas may include surgery, radiation therapy, chemotherapy, targeted therapy, and immunotherapy, depending on the type and location of the tumor and the overall health of the patient."
    },
    'Meningioma': {
        'description': "Meningiomas are tumors that arise from the meninges, the protective membranes surrounding the brain and spinal cord. They are usually benign.",
        'symptoms': "Meningiomas may not cause symptoms until they grow large enough to press on surrounding structures. Common symptoms include headaches, seizures, vision problems, and changes in behavior.",
        'prevention': "There are no known ways to prevent meningiomas. Avoiding exposure to radiation may reduce the risk.",
        'treatment': "Treatment options for meningiomas may include observation, surgery, radiation therapy, and sometimes chemotherapy or targeted therapy, depending on the size and location of the tumor and the patient's overall health."
    },
    'No tumor': {
        'description': "No tumor indicates the absence of any abnormal growth or mass in the brain.",
        'symptoms': "As there is no tumor present, symptoms associated with brain tumors would not be present.",
        'prevention': "There are no specific prevention measures for 'No tumor', as it denotes the absence of abnormal growth.",
        'treatment': "No treatment is required for 'No tumor', as it denotes the absence of abnormal growth."
    },
    'Pituitary': {
        'description': "Pituitary tumors are growths that develop in the pituitary gland, a small gland located at the base of the brain. They can be benign or malignant.",
        'symptoms': "Symptoms of pituitary tumors can vary depending on their type and size, but may include headaches, vision problems, hormonal imbalances, and symptoms related to pressure on surrounding structures.",
        'prevention': "There are no known ways to prevent pituitary tumors. Regular check-ups and early detection may help in better management.",
        'treatment': "Treatment options for pituitary tumors may include surgery, radiation therapy, medication to control hormone levels, and sometimes hormone replacement therapy, depending on the type and size of the tumor and the patient's symptoms."
    }
}
  context_info =  brain_tumor_info[class_label]

  conversation_template  = f"""
                you are a brain tumor medical expert and assistant conversational assistant that help doctors and patients in brain tumor classification informations.
                you are provided with some information about the type of tumor diagnosed.

                The doctor or patiebt will ask some questions based on this and you must use your own knowledge to answer those questions.

              Information on the tumor diagnosed.
              Context: {context_info}

              Tumor detected:
                {class_label}

              user message: {user_message}
            """

  #initialize the embedchain bot

  response = generator(conversation_template, max_new_tokens = 200)[0]['generated_text']

  return response

image_input = gr.Image(label="Upload Image", height = 500, width= 700)
# print(image_input)
object_detection_output = gr.Image(label="Tumor Classification", height = 500, width= 700)
chat_input = gr.Textbox(lines=2, label="Ask a question")
chat_output = gr.Textbox(label="Chatbot Response")

title = "Brain Tumor Classification"
title1 = "Chatbot"
description = "Upload a MRI image, classify the tumor and ask questions about the tumor detected."

detection_interface = gr.Interface(
    fn= run_inference,
    inputs=image_input,
    outputs=object_detection_output,
    title=title,
    description=description,
    allow_flagging='never'  # Disable flagging for simplicity
)

chat_interface = gr.Interface(
    fn=getResponse,
    inputs=chat_input,
    outputs=chat_output,
    title=title1,
    description="Ask questions about the tumor.",
    allow_flagging='never'  # Disable flagging for simplicity
)

demo = gr.TabbedInterface([detection_interface, chat_interface], ["Brain Tumor Classification", "Chat with data"])
demo.launch(share=True, debug = True)