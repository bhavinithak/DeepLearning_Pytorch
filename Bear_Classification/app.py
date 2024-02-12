from fastai.vision.all import *
import gradio as gr

learn = load_learner('model.pkl')

categories =('grizzly','black','teddy')

def classify_image(img):
  pred,idx,probs=learn.predict(img)
  return dict(zip(categories,map(float,probs)))

image=gr.Image(height=192,width=192)
label=gr.components.Label()
examples=['grizzly.jpg','black.jpeg','teddy.jpeg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)