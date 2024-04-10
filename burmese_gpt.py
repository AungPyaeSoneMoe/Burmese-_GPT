import streamlit as st
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load MyanmarGPT-Chat model and tokenizer
model = GPT2LMHeadModel.from_pretrained("jojo-ai-mst/MyanmarGPT-Chat")
tokenizer = GPT2Tokenizer.from_pretrained("jojo-ai-mst/MyanmarGPT-Chat")
def generate_text(prompt, max_length=300, temperature=0.8, top_k=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")# remove .cude() if only cpu
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )
    for result in output:
        generated_text = tokenizer.decode(result, skip_special_tokens=True)
        print(generated_text)
    return generated_text
# def question(text):
#   return f"User : {text} \n Assistant :"
st.title("Burmese GPT")
st.divider()
multi = '''
- It is lightweight and free for everything .
- Made by and Follow us on Facebok [Aung Pyae Sone Moe ](https://www.facebook.com/profile.php?id=100053733635679).
- References Data and Dataset From [MinSiThu](https://www.linkedin.com/in/min-si-thu/)
    - (https://huggingface.co/jojo-ai-mst/MyanmarGPT?fbclid=IwAR2zkHpHvbC-xOC5YWrajPaI45IrWnGFm-3SypJhQ5yEjSTLCgs1dRr63es_aem_ATVkzaTcT1sbBhsS1GXV-Eg4b3b2GQgScfweT3FK1VVb_zTFRrouX8_qRZsY7VY-g9STj1z_QvvIwqmTL2I2g0TL)
'''
mark = st.markdown(multi)
input_text = st.chat_input("ဥပမာ 3လနဲ့Programmerတစ်ယောက်ဖစ်ဖို့လွယ်လား ။") 

user_container = st.container()
assistant_container = st.empty()
markdown_container = st.empty()
if input_text:
    mark.empty()
    user_container.chat_message("user").write(input_text)
    with st.spinner("Processing..."):
        result = generate_text(input_text)
    assistant_container.chat_message("assistant").write (result)
    