import os
from flask import Flask, request, jsonify, render_template
import random
from mykey import key
import requests
from bs4 import BeautifulSoup
os.environ["HUGGINGFACEHUB_API_TOKEN"] = key
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from huggingface_hub import notebook_login
from langchain.llms import HuggingFacePipeline
#from langchain.llms import HuggingFaceHub
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
#from langchain.memory import ConversationBufferMemory
from langdetect import detect
from translate import Translator
import wikipediaapi
checkpoint = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype=torch.float32)

DB_FAISS_PATH = 'myVectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, provide a general response.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


app = Flask(__name__)

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])

    return prompt

#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain


def load_llm():
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.7, "max_length": 2048})
    #llm=OpenAI()
    #return llm
    pipe = pipeline('text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample=True,
        temperature = 0.3,
        top_p = 0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa


@app.route('/')
def index():
    greeting = "Welcome to the Chatbot App! How can I assist you today?"
    return render_template('index.html', greeting=greeting)

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        query = data['query']
        
        if query.lower() in ["hi", "hello", "hey","Is anyone there?","good day","what's up","heyy","how are you","whatsupp"]:
            responses = ["Hello!","Good to see you again!","Hi there, how can I help?","Hello, How can I help you?","Assalamualikum"]
            response = random.choice(responses)
            #response = "Hello! How can I help you?"
        elif query.lower() in ["what are acnes?","what are acnes","what is acnes","what is acne","what is acne?" ,"tell me about acnes", "acne information", "about pimples","what is acne","what is acnes"]:
            responses=["Acne, also known as pimples or zits, is a common skin condition that occurs when hair follicles are clogged with oil and dead skin cells. It often results in the formation of red and swollen bumps on the skin.", "Acne is a skin problem characterized by the appearance of pimples. It can affect people of all ages and is often associated with hormonal changes.", "Acne is a skin disorder that usually occurs in teenagers, but it can affect adults as well. It can be caused by various factors, including hormonal changes, genetics, and diet."]
            response = random.choice(responses)
        elif query.lower() in  ["what causes acnes?","what causes acnes" ,"what causes acne","why do I get pimples?", "acne triggers", "factors leading to acne", "what causes pimples?","what are the causes of acne"," reason for acne","what are the reasons for acne"]:
            responses= ["Acne can be caused by a combination of factors, including excess oil production, clogged hair follicles, bacteria, and hormonal changes. Genetics and diet may also play a role in its development.", "The exact cause of acne is not always clear, but it is believed to be related to the overproduction of oil (sebum) in the skin, which can lead to clogged pores and the growth of acne-causing bacteria.", "Common causes of acne include hormonal fluctuations during puberty, a family history of acne, and the use of certain skincare products or medications."]
            response = random.choice(responses)
        elif query.lower() in  ["how to treat acne?","how to treat acne","how to treat acnes" ,"acne remedies", "clearing pimples", "acne solutions", "how to get rid of pimples?", "how to get rid of acne?", "how to remove pimples?", "how to remove acne?", "how to prevent pimples?", "how to prevent acne?"]:
            responses= ["There are several ways to treat acne, including over-the-counter topical treatments, prescription medications, and lifestyle changes. It's important to consult with a dermatologist to determine the best treatment plan for your specific case.", "Some common acne treatments include using benzoyl peroxide or salicylic acid creams, taking oral antibiotics, or undergoing light and laser therapies. Maintaining a healthy diet and skincare routine can also help prevent and manage acne.", "If you have acne, it's important to keep your skin clean, avoid picking or squeezing pimples, and use non-comedogenic (non-pore-clogging) skincare products."]
            response = random.choice(responses)
        elif query.lower() in ["how to maintain a healthy lifestyle?", "tips for staying healthy", "healthy living", "ways to be healthy"]:
            responses = ["Maintaining a healthy lifestyle involves eating a balanced diet, staying physically active, getting enough sleep, managing stress, and avoiding harmful habits like smoking and excessive alcohol consumption.", "To stay healthy, make sure to eat a variety of fruits and vegetables, exercise regularly, drink plenty of water, and prioritize mental well-being. Regular check-ups with healthcare professionals are also essential for preventive care.", "A healthy lifestyle is essential for overall well-being. It includes eating nutritious foods, staying active, getting enough rest, and finding ways to cope with stress."]
            response = random.choice(responses)
        elif query.lower() in ["what causes health issues?","what causes health issues", "why do people get sick?", "health problems triggers", "factors leading to health issues", "how to stay healthy?", "maintaining a healthy lifestyle causes"]:
            response = "Health issues can have various causes, including genetic factors, lifestyle choices, environmental factors, and exposure to pathogens. Staying healthy involves adopting a balanced lifestyle with good nutrition, exercise, and preventive care."
        elif query.lower() in ["how to improve health?","how to improve health" ,"health improvement remedies", "tips for better health", "health solutions", "how to prevent health issues?", "how to stay disease-free?", "how to boost immunity?"]:
            response = "To improve health, focus on a nutritious diet, regular physical activity, adequate sleep, stress management, and vaccination when necessary. Consult with healthcare professionals for personalized guidance."
        elif query.lower() in ["what is vitiligo?","what is vitiligo", "tell me about vitiligo.", "vitiligo information", "about skin depigmentation", "about skin condition vitiligo", "about skin disorder vitiligo", "about skin disease vitiligo", "about skin problem vitiligo"]:
            responses =  ["Vitiligo is a long-term skin condition characterized by the loss of skin color in patches. It occurs when melanocytes, the cells responsible for producing skin pigment (melanin), are destroyed or stop functioning. This leads to the development of white patches on the skin.", "Vitiligo is not contagious and can affect people of all skin types. While the exact cause is not known, it is believed to involve a combination of genetic, autoimmune, and environmental factors."]
            response = random.choice(responses)
        elif query.lower() in ["what causes vitiligo?", "what causes vitiligo","why do people get vitiligo?", "vitiligo triggers", "factors leading to vitiligo", "what causes skin depigmentation?", "why do people get skin depigmentation?", "skin depigmentation triggers", "factors leading to skin depigmentation"]:
            responses = ["The exact cause of vitiligo is not well understood, but it is thought to involve a combination of genetic predisposition and autoimmune factors. In autoimmune conditions, the immune system mistakenly attacks and destroys melanocytes, leading to depigmentation of the skin.", "Some researchers believe that environmental factors, such as exposure to certain chemicals or stress, may also play a role in the development of vitiligo.", "Vitiligo is not directly linked to diet or lifestyle choices."]
            response = random.choice(responses)
        elif query.lower() in ["how to treat vitiligo?","how to treat vitiligo", "vitiligo remedies", "managing skin depigmentation", "vitiligo solutions"]:
            responses =  ["Fungal infections are caused by fungi that can invade and grow on the skin, nails, hair, or mucous membranes. These infections can vary in severity and may result in symptoms such as itching, redness, rash, or peeling skin.", "Common fungal infections include athlete's foot, ringworm, jock itch, and yeast infections. Fungal infections are contagious and can spread from person to person or through contact with contaminated objects or surfaces."]
            response = random.choice(responses)
        elif query.lower() in ["what are fungal infections?","what are fungal infections", "tell me about fungal infections.", "fungal infections information", "about fungal skin conditions"]:
            response = "Fungal infections are caused by fungi that can invade and grow on the skin, nails, hair, or mucous membranes."
        elif query.lower() in ["what causes fungal infections?","what causes fungal infections", "why do fungal infections occur?", "fungal infection triggers", "factors leading to fungal skin conditions", "what causes fungal skin conditions?", "why do fungal skin conditions occur?", "fungal skin condition triggers", "factors leading to fungal skin conditions"]:
            responses = ["Fungal infections are typically caused by various types of fungi, such as dermatophytes and yeasts. These fungi thrive in warm, moist environments and can infect the skin when there is excessive sweating, poor hygiene, or prolonged exposure to moisture.", "Fungal infections can also occur when the skin's natural barrier is compromised, making it easier for fungi to penetrate and cause an infection.", "Certain medical conditions, like diabetes or a weakened immune system, can increase the risk of fungal infections."]
            response = random.choice(responses)
        elif query.lower() in ["how to treat fungal infections?","how to treat fungal infections", "fungal infection remedies", "managing fungal skin conditions", "fungal infection solutions", "how to get rid of fungal infections?", "how to get rid of fungal skin conditions?", "how to remove fungal infections?", "how to remove fungal skin conditions?", "how to prevent fungal infections?", "how to prevent fungal skin conditions?"]:
            responses = ["Fungal infections are typically treated with antifungal medications, which can be applied topically or taken orally, depending on the severity of the infection. It's essential to keep the affected area clean and dry during treatment to prevent the recurrence of fungal infections.", "Over-the-counter antifungal creams, powders, or sprays can help relieve symptoms of mild fungal infections, but prescription medications may be needed for more severe cases.", "To prevent fungal infections, practice good hygiene, wear breathable fabrics, and avoid sharing personal items with others who may be infected."]
            response = random.choice(responses)        
        elif query.lower() in ["what is eczema?", "what is eczema","tell me about eczema", "eczema information", "about skin condition eczema", "about skin disorder eczema", "about skin disease eczema", "about skin problem eczema"]:
            response = "Eczema, also known as atopic dermatitis, is a chronic skin condition characterized by inflammation, itching, and redness of the skin. It can affect people of all ages and often occurs in individuals with a family history of allergic conditions like asthma and hay fever."
        elif query.lower() in ["what causes eczema?","what causes eczema", "why do people get eczema?", "eczema triggers", "factors leading to eczema", "what causes skin condition eczema?", "why do people get skin condition eczema?", "skin condition eczema triggers", "factors leading to skin condition eczema"]:
            responses = ["The exact cause of eczema is not fully understood, but it is believed to be a combination of genetic, environmental, and immune system factors. People with a family history of eczema or other allergic conditions may have a higher risk of developing it.", "Eczema often occurs due to an overactive immune response to certain triggers. Common triggers include allergens (like pollen, dust mites, and pet dander), irritants (such as soaps, detergents, and perfumes), extreme weather conditions, and stress.", "Skin barrier dysfunction is also thought to play a role in eczema, as individuals with eczema may have a compromised skin barrier that allows moisture to escape, leading to dry and inflamed skin."]
            response = random.choice(responses)
        elif query.lower() in ["how to manage eczema?", "how to manage eczema","eczema remedies", "eczema treatment", "relief for eczema", "eczema solutions", "how to get rid of eczema?", "how to remove eczema?", "how to prevent eczema?"]:
            responses =["Managing eczema involves a combination of preventive measures and treatment. Here are some remedies and tips for managing eczema:",
            "1. Moisturize regularly: Keep your skin well-hydrated by applying a hypoallergenic moisturizer after bathing or showering.",
            "2. Avoid triggers: Identify and avoid substances that trigger your eczema symptoms, such as certain fabrics, harsh soaps, and specific foods.",
            "3. Use gentle skincare products: Choose fragrance-free and mild skincare products to minimize skin irritation.",
            "4. Bathe in lukewarm water: Hot water can dry out the skin, so opt for lukewarm baths or showers. Limit bathing time to avoid excessive dryness.",
            "5. Pat dry, don't rub: After bathing, gently pat your skin with a soft towel instead of vigorously rubbing it.",
            "6. Prescription treatments: In severe cases, your dermatologist may prescribe topical corticosteroids, immunosuppressants, or other medications to manage inflammation and itching.",
            "7. Wet wrap therapy: This involves applying a moisturizer or medication and wrapping the affected areas in wet bandages to promote skin hydration.",
            "8. Antihistamines: These can help reduce itching and improve sleep quality if eczema-related itching keeps you awake at night.",
            "9. Stress management: Practice stress-reduction techniques, such as meditation or yoga, as stress can exacerbate eczema symptoms.",
            "10. Consult a dermatologist: If your eczema is severe or not responding to over-the-counter treatments, seek advice and treatment options from a dermatologist."]
            response = random.choice(responses)
        elif query.lower() in ["what is melanoma skin cancer?","what is melanoma skin cancer", "tell me about melanoma and moles.", "melanoma information", "about skin cancer and moles", "about skin condition melanoma", "about skin disorder melanoma", "about skin disease melanoma", "about skin problem melanoma"]:
            response = "Melanoma is a type of skin cancer that originates in melanocytes, the cells responsible for producing melanin (skin pigment). It is the most dangerous form of skin cancer and can be life-threatening if not detected and treated early."
        elif query.lower() in ["what causes melanoma skin cancer?","what causes melanoma skin cancer" ,"why do people get melanoma?", "melanoma triggers", "factors leading to melanoma", "what causes moles?", "why do people get moles?", "mole triggers", "factors leading to moles"]:
            responses = ["The primary cause of melanoma is exposure to ultraviolet (UV) radiation from the sun or artificial sources like tanning beds. UV radiation can damage the DNA in skin cells, leading to the development of cancerous cells.", "Genetics can also play a role in the development of melanoma. People with a family history of melanoma may have an increased risk of the disease.", "Moles are generally caused by the growth of melanocytes in clusters rather than a specific external factor."]
            response = random.choice(responses)
        elif query.lower() in ["how to prevent melanoma?","how to prevent melanoma", "melanoma and moles remedies", "skin cancer prevention", "mole management", "melanoma solutions", "how to get rid of moles?", "how to remove moles?", "how to prevent moles?"]:
            responses = ["Preventing melanoma involves protecting your skin from excessive sun exposure by using sunscreen, wearing protective clothing, and avoiding tanning beds. Regular skin examinations by a dermatologist can help detect melanoma early.", "For moles, it's important to monitor them for any changes in size, shape, or color. If you notice suspicious changes, consult a dermatologist for a thorough evaluation.", "Treatment for melanoma may involve surgical removal, chemotherapy, radiation therapy, or immunotherapy, depending on the stage of the cancer."]
            response = random.choice(responses)       
        elif query.lower() in ["bye", "goodbye", "end"]:
            response = "Goodbye! If you have more questions, feel free to ask later."
        elif query.lower().startswith("search "):
            parts = query.lower().split(" ", 1)  
            if len(parts) == 2:
                search_term = parts[1] 
                user_agent = "Arsh"  # Replace with an appropriate user agent
                url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch={search_term}"
                headers = {
                    "User-Agent": user_agent
                }
                response = requests.get(url, headers=headers)
                search_results = response.json()

                if "query" in search_results and "search" in search_results["query"]:
                    search_results_list = search_results["query"]["search"]
                    if search_results_list:
                        result_text=""
                        for result in search_results_list:
                            title = result["title"]
                            snippet = result["snippet"]
                            soup = BeautifulSoup(snippet, "html.parser")
                            snippet_text = soup.get_text()
                            snippet_text = snippet_text.replace("Snippet:", "")
                            result_text += f"{snippet_text}\n\n"
                        response = result_text
                    else:
                        response = f"No search results found for '{search_term}'"
                else:
                    response = f"Error performing Wikipedia search for '{search_term}'"
            else:
                response = "Please provide a search term after 'search'."
                
        else:
            detected_language = detect(query)
            print(detected_language)
        
            if (detect(query)!="en"):
               # print(detected_language)
                translator = Translator(to_lang='en', from_lang=detected_language)
                translate_text = translator.translate(query)
                query=translate_text
                print(query)
                qa_result = qa_bot()
                response = qa_result({'query': query})

                if 'result' in response:
                    englishans = response['result']
                    print(englishans)
                    translator2 = Translator(to_lang=detected_language, from_lang='en')
                    translate_text2 = translator2.translate(englishans)
                    #print(translate_text2)
                    response=translate_text2
                else:
                    response = "No answer found"

            else:

                print("Asking from LLM")
                qa_result = qa_bot()
                response = qa_result({'query': query})

                if 'result' in response:
                    response = response['result']
                else:
                    response = "No answer found"

        return jsonify({'answer': response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    
    app.run(host='127.0.0.1', port=5001, debug=True)


