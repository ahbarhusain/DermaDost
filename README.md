# DermaDost - Skin Condition Detector Web App
Derma Dost, a prototype web application, combines advanced computer vision algorithms and a Language Model (LLM) powered multilingual skin assistant to help users identify various skin conditions. With a precise pimple count estimator and multi-class classification of over 6 diseases through transfer learning, it ensures accurate assessment and identification of diverse skin issues. In a groundbreaking addition incorporates GPS technology to connect users with dermatologists and doctors in their vicinity, enabling real-time consultations. Complemented by a skin care routine generator and product analyzer using OpenAI technology, It offers a holistic approach to skincare, positioning itself as an intelligent and comprehensive solution for optimal skin health.

## Hosted here: [DermaDost-Skin-Disease](https://skin-disease.yasharya5.repl.co)
<img src="https://github.com/ahbarhusain/Dermadost/blob/master/static/images/demo.png" width="900"/>

# How It Works
DermaDost combines multiple features to assist users in understanding and managing their skin conditions:
- **Upload Images:** Users can upload images of their skin concerns. The application then uses machine learning models to identify potential skin conditions in the images.

- **Information Articles:** Once a condition is identified, DermaDost provides relevant articles from the internet. These articles offer detailed information about the skin condition, its causes, symptoms, and possible treatments.

- **Chatbot Support:** One of the standout features is the ability for users to engage with a chatbot to discuss their skin concerns further. The chatbot can provide general advice and answer common questions about skin health.

- **Authentication:** To ensure the security and privacy of your invoices, the app includes user authentication. This means you can control who has access to your invoices and other app features.

- **Webcam Capture:** DermaDost also allows users to capture images directly through their webcam. These images can be analyzed for potential skin conditions just like uploaded images.

- **JWT Authentication**: JSON Web Tokens (JWT) are used for secure user authentication. JWTs enable safe transmission of data between the app and the Firebase backend, ensuring that your data remains confidential.

## Prerequisites
Prerequisites
Before you begin, ensure you have met the following requirements:

- **Python:** DermaDost is built using Python, so make sure you have Python installed on your system. You can download it from python.org or use a package manager like Anaconda if you prefer.

- **Webcam Access (Optional):** If you want to use the webcam capture feature, ensure that your computer has a webcam, and you have granted permission for the browser to access it.

- **Git (Optional):** To clone the repository and contribute, you'll need Git installed on your system. Download it from git-scm.com.

- **Internet Connection:** DermaDost fetches information articles from the internet, so an active internet connection is required.

## Installation
To run DermaDost on your system, follow these steps:

1. Download the app's source code by cloning this repository to your local machine using the following command:

```bash
git clone https://github.com/yourusername/DermaDost.git
cd DermaDost
```

2. Install the required packages from requirements.txt using pip:
 ```bash
 py -m pip install -r requirements.txt
 ```
    
3. Run first server of application:
 ```bash
 python app.py
```

4. Run second server of application:
  ```bash
 cd Chatbot
 python model.py
```

## Usage
- Open your web browser 
- Upload an image of your skin concern or capture one through your webcam.
- Wait for the application to identify the skin condition.
- Explore detailed information about the condition and possible treatments through provided articles.
- Use the chatbot feature to discuss your skin concerns further.


### Enjoy using DermaDost! If you have any questions or encounter any issues, please feel free to create an issue on our GitHub repository.
