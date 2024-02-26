
import torch
import numpy as np
import pandas as pd
from newsfetch.news import newspaper
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from newspaper import Article
from sklearn.preprocessing import LabelEncoder
import joblib


# Example usage:

import streamlit as st

def main():
    st.title("URL and Text Input App")

    # Get URL input from the user
    url_input = st.text_input("Enter URL:", "")
    def scrape_news_content(url):
      try:
          news_article = newspaper(url)
          print("scraped: ",news_article)
          return news_article.article
      except Exception as e:
          return "Error: " + str(e)

    def summarize_with_t5(article_content, classification, model, tokenizer, device):
        article_content = str(article_content)
        prompt = "Classification: " + str(classification) + "\n"
        if not article_content or article_content == "nan":
            return "", ""
        if classification == "risks":
            prompt = "summarize the key supply chain risks: "
        elif classification == "opportunities":
            prompt = "summarize the key supply chain opportunities: "
        elif classification == "neither":
            print("Nooo")
            return "None", "None"

        input_text = prompt + article_content
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        model = model.to(device)  #/ Move the model to the correct device
        summary_ids = model.generate(input_ids.to(device), max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(summary)
        if classification in ["risks", "opportunities"]:
            st.write("This article is related to the supply chain.")
            if classification == "risks":
                return summary, "None"
            elif classification == "opportunities":
                return "None", summary
            else:
              return None,None
        else:
            st.write("This article is not classified as related to the supply chain.")
      
    def classify_and_summarize(input_text, cls_model, tokenizer_cls, label_encoder, model_summ, tokenizer_summ, device):
        if input_text.startswith("http"):
            # If the input starts with "http", assume it's a URL and extract content
            article_content = scrape_news_content(input_text)
        else:
            # If the input is not a URL, assume it's the content
            article_content = input_text

        # Perform sentiment classification
        inputs_cls = tokenizer_cls(article_content, return_tensors="pt", max_length=512, truncation=True)
        inputs_cls = {key: value.to(device) for key, value in inputs_cls.items()}

        # Move cls_model to the specified device
        cls_model = cls_model.to(device)

        outputs_cls = cls_model(**inputs_cls)
        logits_cls = outputs_cls.logits
        predicted_class = torch.argmax(logits_cls, dim=1).item()
        print("predicted_class: ", predicted_class)
        classification = label_encoder.inverse_transform([predicted_class])[0]
        print("classification: ", classification)

        # Perform summarization based on the classification
        print("article_content:", article_content)
        summary_risk, summary_opportunity = summarize_with_t5(article_content, classification, model_summ, tokenizer_summ, device)
        if summary_risk is None:
            print("No risk summary generated.")
            summary_risk = "No risk summary available"  # Provide a default value or handle accordingly
        if summary_opportunity is None:
            print("No opportunity summary generated.")
            summary_opportunity = "No opportunity summary available"  # Provide a default value or handle accordingly

        return classification, summary_risk, summary_opportunity


    print(url_input)
    cls_model =AutoModelForSequenceClassification.from_pretrained("/content/drive/MyDrive/riskclassification_finetuned_xlnet_model_ld")
    tokenizer_cls = AutoTokenizer.from_pretrained("xlnet-base-cased")
    label_encoder = LabelEncoder()

    # Assuming 'label_column values' is the column you want to encode
    label_column_values = ["risks","opportunities","neither"]

    # Extract the target column
    #y = data[label_column].values.reshape(-1, 1)

    label_encoder.fit_transform(label_column_values)

    print("Label encoder values")

    # Replace the original column with the encoded values
    label_encoder_path = "/content/drive/MyDrive/riskclassification_finetuned_xlnet_model_ld/encoder_labels.pkl"
    joblib.dump(label_encoder, label_encoder_path)

    model_summ = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer_summ = T5Tokenizer.from_pretrained("t5-small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    classification, summary_risk, summary_opportunity = classify_and_summarize(url_input, cls_model, tokenizer_cls, label_encoder, model_summ, tokenizer_summ, device)

    print("Classification:", classification)
    print("Risk Summary:", summary_risk)
    print("Opportunity Summary:", summary_opportunity)


    # Display the entered URL
    st.write("Entered URL:", url_input)
    st.write("Classification:",classification)
    st.write("Risk Summary:",summary_risk)
    st.write("Opportunity Summary:",summary_opportunity)

if __name__ == "__main__":
    main()
