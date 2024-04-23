import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer

class BertVeracityClassifier:
   
    def __init__(self, model_name,model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #Proceso parametros
        path = self._get_path(model_path)
        state_dict = torch.load(path,map_location= self.device)
        state_clean = self._dict_cleaner(state_dict)

        #Inicializo el Modelo
        model = BertForSequenceClassification.from_pretrained(model_name)
        model.load_state_dict(state_clean)
        model.eval()

        #Declaro los atributos
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = model
        self.state = state_clean
    
    def predict(self,text,threshold = 0.5):   
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
        
        predicted_class = torch.argmax(logits, dim=1).item()
        if probabilities[predicted_class] <= threshold and predicted_class == 1:
            predicted_class = 0
  
        return bool(predicted_class), probabilities
    
    
    def _dict_cleaner(self,model_state_dict):
        model_state_clean = {}
        for key, value in model_state_dict.items():
            if key == "fc.weight" or key == "fc.bias":
                key = key.replace("fc", "classifier")
            model_state_clean[key] = value
        return model_state_clean
    
    def _get_path(self,path_name):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(dir_path, path_name)