import unittest
from veracity_classifier import BertVeracityClassifier

class TestBertVeracityClassifierCased(unittest.TestCase):
    def setUp(self):
        model_path = 'models/spanish_bert_cased.pth'
        model_name ='dccuchile/bert-base-spanish-wwm-cased'
        self.model = BertVeracityClassifier(model_name,model_path)
        
    
    def test_predict_false(self):
        predicted_class1,predictions = self.model.predict("La tierra es plana")
        predicted_class2,predictions = self.model.predict("Estan usando fetos abortados en las vacunas contra el coronavirus")
        predicted_class3,predictions = self.model.predict("Se lo ve a Zelensky consumiendo cocaina")
        self.assertEqual(predicted_class1, False)
        self.assertEqual(predicted_class2, False)
        self.assertEqual(predicted_class3, False)

    def test_predict_true(self):
        predicted_class1,predictions = self.model.predict("La foto del Papa Francisco con una campera blanca no es real, fue creada con inteligencia artificial")
        predicted_class2,predictions = self.model.predict("El uso del barbijo reduce el riesgo de contagio del COVID.")
        predicted_class3,predictions = self.model.predict("Argentina tiene la segunda reserva global de Shale gas y la cuarta de Shale oil")
        self.assertEqual(predicted_class1, True)
        self.assertEqual(predicted_class2, True)
        self.assertEqual(predicted_class3, True)

class TestBertVeracityClassifierUnased(unittest.TestCase):
    def setUp(self):
        model_path = 'models/spanish_bert_uncased.pth'
        model_name ='dccuchile/bert-base-spanish-wwm-uncased'
        self.model = BertVeracityClassifier(model_name,model_path)
        
    def test_predict_false(self):
        predicted_class1,predictions = self.model.predict("La tierra es plana")
        predicted_class2,predictions = self.model.predict("Estan usando fetos abortados en las vacunas contra el coronavirus")
        predicted_class3,predictions = self.model.predict("Se lo ve a Zelensky consumiendo cocaina")
        self.assertEqual(predicted_class1, False)
        self.assertEqual(predicted_class2, False)
        self.assertEqual(predicted_class3, False)

    def test_predict_true(self):
        predicted_class1,predictions = self.model.predict("La foto del Papa Francisco con una campera blanca no es real, fue creada con inteligencia artificial")
        predicted_class2,predictions = self.model.predict("El uso del barbijo reduce el riesgo de contagio del COVID.")
        predicted_class3,predictions = self.model.predict("Argentina tiene la segunda reserva global de Shale gas y la cuarta de Shale oil")
        self.assertEqual(predicted_class1, True)
        self.assertEqual(predicted_class2, True)
        self.assertEqual(predicted_class3, True)
      


if __name__ == '__main__':
    unittest.main()