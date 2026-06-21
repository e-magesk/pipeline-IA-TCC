from sklearn.svm import SVC
from sklearn.metrics import classification_report, balanced_accuracy_score
import joblib
import torch
import numpy as np
import os
from torch import nn

def train_svm(model, train_loader, class_weights, save_path):
    print("\n" + "="*50)
    print("Extraindo características do treino para a SVM...")
    model.eval()
    model.cuda() 

    def extract_features_and_labels(dataloader):
        all_features = []
        all_labels = []
        with torch.no_grad():
            for data in dataloader:
                try:
                    imgs_clin, imgs_derm, labels, metadata, _, _ = data
                except ValueError:
                    imgs_clin, imgs_derm, labels = data
                    metadata = []
                except:
                    break
                
                imgs_clin = imgs_clin.cuda()
                imgs_derm = imgs_derm.cuda()
                metadata = metadata.float().cuda() if len(metadata) else None

                features = model(imgs_clin, imgs_derm, metadata, return_features=True)
                
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
                
        return np.vstack(all_features), np.concatenate(all_labels)

    X_train, y_train = extract_features_and_labels(train_loader)

    print("- Treinando a SVM...")
    svm_classifier = SVC(kernel='rbf', class_weight=class_weights, probability=True, random_state=42)
    svm_classifier.fit(X_train, y_train)
    
    svm_save_file = os.path.join(save_path, 'svm_rbf_model.pkl')
    joblib.dump(svm_classifier, svm_save_file)
    print("="*50 + "\n")
    
    return svm_classifier

class CNNSVM_Wrapper(nn.Module):
    """
    Essa classe disfarça o pipeline CNN+SVM como um modelo normal do PyTorch
    para que a função test_model consiga usá-lo nativamente.
    """
    def __init__(self, cnn_model, svm_model):
        super().__init__()
        self.cnn_model = cnn_model
        self.svm_model = svm_model

    def forward(self, img_clinical, img_dermatoscope, meta_data=None):
        # Desliga o cálculo de gradiente por segurança
        with torch.no_grad():
            # 1. Pega as features extraídas pela CNN
            features = self.cnn_model(img_clinical, img_dermatoscope, meta_data, return_features=True)
            
            # 2. Converte o tensor para NumPy (formato que a SVM entende)
            features_np = features.cpu().numpy()
            
            # 3. Pede para a SVM classificar e retornar as probabilidades (0 a 1)
            preds_np = self.svm_model.predict_proba(features_np)
            
            # 4. Converte de volta para Tensor e devolve para o test_model
            return torch.tensor(preds_np, device=features.device, dtype=torch.float32)