# |ENG| Brain Tumor Classification using EfficientNet and XAI

This repository contains the code for an image classification model based on Transfer Learning (using the EfficientNet architecture) designed to classify brain Magnetic Resonance Imaging (MRI) scans. The project has a central focus on Explainable AI (XAI), implementing advanced qualitative and quantitative techniques to visually explain the model's decisions and validate its clinical reliability.

## Dataset

The model was trained and evaluated using the "[Crystal Clean: Brain Tumors MRI Dataset](https://www.kaggle.com/datasets/mohammadhossein77/brain-tumors-dataset)" available on Kaggle. This dataset contains brain MRI images categorized into 4 classes:

* No tumor
* Glioma 
* Meningioma
* Pituitary

The original dataset was split into training, validation, and test sets using the scikit-learn library, ensuring class stratification.

## Architecture and Pre-processing

The model's pipeline stands out for using robust segmentation prior to classification:

1. Pre-processing (Segmentation): Implementation of Otsu's Thresholding algorithm (via OpenCV) to remove the background and skull, isolating the brain's region of interest (ROI). This ensures the model learns only from brain tissues, avoiding biases caused by background noise.
2. Backbone (Transfer Learning): Use of the EfficientNetB0 network pre-trained on ImageNet as a feature extractor.
3. Top Layers:
    * Global Average Pooling for dimensionality reduction.
    * Batch Normalization to stabilize training.
    * Dropout for regularization and overfitting prevention.
    * Dense output layer with Softmax activation (4 classes).

The model compilation uses the `categorical_crossentropy` loss function and the Adam optimizer.

Network configuration:

![Network configuration](images/configuracao.png)

## Explainable AI (XAI) & Quantitative Evaluation

To ensure the transparency of the "black-box" model, three explainability techniques were implemented, both qualitatively and quantitatively:

### Qualitative Explainers

1. **Grad-CAM++ (Gradient-weighted Class Activation Mapping ++):**
    * Visualization of heatmaps based on the 1st, 2nd, and 3rd order gradients of the last convolutional layer (`top_activation`).
    * Allows for precisely identifying the spatial region that most activated the neurons for the predicted class.
  
2. **LIME (Local Interpretable Model-agnostic Explanations):**
    * Generation of local explanations through image perturbations and superpixel segmentation.
    * Configured with multiple visualization options: side-by-side comparison, confirmation/contradiction areas, and black-background isolation.
    * Robust integration with the segmentation step to prevent the algorithm from considering the black background as a relevant feature.

3. **SHAP (SHapley Additive exPlanations):**
    * Uses game theory (Partition/Black-box Explainer) to attribute the contribution of each pixel to the final probability.
    * **Bias Audit:** Fundamental to identify "Shortcut Learning", revealing initial reliance on skull bone features.
    * **Segmentation Validation:** Confirmed that *Skull Stripping* redirected model focus strictly to the tumor mass and adjacent tissues.

### Quantitative XAI Metrics

To mathematically evaluate and compare the quality of the generated explanations, the following metrics were computed for all XAI methods across sample batches:

* **Gini Index (Sparsity / Concentration):** Measures how concentrated the explanation heatmaps are. Higher values indicate that the model relies on a compact, highly focused set of pixels rather than diffuse noise.
* **AOPC (Area Over the Perturbation Curve / Faithfulness):** Quantifies how faithful the explanation is to the model's decision process by systematically masking top-ranked pixels (in $k$ steps) and measuring the drop in prediction probability.


## Training
1. The dataset went through the segmentation pipeline before entering the network.
2. The model used pre-trained weights (ImageNet) as a starting point (Fine-tuning).
3. The training used callbacks such as Early Stopping (monitoring validation loss) and ReduceLROnPlateau for dynamic learning rate adjustment.
4. The best model (based on val_loss) was automatically saved through ModelCheckpoint.

## Results

The chart below shows the evolution of accuracy and loss during the training and validation epochs:

![Chart](images/grafico.png)

Below is the trained model's confusion matrix, demonstrating the performance per class:

![Confusion matrix](images/matriz.png)

After training, the model showed the following results on the test set:

* **Test Loss:** 0.1755
* **Test Accuracy:** 0.9654 (96.54%)
* **Test Precision:** 0.9671 (96.71%)
* **Test Recall:** 0.9626 (96.26%)
* **Test F1-score:** 0.9653 (96.53%)

## How to Use

1.  Clone this repository.
2.  Install the necessary dependencies:
`pip install tensorflow opencv-python matplotlib lime scikit-image`
4.  Download or import the "[Crystal Clean: Brain Tumors MRI Dataset](https://www.kaggle.com/datasets/mohammadhossein77/brain-tumors-dataset)" dataset from Kaggle.
5.  Organize the dataset in the standard format (folders by class).
6.  Run the training notebook (`TLSeg_128_6.ipynb`) to generate the `.keras` file.
7.  Generating Explanations:
   * Unified Quantitative & Qualitative Pipeline: Run XAI_QUANTITATIVA.ipynb (or .py) to calculate Gini Index and AOPC scores alongside generating visualizations for Grad-CAM++, LIME, and SHAP in batch mode.
   * Individual Scripts: Alternatively, individual qualitative routines can still be accessed via gc_shap_lime.py (or individual gradcam, lime, shap scripts).


--------------------------------------
# |PT/BR| Classificação de tumores cerebrais usando EfficientNet e XAI

Este repositório contém o código para um modelo de classificação de imagens baseado em Transfer Learning (utilizando a arquitetura EfficientNet) projetado para classificar imagens de ressonância magnética (RM) de cérebros. O projeto tem um foco central em Explainable AI (XAI), implementando técnicas avançadas qualitativas e quantitativas para explicar visualmente as decisões do modelo e validar sua confiabilidade clínica.

## Dataset

O modelo foi treinado e avaliado utilizando o dataset "[Crystal Clean: Brain Tumors MRI Dataset](https://www.kaggle.com/datasets/mohammadhossein77/brain-tumors-dataset)" disponível no Kaggle. Este dataset contém imagens de RM cerebrais categorizadas em 4 classes:

*   Sem tumor
*   Glioma 
*   Meningioma
*   Pituitary

O dataset original foi dividido em conjuntos de treino, validação e teste utilizando a biblioteca scikit-learn, garantindo a estratificação das classes.

## Arquitetura e pré-processamento

O pipeline do modelo diferencia-se pelo uso de segmentação robusta antes da classificação:

1. Pré-processamento (Segmentação): Implementação do algoritmo Otsu's Thresholding (via OpenCV) para remover o fundo e o crânio, isolando a região de interesse (ROI) do cérebro. Isso garante que o modelo aprenda apenas com tecidos cerebrais, evitando vieses causados por ruídos de fundo.
2. Backbone (Transfer Learning): Utilização da rede EfficientNetB0 pré-treinada no ImageNet como extrator de características.
3. Camadas Superiores (Top Layers):
    *   Global Average Pooling para redução dimensional.
    *   Batch Normalization para estabilizar o treinamento.
    *   Dropout para regularização e prevenção de overfitting.
    *   Camada Dense de saída com ativação Softmax (4 classes).

A compilação do modelo utiliza a função de perda `categorical_crossentropy` e o otimizador Adam.

Configuração da rede:

![Configuração da rede](images/configuracao.png)

## Explainable AI (XAI) & Avaliação Quantitativa

Para garantir a transparência do modelo "caixa-preta", foram implementadas três técnicas de explicabilidade, avaliadas tanto qualitativa quanto quantitativamente:

1.  Grad-CAM++ (Gradient-weighted Class Activation Mapping ++):
    *   Visualização de mapas de calor baseados nos gradientes de 1ª, 2ª e 3ª ordem da última camada convolucional.
    *   Permite identificar com precisão a região espacial que mais ativou os neurônios para a classe predita.
  
2.  LIME (Local Interpretable Model-agnostic Explanations):
    *   Geração de explicações locais através de perturbações na imagem e segmentação em superpixels.
    *   Configurado para exibir áreas de confirmação (que apoiam a decisão) e áreas de contradição.
    *   Integração robusta com a etapa de segmentação para evitar que o algoritmo considere o fundo preto como característica relevante.

3. SHAP (SHapley Additive exPlanations):
   * Utiliza a teoria dos jogos para atribuir a contribuição de cada pixel (ou bloco de pixels) para a probabilidade final de cada classe.
   * Auditoria de Viés: Foi fundamental para identificar o "Shortcut Learning", revelando que o modelo inicialmente focava no osso do crânio. 
   * Validação de Segmentação: Após a implementação do *Skull Stripping* via OpenCV (Contornos + Erosão), o SHAP confirmou que o modelo passou a focar exclusivamente na massa tumoral e tecidos adjacentes.
  
### Métricas Quantitativas de XAI

Para avaliar e comparar matematicamente a qualidade das explicações geradas, as seguintes métricas foram calculadas para todos os métodos de XAI em lotes de teste:

* **Índice Gini (Sparsity / Concentração):** Mede o quão concentrado está o mapa de calor da explicação. Valores mais altos indicam que o modelo toma decisões com base em um conjunto focado e compacto de pixels, em vez de ruídos dispersos.
* **AOPC (Area Over the Perturbation Curve / Fidelidade):** Quantifica a fidelidade da explicação ao processo de decisão do modelo, mascarando progressivamente os pixels mais importantes (em $k$ passos) e medindo a queda resultante na probabilidade da classe correta.

## Treinamento
1. O conjunto de dados passou pelo pipeline de segmentação antes de entrar na rede.
2. O modelo utilizou pesos pré-treinados (ImageNet) como ponto de partida (Fine-tuning).
3. O treinamento utilizou callbacks como Early Stopping (monitorando a perda de validação) e ReduceLROnPlateau para ajuste dinâmico da taxa de aprendizado.
4. O melhor modelo (baseado na val_loss) foi salvo automaticamente através do ModelCheckpoint.

## Resultados

O gráfico abaixo mostra a evolução da acurácia e da perda durante as épocas de treinamento e validação:

![Gráfico](images/grafico.png)

Abaixo, a matriz de confusão do modelo treinado, demonstrando o desempenho por classe:

![Matriz de confusão](images/matriz.png)

Após o treinamento, o modelo apresentou os seguintes resultados no conjunto de teste:

*   **Loss no teste:** 0.1755
*   **Acurácia no teste:** 0.9654 (96.54%)
*   **Precisão no teste:** 0.9671 (96.71%)
*   **Recall no teste:** 0.9626 (96.26%)
*   **F1-score no teste:** 0.9653 (96.53%)

## Como Usar

1.  Clone este repositório.
2.  Instale as dependências necessárias:
`pip install tensorflow opencv-python matplotlib lime scikit-image`
4.  Faça o download ou importe o dataset "[Crystal Clean: Brain Tumors MRI Dataset](https://www.kaggle.com/datasets/mohammadhossein77/brain-tumors-dataset)" do Kaggle.
5.  Organize o dataset no formato padrão (pastas por classe).
6.  Execute o notebook (`TLSeg_128_6.ipynb`) de treinamento para gerar o arquivo `.keras`.
7.  O melhor modelo treinado será salvo no caminho configurado para posteriores inferências ou análises.
8.  The best trained model will be saved in the configured path for further inferences or analysis.
9.  Geração de Explicações:
   * Pipeline Quantitativo e Qualitativo Unificado: Execute o notebook/script XAI_QUANTITATIVA (.ipynb ou .py) para calcular automaticamente o Índice Gini e o AOPC enquanto gera as imagens explicativas em lote para Grad-CAM++, LIME e SHAP.
   * Scripts Individuais: Alternativamente, os scripts qualitativos individuais ou consolidados anteriores ainda podem ser acessados via gc_shap_lime.py (ou scripts isolados de cada método).
