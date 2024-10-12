# SREDH/AI CUP 2023


<div align="center">

<!-- Add your project logo if you have any -->
<!-- <img width=200px height=200px src="images/logo.png" alt="Project logo">-->

</div>

<h1 align="center">Patient DeIdentification</h1>

 <p align="center">
 	<!-- Add your tagline or very short intro of your project -->
    Leveraging Large Language Models for De-identification and Temporal Normalization of Sensitive Health Information in Electronic Health Records
<h3 align="center"> Find Detailed Implementation, Summaries and Descriptions Inside Each Project</h3>

<hr />
<br />

<div align="center">

<!-- Add your project demo gif here -->

</div>

<!-- You may write notes in your readme this way if you want to, it looks good and also different from other text -->


<p align="center">Innovative solutions developed by the top 10 teams to tackle the challenge of deidentifying sensitive health information (SHI). This repository showcases cutting-edge approaches leveraging Large Language Models (LLMs), machine learning, and rule-based techniques to ensure robust privacy protection while maintaining the integrity of medical data.</p>

## 🧐Background

In recent years, artificial intelligence (AI) technology has developed rapidly. Especially in the past year, companies such as OpenAI, Microsoft and Google have introduced and used their own large-scale language models in related products. These applications such as ChatGPT shown the application potential of Large Language Models (LLMs) in various fields. The application of LLM in clinical medicine is therefore regarded as the future of AI in the field of digital health, which is now a very important and evolving research area. However, when applying such AI models, ordinary users and even system or program developers often do not realize the privacy information issues when interacting with LLMs, which may lead to the risk of leaking important confidential information. In addition, if the training data used in training such large language models contains real private information (such as an individual's name, phone number, ID card number, etc.), there is a certain possibility that it will be affected by the memory capacity of the LLM. The ability to interact with users leads to the leakage of private information.

## 💡Top-10 Teams Summarries.

The competition attracted a total of 721 participants, constituting 291 teams. Teams played a crucial role in the competition. During the development phase, only the team leader could submit up to three runs per day to evaluate their model performance using the leaderboard. In the final testing phase, each team can submit a maximum of six runs over the course of two days. Out of the participating teams, 103 teams submitted their prediction results for the test set, resulting in 218 submissions during the final testing phase. Since there were 103 teams, the committee invited only the top 10 teams on the leaderboard. Below are the summaries of 10 team methodologies.

## R-01 [R-01](../R-01@b5b17b7/)

- The R-01 team developed a textual data model for clinical entity recognition using a stringent preprocessing protocol. They refined the T5-Efficient-BASE-DL2 model for computa-tional efficiency to generate text. They further structured the dataset into two training sets, one validation set, and a unique test set. For the AutoTokenizer, they used AutoModel-ForSeq2SeqLM classes from the Hugging Face library for consistency. The model's performance was assessed without gold standard annotations, highlighting the value of rule-based methods in clinical entity recognition. However, recall metrics show areas for improvement.

## R-02
The R-02 team conducted a comparative study to assess the efficacy of two approaches: fine-tuning a large language model based on the Chat generative pre-trained transformer (ChatGPT) and a rule-based approach. The ChatGPT-based approach achieved high macro-F-scores for SHI recognition and temporal information normalization, while the rule-based ap-proach demonstrated lower latency and power consumption. Future enhancements may involve augmenting training data and patterns for rare SHI types and addressing challenges in LLM out-put interpretation.
## R-03
The R-03 team model architecture is based on the Transformer framework, incorporating GPTNeoXAttention, GPTNeoXRotaryEmbedding, parallel computation of Attention and Feed-Forward, and all Dense Layers. They used the "AdamW" optimization method to enhance the Adam algorithm. Their approach employed a scheduler to adjust the learning rate and selected the "linear warmup" algorithm. Furthermore, they employed rule-based postprocessing to ad-dress low recall rates, particularly for unpopular labels. The team applied rule-based processing to seven labels in the competition and expanded it to include other unpopular and popular la-bels, resulting in improved scores. The study concludes that combining traditional LLM training methods and rule-based postprocessing shows significant promise for future SHI de-identification tasks.
## R-04
The R-04 team developed an algorithm using Inverse Correlational Learning (ICL) and the Sliding Windows method to improve the correlation between sequences and their labels. The sliding windows method understands the start and end of each sequence but could lead to learning irrelevant la-bels. To eliminate these, they introduced a parameter. Furthermore, they used the Fully Sharded Data Parallel (FSDP) technique for stability. Their method proposed two measures to counteract overfitting: the NEFTune method and random extraction of windows. The approach demonstrat-ed outstanding performance in de-identification tasks, with a success rate of over 90%. The COT-few method was integrated into the Prompting approach to address the small number of labels in the DURATION and SET categories.
## R-05
The R-05 team utilized a deep learning model, Pythia-160 m, to train and analyze its performance in a competi-tion. Their training method involved 12 iterations, with AutoModelForCausalLM for consistent text length and a 5e-5 learning rate. They further created a BatchSampler for efficient data split-ting; in their approach, Checkpoint saved weights, and ChatGPT generated synthetic data. Fur-thermore, the data processing included preprocessing and postprocessing. Their results demon-strated that Pythia-160 m performed well in the non-competition dataset, but some scores were lower in the competition set. Despite exploring different models and methods, we chose Pythia-160 m as the prominent architecture.
## R-06
The R-06 team approach used advanced pre-trained models and evaluated both discriminative and generative approaches for SHI extraction and time information normalization in medical records. The study introduces a novel Dual-Model Fusion with a Voting Mechanism, preserving the strengths of auto-regressive and auto-encoder models for optimal performance. The study also uses various data processing methods, including concatenation, cropping, and data augmentation, especially for smaller datasets. The models' learning efficacy is enhanced using advanced training tech-niques like parameter freezing and Low-Rank Adaptation (LoRA).
## R-07
The R-07 team used three distinct corpora to identify sensitive personal health information. They used the OpenDeID corpus, the 2014 i2b2/UTHealth de-identification corpus, and the 2016 CEGS N-GRID de-identification corpus. The deidentification process was segmented and tokenized using the Spacy library, and the BIESO tagging scheme was adopted. The Discharge Summary Bi-oBERT hybrid de-identification model demonstrated the ability to maintain a delicate balance between preserving patient privacy and ensuring accurate deidentification. Future research could explore the model's scalability, robustness across diverse settings, and potential integration into health informatics frameworks.
## R-08
The R-08 team used a rule-based NER system to categorize annotation files using HashMap and HashSet. The method consists of an Information Extraction Algorithm, Time Normalization, and System Evaluation and Optimization processes. Precision, recall, and F1-score metrics evaluate the sys-tem's effectiveness. The Discharge Summary BioBERT hybrid de-identification model demon-strated the ability to maintain a delicate balance between preserving patient privacy and ensuring accurate deidentification. They further evaluated model performance under various configura-tions, with Run4 exhibiting the most favorable overall performance.
## R-09
The R-09 team developed a hybrid approach to recognize and standardize Sensitive Health Identifiers (SHIs). The approach used the Pythia language model and a regular expression program to enhance recognition accuracy. The team created a SHI retrieving program that uses regular expression patterns to identify SHIs in medical records. The hybrid approach provides a dataset capture and standardization solution within complex medical records.
## R-10
The R-10 team implemented a longformer model for tasks such as PPIR, NER, and TIN, addressing class distri-bution imbalances and handling overlapping text segments. They further developed a rule-based approach for TIN tasks, addressing limited data and unstructured time-related categories. The study used an ensemble learning technique to integrate five models trained on different datasets, enhancing the robustness and accuracy of entity recognition. Both models showed high preci-sion, recall, and F1 scores across most categories. The Longformer model showed variable per-formance across different categories, suggesting future iterations should focus on category-specific loss weighting and rule-based secondary processing steps.


