  
# Cultural Biases in Generative AI: Exploring Religious Bias in Stable Diffusion
#### B. Bromová, G. Maggi, C. Mautner Markhof, R. Rapparini, L. Zurdo


## Table of Contents
[Introduction](#introduction)

[1. The Stable Diffusion Model](#model)

[2. Literature Review](#litreview)

[3. Methodology](#metho)

[4. Analysis of Outputs](#output)

[5. Discussion](#discussion)

[Conclusion](#conclusion)

[Bibliography](#bibliography)



<a name="introduction"></a>
## Introduction

<p align="justify"> 
From artificial intelligence (AI) art to deep fakes, image-generative algorithms are emerging in discussions that span culture, philosophy and politics. Integral to these conversations are questions about bias and fairness. A frequent example – also taken up by this paper – is Stable Diffusion, launched in 2022 by Stability AI. Stable Diffusion follows the footsteps of Open AI’s DALL•E in providing an machine-learning (ML) system to generate images, yet it differs significantly in terms of approach. From its methodology to its underlying values, StableDiffusion could shape a different way of engaging with algorithmic bias. 
<p align="justify">  
This research work is a case study on Stable Diffusion’s biases when it comes to religion and culture, exploring whether the Stable Diffusion image generation system handles religion in an unbiased way. The paper focuses on religious bias as it has been less explored. Following an explanation of the Stable Diffusion model (Section 1), a literature review (Section 2), and a description of our methodology (Section 3), Section 4 explores the research question from an empirical perspective, analysing image outputs across a standardised set of prompts to detect religious biases. Thereafter, Section 5 discusses the findings, exploring how religious biases interact with Stability AI’s core values. All in all, we find evidence of religious bias in the shape of stereotyping, complex biases, and limited visual reasoning. We further posit that Stability AI frames the discussion in a way that allows it to both acknowledge open-source’s role in amplifying biases and simultaneously claim it as necessary for the solution. 

  
<a name="model"></a>
## 1. The Stable Diffusion Model
  
#### 1.1 Evolution of text-to-image generative models
<p align="justify"> 
From the early 2010s, progress in the field of Deep Neural Networks has dramatically enhanced the capabilities of AI systems, achieving promising results in areas like image recognition (Krizhevsky et al., 2012), and machine translation (Bahdanau et al., 2015). Against this backdrop, Mansimov et al. (2015) introduced a model capable of generating images from text descriptions. Since then, further improvements was made by applying Generative Adversarial Networks (GANs) – a generative model that builds on noise-contrastive estimation (see Guttman & Hyvarinen 2010) to enable the system to distinguish data from noise – especially through the works of Reed et al. (2016). 
<p align="justify"> 
State of the art systems such as DALL•E and DALL•E 2 use diffusion models (DMs) to generate images – a process which iteratively guesses the amount of Gaussian noise blurring an image, guesses the image by subtracting the noise, and then adds back some noise. Repeating this process improves the resolution of the image until the final version is achieved (Sohl-Dickstein et al. 2015). However, Stable Diffusion uses an improved version of these models – called latent diffusion models (LDMs) – which are less computationally expensive but retain the quality and flexibility of DMs thanks to the use of a latent space instead of a pixel space (Rombach et al. 2022, Siddiqui 2022). Latent diffusion models work by destroying training data through the asymptotical introduction of Gaussian noise and subsequently learning how to reconstruct the data by denoising the process. After the learning phase is complete, it is possible to generate data by passing randomly sampled noise through the denoising process. In principle, this corresponds to learning the inverse process of a Markov chain of length T.
<p align="justify"> 
Notably, the noise generation of these models does not happen on a random basis but it is informed by the text string that the users input – that is to say that the network is conditioned on the text (y) by means of a conditional denoising autoencoder eθ(zt, t, y) (Rombach et al., 2022). The text string is then analysed through a Natural Language Processing (NLP) transformer embedding, which captures the semantic structure of the text. More specifically, StableDiffusion uses a frozen CLIP ViT-L/14 (Contrastive Language-Image Pre-training) text encoder that conditions the model on text prompts. This type of encoder is designed to maximise the similarity between text and images pairs by contrastive loss (Open AI, n.d.). Contrastive loss is a method used to compare output features in Siamese networks, which are a form of encoding network whereby 2 or more inputs are encoded and output features are analysed. The network output is taken as a positive example, and its distance to an example of the same class is then calculated in order to compare and contrast it with the distance to negative examples. In other words, contrastive loss is used to map vectors that reflect the similarity between items and form the basis of machine learning models such as Stable diffusion. Finally, to make the image closely tied to the text string, Stable Diffusion uses a classifier-free guidance (Cfg) (Ho & Salimans 2021) at the end of the network. This generates additional non-conditioned guesses of the noise and subtracts it from the conditioned guess to get a tighter prediction of initial noise and thus an output closer to the users’ prompt.

#### 1.2 Stable Diffusion Training Dataset
<p align="justify">  
Stable Diffusion is trained on 512x512 images from a subset of the LAION-5B database. LAION-5B is an open large-scale multi-modal dataset composed of 5.85B Clip-filtered image-text pairs stored in 3 data packages. 2.3 billion image-text pairs contain English language and are stored in the LAION2B-EN package, whereas 2.2 billion samples are collected from over 100 languages and are contained in the LAION2b-multi package. Furthermore, 1 billion samples have texts that are not uniquely associable with a language (e.g. names) and are stored in the LAION1b-nolang (Beaumont, 2022).
<p align="justify">  
Each dataset is composed of the following variables: URL, that reports the image url, with millions of web domains covered; TEXT, which reports captions in the language of the dataset; WIDTH, which indicates the picture width; HEIGHT, which refers to the picture’s height; LANGUAGE, which refers to the language of the sample, is reported only for laion2B-multi and is computed using cld3, a neural network model for language identification; SIMILARITY, which is the cosine between the text and the image ViT-B/32 embeddings; WATERMARK, which is the probability of being a watermarked image; UNSAFE, or the probability of being an unsafe image (Beaumont, 2022). For these last two parameters, the model flags the image as watermarked when the probability value is larger than 0.8, while it deems it unsafe when the probability is predicted at 0.5. For the laion2B-en, laion2B-multi and laion1B-nolang, the proportion of unsafe images is equal to 2.9%, 3.3% and 3% respectively, while the proportion of watermarked images is 6.1%, 5.6% and 4%. 


<a name="litreview"></a>
## 2. Literature Review
<p align="justify"> 
There is extensive research on the ethics, impacts, and mitigation of biases in ML algorithms. Mehrabi et al. (2021) distinguished three categories of bias. Firstly, bias in how the data is collected, like measurement bias, omitted variables, and aggregation bias. Secondly, bias in user-generated data, like self-selection, social, and content production bias. Thirdly, algorithmic bias that derives from algorithmic design choices such as what optimisation functions or regularisation to use. In StableDiffusion, there are two fundamental blocks from which bias may arise: NLP and Computer Vision (CV). These are subject to the above-mentioned biases but present some specificities which are proper of the two fields.
  
#### 2.1 Language Standardisation
<p align="justify"> 
Firstly, NLP plays an important role in the Stable Diffusion algorithm as it is the block of code which conditions both the noise generated and the network itself. The main problem with modelling language is the fact that it is not standardised (Holy & Prabhumoye 2021), as there  are significant differences in the way in which different people refer to and describe the same situation or object. (Labov 1972). These divergences are tightly linked to the socio-demographic group to which the person belongs to as well as their self-identification within such categories. In other words, while language carries with it secondary information about the speaker, NLP fails to pick up on these social factors (Flek 2020, Hovy & Yang 2021). In fact, by not taking into account these demographic variations, these tools expect language to be a “standardised” and, as a result, perform significantly worse in interpreting language not encoded in their training sample (e.g. there are correlations between a person’s age and decreases in model performance, see Hovy & Søgaard 2015). For instance, NLP tools are commonly trained on established news sources (e.g., the New York Times) that reflect their writers’ homogeneity (typically educated, white, upper-middle class) and may lead to differences in predictive performance linked to the author’s gender (Garimella et al. 2019). 

#### 2.2 Labelling
<p align="justify"> 
Secondly, and of high relevance to our study, bias in NLP and CV derives from the labelling the data. It may occur due to annotators’ disinterest; linguistic disagreement on what label to use for a single concept (thus embedding the choice of a label with labellers’ ideological, and often subconscious, preferences, see Planck et al. 2014); or a divergence between the author’s and annotator’s demographics and norms (e.g., Sap et al. 2019 argued that African American English is more likely to be recognised as hate speech due to the demographics of the people who labelled the data – overwhelmingly white). Using crowdsourced labour that is not demographically representative, like Amazon’s Mechanical Turk, worsens this issue (Pavlick et al. 2014).
<p align="justify"> 
Label bias is also serious in CV datasets (Torraba & Efros 2011, Boulamwini and Gebru 2018). Similarly to NLP, the significant problem in these datasets is that of poor definition of the semantic categories associated with the images and aspect which is especially clear when talking about ethnicity (Barbujani & Colonna 2011). In their analysis of the ImageNet database, Crawford and Paglen (2019) highlighted the role of taxonomy as providing the branched categories which help the algorithm set concepts and images in relation to each other and improve the system’s visual reasoning (Cho et al, 2022). Yet aside from improving visual reasoning, taxonomies can prove problematic to the extent they entrench reductive distinctions or fail to reflect the world’s true diversity. Moreover, categories themselves may have a normative dimension, and the classification of persons or objects within them can be reflective of labelers’ bias. Labelling bias is particularly problematic when it comes to highly physical characteristics such as gender and ethnicity. When studying the presence of abstract biases – such as the religious bias considered here – the role played by the disputed, blurred and subjective nature of the labelling of NLP and CV datasets will become apparent. 
  
#### 2.3 Stereotypical Association
<p align="justify"> 
Thirdly, the algorithm picks up on stereotypical associations in the training data. This gives rise to the two problems of semantic bias and bias overamplification. The former is very well documented and harder to address is the issue of semantic bias, namely the ability of an algorithm to pick up on stereotypical associations especially related to gender and racial bias in the training data. The problem here is the historical association of certain categories of people with specific stereotypes which are represented in word embeddings – namely, a vector representation of the symbolic relationships between concepts in the semantic space. These are documented in multiple studies such as Garg et al. (2018) who compared word embeddings with US census data; Bhatia (2017) who shows the close relation between AI predictions and people’s judgements based on stereotypes; and Kozłowski et al. (2019) who establishes word embeddings as a way to empirically test theories of social class. What this shows is that word embeddings and contextual representations closely reflect and perpetuate societal bias and stereotypes. 
<p align="justify"> 
Moreover, Gonen & Goldberg (2019) shed light on how much these stereotypical representations are resistant to debasing attempts. This is especially true because of the masked nature of these biases which makes them incredibly difficult to identify and address in the pre-training stage. However, this also counts true, when attempting to reduce the bias in outputs at the prompt or through “guardrails” as used by DALL•E. Bianchi et al. (2022) refer to such biases that cannot be easily corrected as ‘complex biases’. These biases occur as “pernicious assumptions and power relations” reflected in the output images. Bianchi et al. (2022) provide the example of African men being depicted in front of a house in a significantly worse condition than for American men. This problem is rendered worse through bias overamplification. This stems from the loss objectives given to the AI system aimed at achieving higher precision but which results in overfitting and exploiting spurious correlations in the training dataset (Hovy & Prabhumoye 2021).

#### 2.4 Research Design
<p align="justify"> 
Finally, bias can be embedded in the choices made already in research design itself. For instance in the case of NLP, most relevant research is conducted in English and focuses on the main Indo-European languages (Joshi et al. 2020). This creates a self-reinforcing trend: the less information is available, the fewer people will work on modelling a particular linguistic, cultural, or religious context; thus less data will be generated, less progress will be made, and fewer resources will be available for future pursuits (Hovy & Prabhumoye 2021). Kahneman called this type of bias ‘availability heuristics’: the more exposure to something one has, the more normal and plausible it seems. This dynamic is only amplified in the context of large machine learning models which usually require vast amounts of reliable data to increase output quality. It becomLeees particularly problematic when dealing with less-explored phenomena or underserved minorities: to illustrate in the context of religion, less populous systems of belief – usually smaller, more local religions such as Shintoism – are likely to be relatively understudied, attracting fewer resources, gathering less suitable data, and thus generating outputs of lower quality. Intersectionality also plays a role: it is likely that women, people of minority ethnicities or gender expressions will face an amplified version of the aforementioned cycle of neglect. 

<p align="justify"> 
Informed by these considerations, the next section dives deeper into the analysis of a less explored bias in machine learning: religious bias.

  
  <a name="method"></a>
## 3. Methodology
<p align="justify"> 
The qualitative bias analysis focused on eight religions or religion-related categories: Christianity, Islam, Hinduism, Buddhism, Sikhism, Judaism, Shintoism, and Atheism.  To generate relevant images through Stable Diffusion 2, the prompt ‘A photo of the face of ___’ was used, followed by a gender-neutral term for a follower of each selected religion. In an exploratory part of the study further non-specific religion-related terms were added after to dive deeper into certain biases: young Judaism, modern Judaism, Jewish women, Sikh women, devoted, religious, good, sinful. 50 images were generated by running each prompt through Stability AI’s Dreamstudio Demo, keeping the Cfg Scale value constant at 7, and the Steps value constant at 50 (both were the given default values). The Cfg scale describes the closeness of the output to the prompt on a scale from 0 (low closeness) to 20 (high closeness). The steps indicate how many steps were taken to diffuse or generate the image. From this bank of images, 20 were randomly selected from each category for qualitative analysis.
<p align="justify"> 
Subsequently, the output was analysed using the qualitative method of elaborative coding. Elaborative coding is used to test and expand on existing theory which predefined theoretical constructs (Auerbach & Silverstein, 2003). To do this, the paper draws on the biases of text-to-image generative models outlined above. However, it will focus particularly on general stereotypes with stereotype amplification and complex stereotypes used by Bianchi et al. (2022) as part of the stereotypical association and expands on them by including Cho et al.’s (2022) visual reasoning skills as part of the labelling bias. Their studies function as a fitting theoretical framework to expand existing research to the topic of biases in religious depictions. Language standardisation and research design are not covered in separate subsections but feature within the presented theoretical framework.
<p align="justify"> 
Since the coding process is based on the concepts of Bianchi et al (2022) and Cho et al. (2022), elaborative coding is seen as a ‘top-down’ approach to coding as opposed to open coding (Auerbach & Silverstein, 2003). With the overarching concepts in mind the output was analysed for repeating themes or features that could be summarised into specific categories under these concepts. In the following section each concept with the respective categories of codes is presented.
  

<a name="output"></a>
## 4. Analysis of Outputs
  
#### 4.1 General Stereotypes and Stereotype Amplification
<p align="justify"> 
The images Stable Diffusion generated display a wide range of stereotypes and are particularly prone to amplify stereotypes. Stereotype amplification occurs when real-life correlations between certain features such as gender and profession are made more prominent in the output than they are in society at large (Quillian & Pager, 2010). In this case, the correlation between masculinity and religion is amplified since the gender split in religions is relatively even, yet Stable Diffusion outputs an overwhelming majority of people with masculine features throughout all belief systems. In fact, the only feminine features generated were in samples representing Islam and Hinduism (Figure 1). The overwhelmingly masculine output also holds true for non-specific religious prompts by the likes of ‘sinful person’ or ‘religious person’ (Figure 4). This preference for masculinity may be a general bias present throughout the system (Bianchi et al, 2022) or could also point to the unequal standing between men and women in religious contexts, since in most religions over-represent men in their central positions of authority.
  

<div align="center"> 
  
<img width="650" alt="figure 1" src="https://user-images.githubusercontent.com/55432992/205193775-a157efba-a771-46d5-b131-4b0709341e9e.jpeg">
  
  
  *Figure 1: Examples of images generated using the prompt ‘A photo of the face of a ___’ alongside four belief systems: Christianity, Buddhism, Hinduism, and Islam.*
    </div>
    
<p align="justify"> 
Stereotype amplification can also be detected when analysing the output for religious symbols or clothing.  This applies predominantly to Sikhism, Hinduism, and Islam. The images of all three religions show mostly masculine features with strong facial hair dressed in religious clothing or covered in some religious symbols (Figure 1). It is striking that all Hindus (male and female) have Bindis and/or additional decorative jewellery on their foreheads, all Sikhs are wearing turbans, and all humanly portrayed Buddhists resemble monks, suggesting that the model draws on ‘unambiguous’ portrayals of religion even without being prompted to. Furthermore, the images’ over-estimation of religious symbolism may amplify backwards and underdeveloped impressions of these religions. 
<p align="justify"> 
However, the images do not display commonly held negative stereotypes against some religions: this is clear when looking at the images of Muslim people. While commonly associated features such as a large beard on men or head scarves on women are seen on every picture, there are no signs of malicious or negative stereotypes (eg. male aggressiveness, women wearing burkas, etc.). Since media coverage of terrorist events often draws a link to Islam and creates a harmful association (Sutkute, 2019), such a stereotype could have been reflected in the samples. However, this was not the case – at least in the sample generated for this paper.  
  
  <div align="center"> 
  
<img width="650" alt="figure 2" src="https://user-images.githubusercontent.com/55432992/205193941-b2b5139e-4aa4-4a50-bf05-e2adb03758e4.jpeg">
  
  
  *Figure 2: Examples of images generated using the prompt ‘A photo of the face of a ___’ alongside four belief systems: Judaism, Sikhism, Shintoism, and Atheism.*
    </div>
    
<p align="justify"> 
For the other religions, there were fewer stereotypically distinct features identifiable in terms of traditional stereotypes. Only a small number of Shintoists were shown wearing traditional Japanese clothing, while Jews for example were not portrayed with side curls or headwear, which are more common amongst orthodox Jews. Finally, Christians and Atheists displayed no distinguishing features that may indicate their religious orientation.  
<p align="justify"> 
In terms of ethnicity, most religions were represented by their majority cultural origin. Muslims appeared Middle Eastern; Sikhs and Hindus were of Indian descent; followers of Shintoism were Japanese; and Buddhists Eastern Asian. Jews and Atheists were all white, while Christians were the only religion to have images of black, mixed and white people (but no Asian-seeming people). As with traditional clothing, the ethnicity of people following religions aside from Christianity is amplified to be limited to only the region of the respective religion’s origin. The aspect of a ‘World Religion’ is therefore only applied to Christianity, while the others are depicted as less diverse. However, when searching for a ‘religious person’ all ethnicities are represented, but with the majority including either Christian symbols or clothing associated with Islam.

#### 4.2 Complex Biases in Diffusion Outputs
<p align="justify"> 
Aside from perpetuating – and at times amplifying – inequalities and stereotypes prevalent in the wider population, the model outputs also exhibited signs of what Bianchi et al (2022) call ‘complex biases’ that draw on ‘more pernicious assumptions and power relations’ (p. 3). Within this analysis, this meant that the model seemed to associate certain religious groups with a particular set of backgrounds, attitudes or circumstances. Perhaps most visibly, this was apparent in the case of Judaism: while the model did seem to fulfil the prompt’s request for a photo-realistic image, the pictures it generated distinctly resembled early 20th century monochrome portrait photographs (Figure 2). As a result, most of the generated representations of Jews were not only exclusively masculine, but also lacked full colour and included outdated fashion and posing. This tendency to represent Judaism as archaic can be thought of as a complex bias also because it could not be mitigated by prompt re-writing (Bianchi et al, 2022, p. 9). Even upon adding modifying words such as ‘young’ or ‘modern’ to the prompt, most generated images remained black and white and continued exhibiting archaic features (Figure 3). Meanwhile, the historicization of Judaism has been rejected by various Jewish organisations, who claim that recognising Judaism only in relation to its fraught and traumatic history is reductive (Skrodzka et al, 2022) and serves to further marginalise the religion as it is practised today. 
  
  <div align="center"> 
  
<img width="650" alt="figure 3" src="https://user-images.githubusercontent.com/55432992/205194066-f14d089f-5366-4151-86fc-858c4e2fb20a.jpeg">
  
  
  *Figure 3: Examples of images generated by modifying the initial prompt with specifications of time period or gender.*
    </div>
    
<p align="justify"> 
The model’s representations of Atheism were also notable in terms of complex bias. Not only were the generated faces exclusively masculine, they also appeared particularly uniform in terms of profile and expression: they were predominantly caucasian, middle-aged, with long brown hair and moderate facial hair. Relatedly, it may be interesting to note that in multiple of the religious samples, the faces generated appeared to resemble the relevant religion’s traditional depictions of its holy figures. The Atheists, though by definition unaffiliated with Christianity, seemed to resemble frequent depictions of Jesus, while the Buddhist representations were pictured with shaved heads and closed eyes in close resemblance to Buddhist statues (which the model also mistook for humans).  The Atheism sample was also the only one where the majority of generated faces expressed discernible emotion – most faces were smiling. 

   <div align="center"> 
  
<img width="650" alt="figure 4" src="https://user-images.githubusercontent.com/55432992/205194207-2b707396-f0be-4327-a88b-7d856d8aaf8f.jpeg">
  
  
  *Figure 4: Examples of images generated by modifying the initial prompt to refer to religious or normative status without specifying a belief system.*
    </div>
  
  
<p align="justify"> 
Complex biases also appeared to be at play when using prompts with religious or normative phrases without an affiliation to a specific religion. Almost all images of a ‘devoted person’ depict women, which suggests an amplification of a stereotype that sees women as relatively more peaceful and virtuous compared to men. Furthermore, devotion could also be seen as a devotion to husbands, which further points towards the stronger focus on women and their ‘pure and subordinated’ role in society. Finally, when depicting a ‘good person’ most of the outputs resembled children of both genders. This could be based on the idea of the innocence of childhood. Children are seen as pure and not spoilt by the troubles in life. Seemingly, happiness is also connected to being a good person, as most people are depicted with a smile (Figure 4), which was generally not observed in most religious categories – except for Atheism, suggesting a peculiar association of Atheism to these more explicitly positive queries.

#### 4.3 Limited Visual Reasoning Skills
<p align="justify"> 
In addition, what becomes apparent in this analysis are the model’s limitations in terms of visual reasoning skills. Taking inspiration from the evaluation model proposed in Cho et al (2022), this analysis understands ‘visual reasoning skills’ as the model’s ability to correctly identify objects, count their quantities, and accurately portray the relation between them (p. 3). This is admittedly particularly difficult when dealing with religion as a complex social phenomenon often expressed through a nuanced network of visual symbols. The Stable Diffusion system struggled with identifying certain artefacts and religious signifiers as humans. Particularly in the case of Buddhism, when given the prompt ‘A photo of the face of a Buddhist’, 85% of all generated images were of photo-realistic pictures of statues (Figure X). While this mistake was most prevalent with Buddhist representations, some images generated in other samples – notably those for Christianity – also seemed to mistake artworks for humans. Some features of these outputs suggested that the model indeed did not recognise these as objects and treated them as it would humans, adding clothing items or facial expressions. The model also struggled with generating accurate religious accessories, particularly if their use tends to be gender-specific. When working with headscarves (in the case of Islam or Sikhism) or facial markings (in the case of Hinduism), it frequently pictured masculine faces with accessories usually worn by women of the faith. Upon adding ‘women’ to the prompt, the model became much more accurate at depicting gender-specific attire – though some mistakes prevailed (Figure 3)
  
  
<a name="discussion"></a>
## 5. Discussion
<p align="justify"> 
This Section engages in a two-tiered discussion. First, drawing on the literature review and on AI ethics, it focuses on the religious biases detected, outlining how they are harmful and exploring potential sources for them. Thereafter, it broadens the scope, framing these religious biases within Stability AI’s core values. This aims to understand how Stability AI’s designer subjectivities and ethical preferences influence the company’s perspective of, and approach to, religious biases in Stable Diffusion. 

#### 5.1 Religious Biases: What harm, to whom, and why?
<p align="justify"> 
Blodgett et al. (2020) highlighted the importance of specifying the normative assumptions made when working on algorithmic bias: why is a behaviour biassed, what harm does it create, and to whom? In the behaviours analysed above, the outputs are biassed when they amplify religious stereotypes held from a Western perspective, perpetuate essentializing gender attributes (e.g., women as ‘devoted’), and propagate hierarchies that position the West in the centre. On this point, it is noteworthy that the two belief systems most associated with the West – Christianity and Atheism – are the only ones not characterised by an ethnicity or outward symbol. Arguably, this mirrors the view of Western phenomena as the norm, in what could constitute visual exnomination (Crawford, 2017): the Western belief systems are perpetuated as the norm by not specifying them as a category through, in this case, visual cues. Our findings of religious bias in Stable Diffusion mirror prior findings of Western cultural bias in the model (see e.g. Bansal et al. 2022, Struppek et al. 2022).
<p align="justify">  
These biases generate representational harms, which are damaging in their own right and may furthermore lead to allocational harms downstream (Blodgett et al., 2020). The representational harms observed include stereotyping that impacts the narratives constructed about social groups (e.g., the historicization of Judaism); differences in system performance (e.g. in the previous analysis, Stable Diffusion performed disproportionately worse in generating the face of a Buddhist person, in 85% of cases it generated a statue); and misrepresenting social groups' distribution. The latter occurs both from a gender axis, by underrepresenting women, and from an ethnicity axis, as the outputs homogenise non-Christian religions when their ethnic make-up may be much more diverse. This is most evident regarding Islam, a major religion in countries as different as Senegal, Indonesia, and Bosnia and Herzegovina. Notably, however, the analysed outputs did not propagate some of the most common negative stereotypes against Muslims held in the West.
<p align="justify">   
Regarding the potential sources for these biases, it is possible that some outputs simply reflect systematic misrepresentations in the training data. It could be that within the LAION-5B dataset, certain religious groups are represented by a majority of outdated, historical images (e.g. Judaism) or that some religion-related labels are disproportionately assigned to religious objects or symbols as opposed to people (considering outputs representing Buddhism). Stability AI has been very open about the fact that Stable Diffusion was trained on a database subset that is limited to the English language, impacting the quality of outputs for non-English speaking cultures (Stability AI, n.d.). Alternatively, some of the outcomes could be hypothetically explained by label bias, as conceptualised in the literature review. The analysis found that sometimes the model exaggerated religious symbols. This could derive from human labellers only labelling something as 'religious' when it carries a significant amount of the signs of that religion, leading the system to yield an exaggerated and stereotypical representation of religious symbols in its generations. 
<p align="justify"> 
Beyond the dataset, Stability AI has noted that the provided weights themselves may be biassed, explicitly advising against using them for services or products (i.e., for downstream, 'real-life' applications) without additional safeguards (Stability AI, 2022d). Finally, Stability AI has explicitly warned that Stable Diffusion’s biases set white and Western culture as the default (e.g., the outputs for 'wedding' are Western-style weddings) (Stability AI, 2022c). This could be interpreted in terms of research design bias, as conceptualised in the review. 


#### 5.2 Stability AI's Approach to Religious Bias
<p align="justify">   
While Stability AI acknowledges algorithmic bias in Stable Diffusion, and the ensuing ethical issue of fairness, it operationalises them in a very distinct way, especially when contrasted with the space’s other big players, like Open AI’s DALL•E 2 or Google’s Imagen. For instance, DALL•E 2 is closed sourced and has numerous guardrails for pre-generation moderation; two choices that OpenAI justified citing ethical concerns (Tang et al., 2022). In comparison, Stable Diffusion is open source: users can access the code, the weights, create forks and, all in all, use Stable Diffusion with few compulsory safety measures, excluding a pre-generation keyword filter to prevent specific uses like antisemitic iconography (Stability AI, 2022a).
<p align="justify">   
Stable Diffusion's technical choices are underpinned by a normative view that prioritises openness over other values (or, rather, argues that community-wide building is the best way of reaching other values, including fairness). Stability AI quotes democratising AI as its primary motivation (Wiggers, 2022). It views AI as a technology that can “solve some of humanity’s biggest challenges” (Stability AI, 2022b) – bordering on tech solutionism – but only if made accessible to everyone. Stability AI acknowledges that bias propagation is a negative consequence of open access. Indeed, users may take Stable Diffusion's outputs and amplify harmful stereotypes. Notwithstanding, with a utilitarian view of AI ethics (Vallor, 2018), it argues that the open-source approach will lead to a net benefit (Biewald, 2022). In this sense, it predicates trust in the community, even for fairness concerns: ethical issues will be tackled by the community and/or by regulators, but they "should not be decided in San Francisco'' (Biewald, 2022, n.d.) by a small number of private companies who, through close-sourcing and a paternalistic approach to AI ethics (Wiggers, 2022), limit how much the public benefits from AI. In this sense, Stability AI views itself as a community catalyst and does not want to be made responsible for managing it – to the extent that Stability AI, who currently provides selected community members with computing power for large projects that use Stable Diffusion, advocates for making Graphics Processing Unit (GPU) a public good, allocated nationally and internationally (Biewald, 2022).
<p align="justify">   
The ideals of openness and community-building permeate Stability AI's approach to cultural and religious biases, too. If the released Stable Diffusion model is culturally biassed because of English-language and Western-centric data it is trained on, open-sourcing the model is arguably the key to enable individuals from other countries and cultures to replicate it in a way that is more representative of their culture, using data that has better quality for these purposes (Kilcher, 2022). Importantly, this implies that Stability AI’s vision for Stable Diffusion is not of a large generalist model (Kilcher, 2022). In our examples, if the historicization of Judaism is due to the absence in the training data of more modern depictions then, from  Stability AI’s perspective, a culturally specific model and a more accurate training database would improve the problem. 
<p align="justify">   
This approach to cultural bias – and to religious bias, as concerns this paper – raises the key question of whether there is meaningful opportunity to retrain Stable Diffusion. Stability AI’s vision of decentralised communities building their own culturally-attuned models arguably requires, beyond modifying the code and weights of a trained algorithm, retraining it. At first glance, this is an expensive endeavour: Stable Diffusion v1 reportedly cost 600.000€ to train (Mostaque, 2022). At the same time, Stability AI provides grants and computing power for selected projects (Biewald, 2022). Moreover, it appears that the community is organically attempting to circumvent these limits and create “retrainable-ish model[s]” (u/Yacben, 2022), with tools like Invoke AI (Foong, 2022) or fast-stable-diffusion (TheLastBen, 2022). These efforts would be possible in other generative models “if one only had the same kind of extraordinary access to them that Stability.ai has allowed by open-sourcing Stable Diffusion” (Anderson, 2022). While it remains unclear whether “retrainable-ish” is enough to achieve Stability AI’s purported solution to cultural biases, these efforts arguably help paint its open-sourced approach in a positive light.

<a name="conclusion"></a>
## Conclusion
<p align="justify"> 
Since its release in August 2022, Stable Diffusion’s open-sourced approach has been the topic of many conversations around the future of generative AI. Preliminary research soon found the presence of gender, racial, and cultural bias in Stable Diffusion. This paper has taken this research further by focusing on religious bias: a dimension of cultural bias which, to our knowledge, was unexplored. Following a qualitative bias analysis using elaborative coding, we have found that Stable Diffusion not only perpetuates and amplifies stereotypes, but it also exhibits signs of complex biases and poor visual reasoning skills. Our findings point to a Western bias that generates representational harms for religious minorities in the West, including misrepresentations and disproportionally worse performance. 
<p align="justify"> 
While pinpointing the exact cause for the religious biases observed exceeds the scope of this paper, our analysis has demonstrated that biases – and their related harms – can be clearly observed in the case of religion. As noted in Section 5, it is possible that the skew observed originates in stereotypes inherent to the model’s training data, yet we cannot be certain: while the Stability AI team has embraced a principle of transparency, the datasets they use remain difficult to audit (Baio, 2022). Unbiasing the foundational data at pre-processing level has been the primary concern in recent years. Bender & Friedman (2018), Gebru et al. (2018) and Hutchison et al (2021) advocate for developing ‘data statements’ and ‘data sheets’ to increase transparency, where the data curator would summarise and motivate the dataset’s characteristics and creation methods. These are of crucial importance especially in open-source databases such as LAION-5b, and could shed light when tracking the origins of biases like those identified here. 
<p align="justify"> 
Finally, this paper has situated the religious biases observed within Stability AI’s broader approach to bias. Enshrining openness as Stable Diffusion’s core value has impactful consequences on how Stability AI views bias: the belief is that, while Stable Diffusion may indeed perpetuate bias, in the long term, the community-based approach will produce a net benefit. In the case of cultural and religious biases, in Stability AI’s view, these principles should allow communities themselves to reproduce the model in a way that is attuned to them. Notwithstanding this forward-looking vision, the question remains of whether Stability AI has meaningfully engaged with the negative consequences that are currently taking place. Will the effects of perpetuating religious stereotypes truly be outweighed?

<a name="bibliography"></a>
## Bibliography
Anderson, M. (2022, October 3). Custom Styles in Stable Diffusion, Without Retraining or High Computing Resources. Metaphysic.ai. Retrieved from https://metaphysic.ai/custom-styles-in-stable-diffusion-without-retraining-or-high-computing-resources/ [Last accessed 01/12/2022]
  
Auerbach, C., & Silverstein, L. B. (2003). Qualitative data: An introduction to coding and analysis (Vol. 21). NYU press.

  Baio, A. (2022, August 30). Exploring 12 Million of the 2.3 Billion Images Used to Train Stable Diffusion’s Image Generator. Waxy. Retrieved from https://waxy.org/2022/08/exploring-12-million-of-the-images-used-to-train-stable-diffusions-image-generator/ [Last accessed 01/12/2022]

  Bansal, H., Yin, D., Monajatipoor, M. & Chang, K-W. (2022). How well can Text-to-Image Generative Models understand Ethical Natural Language Interventions?  ArXiv preprint, https://doi.org/10.48550/arXiv.2210.15230

  Barbujani, G. & Colonna, V. (2011). Genetic basis of human biodiversity: An update.  In F. Zachos & J. Habel (Eds.). Biodiversity Hotspots. Berlin, Heidelberg: Springer http://dx.doi.org/10.1007/978-3-642-20992-5_6  

  Bender, E. M.  & Friedman, B. (2018). Data statements for natural language processing: Toward mitigating system bias and enabling better science. Trans. Assoc. Comput. Ling. 6 (2018), 587–604. DOI: https://doi.org/10.1162/ tacl_a_00041 

  Bhatia, S. (2017). Associative judgment and vector space semantics. Psychological Review, 124, 1–20. https://www.psychologie.uzh.ch/dam/jcr:867b50aa-8bd2-4103-961c-557d44ccd63d/Bhatia.PsychRev.124.2017.pdf 

  Bianchi, F., Kalluri, P., Durmus, E., Ladhak, F., Cheng, M., Nozza, D., Hashimoto, T., Jurafsky, D., Zou, J., and Caliskan, A. (2022). Easily Accessible Text-to-Image Generation Amplifies Demographic Stereotypes at Large Scale. ArXiv preprint. Available at https://arxiv.org/pdf/2211.03759.pdf-

  Biewald, L. [Weights & Biases] (2022, November 15). Emad Mostaque — Stable Diffusion, Stability AI, and What’s Next [Video file]. Retrieved from https://www.youtube.com/watch?v=bG5hTokyh5Q&t=2s [Last accessed 30/11/2022]

  Blodgett, S.L., Barocas, S., Daumé, H., III, & Wallach, H. (2020). Language (Technology) is Power: A Critical Survey of “Bias” in NLP. ArXiv preprint, https://doi.org/10.48550/arXiv.2005.14050 

  Cho, J., Zala, A., and Bansal, M. (2022). DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformers. ArXiv preprint, abs/2202.04053. Available at https://arxiv.org/pdf/2202.04053.pdf.

  Crawford, K. (2017). The Trouble with Bias [Panel]. 2017 Conference on Neural Information Processing Systems (NIPs 2017). Retrieved from https://www.youtube.com/watch?v=fMym_BKWQzk [Last accessed 30/11/2022]

  Crawford, K. & Paglen, T. (2019, September 19). Excavating AI: The Politics of Training Sets for Machine Learning. Excavating AI. Retrieved from https://excavating.ai [Last accessed 01/12/2022]

  Flek, L. (2020). Returning the N to NLP: Towards contextually personalized classification models. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 7828–7838. Online: Association for Computational Linguistics. https://www.aclweb.org/anthology/2020.acl-main.700 

  Foong, N.W. (2022, August 31). How to Fine-tune Stable Diffusion using Textual Inversion. towards data science. Retrieved from https://towardsdatascience.com/how-to-fine-tune-stable-diffusion-using-textual-inversion-b995d7ecc095 [Last accessed 01/12/2022]

  Garg N., Schiebinger L., Jurafsky D., & Zouu J., (2018). Word embeddings  quantify 100 years of gender and ethnic stereotypes. 2018 Proceedings of the National Academy of Sciences, 115(16), 3635-3644. https://www.pnas.org/doi/full/10.1073/pnas.1720347115 

  Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., Crawford, K. (2018). Datasheets for datasets. Communications of the ACM, 64(12), 86-92. http://dx.doi.org/10.1145/3458723 

  Gonen, H., & Goldberg, Y. (2019). Lipstick on a pig: Debiasing methods cover up systematic gender biases in word embeddings but do not remove them. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 609–614. https://www.aclweb.org/anthology/N19-1061 

  Hachman, M. (2022, August 26). The new killer app: Creating AI art will absolutely crush your PC. PC World. Retrieved from https://www.pcworld.com/article/916785/creating-ai-art-local-pc-stable-diffusion.html [Last accessed 01/12/2022]

  Hovy, D., & Søgaard, A. (2015). Tagging performance correlates with author age. Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), 483–488. 

  Hovy, D., & Yang, D. (2021). The importance of modeling social factors of language: Theory and practice. Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 588–602. Online: Association for Computational Linguistics. https://www. aclweb.org/anthology/2021.naacl-main.49 

  Joshi, P., Santy, S., Budhiraja, A., Bali, K., & Choudhury, M. (2020). The state and fate of linguistic diversity and inclusion in the NLP world. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 6282–6293. Association for Computational Linguistics. https://www.aclweb.org/anthol- ogy/2020.acl-main.560 

  Labov, W. (1972). Sociolinguistic patterns. Philadelphia: University of Pennsylvania Press.

  Mansimov, E., Parisotto, E., Ba, J.L., & Salakhutdinov, R. (2015). Generating images from captions with attention. ArXiv preprint, https://arxiv.org/pdf/1511.02793.pdf 

  Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., Galstyan. A. (2021). “A Survey on Bias and Fairness in Machine Learning”. ACM Comput. Surv. 54(6). 1-35. https://doi.org/10.1145/3457607  

  Mostaque, E. [@EMostaque] (2022, August 28). We actually used 256 A100s for this per the model card, 150k hours in total so at market price $600k [Tweet]. Twitter. Retrieved from https://twitter.com/EMostaque/status/1563870674111832066 [Last accessed 01/12/2022]

  Open AI (n.d.). Model Card: CLIP. Hugging Face. Retrieved from https://huggingface.co/openai/clip-vit-large-patch14 [Last accessed 01/12/2022]
Quillian, L., & Pager, D. (2010). Estimating risk: Stereotype amplification and the perceived risk of criminal victimization. Social Psychology Quarterly, 73(1), 79-104. 

  Reed, S., Akata, Z., Yan, X., Logeswaran, L., Schiele, B., & Lee, H. (2016). Generative Adversarial Text to Image Synthesis. ArXiv preprint, arXiv:1605.05396 

  Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. ArXiv preprint, https://arxiv.org/pdf/2112.10752.pdf 

  Sap, M., Card, D., Gabriel, S., Choi, Y., & Smith, N. A. (2019). The risk of racial bias in hate speech detection. Proceedings of the 57th Conference of the Association for Computational Linguistics, 1668–1678. Florence, Italy: Association for Computational Linguistics. https://www.aclweb.org/anthology/P19-1163 

  Skrodzka, M., Kende, A., Faragó, L. & Bilewicz, M. (2022). “Remember that we suffered!” The effects of historical trauma on anti-Semitic prejudice. Journal of Applied Social Psychology, 52, 341– 350. https://doi.org/10.1111/jasp.12862 

  Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N. &amp; Ganguli, S.. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics. Proceedings of the 32nd International Conference on Machine Learning, in Proceedings of Machine Learning Research, 37:2256-2265 Available from https://proceedings.mlr.press/v37/sohl-dickstein15.html 

  Stability AI (2022a, November 24). Stable Diffusion License Model. GitHub. Retrieved from https://github.com/Stability-AI/stablediffusion/blob/main/LICENSE-MODEL [Last accessed 30/11/2022}

  Stability AI (2022b, October 17). Stability AI announces 101 million in funding for open-source Artificial Intelligence. PRNewswire. Retrieved from https://www.prnewswire.com/news-releases/stability-ai-announces-101-million-in-funding-for-open-source-artificial-intelligence-301650932.html [Last accessed 30/11/2022]

  Stability AI (2022c, August). StableDiffusion Read Me. GitHub. Retrieved from https://github.com/CompVis/stable-diffusion/blob/main/README.md [Last accessed 30/11/2022]

  Stability AI (2022d, June). Stable Diffusionv1 Model Card. GitHub. Retrieved from https://github.com/CompVis/stable-diffusion/blob/main/Stable_Diffusion_v1_Model_Card.md [Last accessed 30/11/2022]

  Stability AI (n.d.). What was the Stable Diffusion Model trained on? Stable Diffusion FAQ. Retrieved from https://stability.ai/faq [Last accessed 30/11/2022].

  Struppek, L., Hintersdorf, D. & Kersting, K. (2022). The Biased Artist: Exploiting Cultural Biases via Homoglyphs in Text-Guided Image Generation Model. https://doi.org/10.48550/arXiv.2209.08891

  Sutkutė, R. (2019). Media, stereotypes and muslim representation: world after Jyllands-Posten Muhammad cartoons controversy. EUREKA: Social and Humanities, (6), 59-72. 

  Tang, R., Pandey, A., Jiang, Z., Yang, G., Kumar, K., Lin, J. & Ture, F. (2022). What the DAAM: Interpreting Stable Diffusion Using Cross Attention. https://doi.org/10.48550/arXiv.2210.04885

  TheLastBen (2022). Fast-stable-diffusion. Github. Retrieved from https://github.com/TheLastBen/fast-stable-diffusion [Last accessed 01/12/2022]
u/Yacben (2022, October 25). New (simple) Dreambooth method is out, train under 10 minutes without class images on multiple subjects, retrainable-ish model [Reddit post]. Retrieved from https://www.reddit.com/r/StableDiffusion/comments/yd9oks/new_simple_dreambooth_method_is_out_train_under/ [Last accessed 01/12/2022]

  Vallor, S. (2018). An Introduction to Data Ethics. Markkula Center for Applied Ethics at Santa Clara University. Retrieved from https://www.scu.edu/media/ethics-center/technology-ethics/IntroToDataEthics.pdf [Last accessed 1/12/2022] 
