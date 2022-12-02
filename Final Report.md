  
# Cultural Biases in Generative AI: Exploring Religious Bias in Stable Diffusion
##B. Bromová, G. Maggi, C. Mautner Markhof, R. Rapparini, L. Zurdo


## Table of Contents
[Introduction](#introduction)

[1 The Stable Diffusion Model](#model)

[2 Literature Review](#litreview)

[3 Methodology](#metho)

[4 Analysis of Outputs](#output)

[5 Discussion](#discussion)

[Conclusion](#conclusion)

[Bibliography](#bibliography)

<a name="introduction"></a>
## 1 Introduction

<p align="justify"> 
From artificial intelligence (AI) art to deep fakes, image-generative algorithms are emerging in discussions that span culture, philosophy and politics. Integral to these conversations are questions about bias and fairness. A frequent example – also taken up by this paper – is Stable Diffusion, launched in 2022 by Stability AI. Stable Diffusion follows the footsteps of Open AI’s DALL•E in providing an machine-learning (ML) system to generate images, yet it differs significantly in terms of approach. From its methodology to its underlying values, StableDiffusion could shape a different way of engaging with algorithmic bias. 
<p align="justify">  
This research work is a case study on Stable Diffusion’s biases when it comes to religion and culture, exploring whether the Stable Diffusion image generation system handles religion in an unbiased way. The paper focuses on religious bias as it has been less explored. Following an explanation of the Stable Diffusion model (Section 1), a literature review (Section 2), and a description of our methodology (Section 3), Section 4 explores the research question from an empirical perspective, analysing image outputs across a standardised set of prompts to detect religious biases. Thereafter, Section 5 discusses the findings, exploring how religious biases interact with Stability AI’s core values. All in all, we find evidence of religious bias in the shape of stereotyping, complex biases, and limited visual reasoning. We further posit that Stability AI frames the discussion in a way that allows it to both acknowledge open-source’s role in amplifying biases and simultaneously claim it as necessary for the solution. 

  
<a name="model"></a>
## 1 The Stable Diffusion Model
<p align="justify"> 
From the early 2010s, progress in the field of Deep Neural Networks has dramatically enhanced the capabilities of AI systems, achieving promising results in areas like image recognition (Krizhevsky et al., 2012), and machine translation (Bahdanau et al., 2015). Against this backdrop, Mansimov et al. (2015) introduced a model capable of generating images from text descriptions. Since then, further improvements was made by applying Generative Adversarial Networks (GANs) – a generative model that builds on noise-contrastive estimation (see Guttman & Hyvarinen 2010) to enable the system to distinguish data from noise – especially through the works of Reed et al. (2016). 
<p align="justify"> 
State of the art systems such as DALL•E and DALL•E 2 use diffusion models (DMs) to generate images – a process which iteratively guesses the amount of Gaussian noise blurring an image, guesses the image by subtracting the noise, and then adds back some noise. Repeating this process improves the resolution of the image until the final version is achieved (Sohl-Dickstein et al. 2015). However, Stable Diffusion uses an improved version of these models – called latent diffusion models (LDMs) – which are less computationally expensive but retain the quality and flexibility of DMs thanks to the use of a latent space instead of a pixel space (Rombach et al. 2022, Siddiqui 2022). Latent diffusion models work by destroying training data through the asymptotical introduction of Gaussian noise and subsequently learning how to reconstruct the data by denoising the process. After the learning phase is complete, it is possible to generate data by passing randomly sampled noise through the denoising process. In principle, this corresponds to learning the inverse process of a Markov chain of length T.
<p align="justify"> 
Notably, the noise generation of these models does not happen on a random basis but it is informed by the text string that the users input – that is to say that the network is conditioned on the text (y) by means of a conditional denoising autoencoder eθ(zt, t, y) (Rombach et al., 2022). The text string is then analysed through a Natural Language Processing (NLP) transformer embedding, which captures the semantic structure of the text. More specifically, StableDiffusion uses a frozen CLIP ViT-L/14 (Contrastive Language-Image Pre-training) text encoder that conditions the model on text prompts. This type of encoder is designed to maximise the similarity between text and images pairs by contrastive loss (Open AI, n.d.). Contrastive loss is a method used to compare output features in Siamese networks, which are a form of encoding network whereby 2 or more inputs are encoded and output features are analysed. The network output is taken as a positive example, and its distance to an example of the same class is then calculated in order to compare and contrast it with the distance to negative examples. In other words, contrastive loss is used to map vectors that reflect the similarity between items and form the basis of machine learning models such as Stable diffusion. Finally, to make the image closely tied to the text string, Stable Diffusion uses a classifier-free guidance (Cfg) (Ho & Salimans 2021) at the end of the network. This generates additional non-conditioned guesses of the noise and subtracts it from the conditioned guess to get a tighter prediction of initial noise and thus an output closer to the users’ prompt.




<a name="litreview"></a>
## 2 Literature Review
<p align="justify"> 
....
  
  <a name="method"></a>
## 3 Methodology
<p align="justify"> 
  
  

<a name="output"></a>
## 4 Analysis of Outputs
<p align="justify"> 

<div align="center"> 
  
  <img width="452" alt="figure 2" src="https://user-images.githubusercontent.com/115728895/205109681-659a9bc5-8577-43a4-9a1e-d150a29ec6de.png">
  
  
  ***Figure 2 - Visibility of test comments on TikTok (by theme). Note: one box represents one test comment.***
    </div>
    
<p align="justify"> 


<a name="discussion"></a>
## 5 Discussion
<p align="justify"> 


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

