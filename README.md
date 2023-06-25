# predict-crisis-supplychain
Use machine learning models to predict possible risks in supply chain networks by using disaster tweets datasets

The main goal of this project is to use disaster tweets datasets to train machine learning models that could tell us if a disruption in supply chain is expected to happen or not. Since it's hard for me to deploy and test the models in real-world supply chain systems, I will look into a metric called <b>"reaction time"</b>, which basically determine how quick the model can detect disruption after a disaster happens. In real-world supply chain systems, you would likely to be looking into some other important KPIs such as:
<ul> 1. Model's ability to respond to disruption compared to manual responses </ul>
<ul> 2. Model's ability to increase early risk detection (%) </ul>
<br><br>
<h2>Dataset</h2>

The dataset is retrieved from the website [CrisisNLP](https://crisisnlp.qcri.org/lrec2016/lrec2016.html), where contained crisis-related tweets ranging from earthquakes, typhoons to airline accident. I used their labeled dataset by paid workers as the labeled dataset by volunteers have lots of non-relevant labels for our use case. 
<br>
<br>
<p>There are 3 custom labels that I will assign all the labels to - <b>"Disruption", "Potential disruption" and "No disruption".</b> The intuition behind having 3 instead of 2 custom labels is that I believe there will be times where the crisis doesn't lead to heavy damage so there will be little to no effects to supply chain, and these will be categorised as "potential disruption".</p>
<p>Let's take a look at the labels mapping I did: </p>


````
Disruption:
1. `infrastructure_and_utilities_damage`: Damage to infrastructure/utilitise directly impacts supply chain, pretty self-explanatory
2. `displaced_people_and_evacuations`: Could possibly imply a serious event that disrupt supply chains
3. `affected_people`: A situatioin where large number of people being affected by a disaster could impact workforce availability and demand patterns
4. `missing_trapped_or_found_people`: This would probably indicate a severe crisis such as earthquake that causes trapped/missing people
5. `death_reports`: Reporting of deaths probably indicate severe disasters which could disrupt supply chain
6. `injured_or_dead_people`

Potential Disruption:
1. `disease_transmission`: Depends on the widespread severity of the disease, so I put this in potential disruption category.
2. `prevention`: Could indicates lockdown or restrictions which is indicative of supply chain disruption.
3. `caution_and_advice`: If there's caution and advice given on incoming disaster, this could indicate a potential disruption.
4. `disease_signs_or_symptoms`: Quite a vague label, but since its linked to disease so I'll put it into potential disruption.
5. `donation_needs_or_offers_or_volunteering_services`: Most involve donations, volunteers, offers of help which could be indicative of an ongoing/potential disruption
6. `treatment`: suggests actions needed to address a crisis. The tweet content contained a mixed of information, but it could potentially cause disruption due to a need in aid and assistance
7. `other_useful_information`: I looked at the some tweet texts with this label and seemed like it contained a mix of information where some could be relevant to SP disruption, but some doesn't, but I will put this into disruption atm due to the natura of the crisis dataset

No disruption:
1. `not_related_or_irrelevant`
2. `sympathy_and_emotional_support`: Focused on emotional support and solidarity, which doesn't directly indicate a disruption
````

<h2>High-Level Approach</h2>

![approach.png](https://github.com/owaikien/predict-crisis-supplychain/blob/b991c08d9254690f046ccaf1e20d6cb7ba08b15f/approach.png)

<p>
The approach was fairly straightforward I believe. After mapping the dataset labels to our custom labels, I will pre-process the dataset by removing URLs, punctuations, stopwords as well as perform text tokenization and lemmatization.
</p><br/>
For text vectorization, I've used `tf-idf` in this case. Perhaps using BERT might be a better choice.

<h2>Model Building</h2>
<p>I trained 4 models: logistic regression, Decision Tree, Support Vector Machine and Gradient Boosting. Logistic Regression performed the best as it achieved a weighted F1-score of 0.80. However, for the "disruption" class it has only a recall score of 0.74 which was relatively lower compared to other classes. I believe recall should be an important metric as we wanted to catch all the actual disruptions.</p>

````
Classification report:
                      precision    recall  f1-score   support

          disruption       0.86      0.74      0.79      1108
       no_disruption       0.80      0.82      0.81      1346
potential_disruption       0.78      0.83      0.80      1649

            accuracy                           0.80      4103
           macro avg       0.81      0.80      0.80      4103
        weighted avg       0.81      0.80      0.80      4103
````
<h2>Model Testing</h2>

I then tested the model on the [Turkey and Syria Earthquake Tweet dataset](https://www.kaggle.com/datasets/swaptr/turkey-earthquake-tweets), which is retrieved from Kaggle. The way I tested the model is to look the the first tweet where the model predicted `disruption` and `potential disruption`, look at the timestamp of that tweet and find out the time difference between that timestamp and the official happening time of the earthquake. Here's what I found:

`Disruption`
<p>Reaction Time: <b>~7 minutes</b></p>

````
	date	                        content	                                                predictions     probability
0	2023-02-06 01:24:33+00:00	6 minago earthquake 77 hit gaziantep turkey 62...	disruption	0.249040
1	2023-02-06 01:30:54+00:00	thing missing syria crisis earthquake            	disruption	0.342555
2	2023-02-06 01:35:05+00:00	huge 78 earthquake gaziantep turkeymy sisterin...	disruption	0.110869
3	2023-02-06 01:35:09+00:00	whole building shaking along earthquake lebanon	        disruption	0.252122
4	2023-02-06 01:38:39+00:00	earthquake felt beirut lebanon 318am aftershoc...	disruption	0.086312
````

`Potential disruption`

<p>Reaction Time: <b>~37seconds</b></p>

````
	date	                        content              	                                predictions             probability
0	2023-02-06 01:17:37+00:00	earthquake séisme m28 strike 24 km se monaco m...	potential_disruption	0.127280
1	2023-02-06 01:19:42+00:00	⚠preliminary info earthquake deprem 30 km n me...	potential_disruption	0.053286
2	2023-02-06 01:21:33+00:00	⚠preliminary info m72 earthquake deprem 30 km ...	potential_disruption	0.030897
3	2023-02-06 01:21:53+00:00	scary earthquake cyprus	                                potential_disruption	0.157591
4	2023-02-06 01:22:17+00:00	earthquake nicosia right	                        potential_disruption	0.183503
````
<p>Insights:</p>

1. Model Performance: The model performance in predicting disruptions seemed reasonable to me, although the prediction probabilities aren't that high ranges (below 0.50), which implies the model is still quite unsure about its predictions. 
2. Detection of Actual Disruptions: The first tweet that got predicted as 'disruption' was identified approximately ~7minutes after official earthquake time, which I think was a considerably fast.
3. Detection of Potential Disruption: Interestingly, the model was able to detect a 'potential disruption' just 37 seconds after the earthquake. This suggests the model is able to quickly detect signs of disruption, even if it's not immediately sure if a full disruption has occured. 

Some further improvements could be: 
1. Additional model traning / tuning
2. The custom labels were set by me, perhaps it might be better to be reviewed by experts and I suspect this is also the main reason why the prediction probabilities are pretty low
3. Perhaps it will be interesting to use pre-trained language models like BERT for text vectorization
