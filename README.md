# predict-crisis-supplychain
Use machine learning models to predict possible risks in supply chain networks by using disaster tweets datasets

The main goal of this project is to use disaster tweets datasets to train machine learning models that could tell us if a disruption in supply chain is expected to happen or not. Since it's hard for me to deploy and test the models in real-world supply chain systems, I will look into a metric called <b>"reaction time"</b>, which basically determine how quick the model can detect disruption after a disaster happens. In real-world supply chain systems, you would likely to be looking into some other important KPIs such as:
<ul> 1. Model's ability to respond to disruption compared to manual responses </ul>
<ul> 2. Model's ability to increase early risk detection (%) </ul>
<br>
<h2>Dataset</h2>
<p>The dataset is retrieved from the website [CrisisNLP](https://crisisnlp.qcri.org/lrec2016/lrec2016.html), where contained crisis-related tweets ranging from earthquakes, typhoons to airline accident. I used their labeled dataset by paid workers as the labeled dataset by volunteers have lots of non-relevant labels for our use case. </p>
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
