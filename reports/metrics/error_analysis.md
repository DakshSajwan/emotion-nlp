# Error Analysis: Baseline vs DistilBERT

This document provides a comparative error breakdown on the 2,000-sample test set of `dair-ai/emotion`.

## 1. Top Confident Discrepancies

### A. Baseline Wrong, DistilBERT Right (Total: 150 instances)
Cases where transformer contextual representations successfully resolved nuanced syntactic or semantic structures that fooled bag-of-words:

| # | Text | True Label | Baseline Pred (Conf) | DistilBERT Pred (Conf) |
|---|---|---|---|---|
| 1 | i feel so amazing musicjuzz | **joy** | surprise (0.99) | joy (0.63) |
| 2 | i feel that passionate about | **joy** | love (0.98) | joy (0.56) |
| 3 | i feel respected and secure where i can journey toward loving and be loved in return | **joy** | love (0.98) | joy (1.00) |
| 4 | i work well with almost every client ive ever been in contact with because i know what it means to feel depressed angry frustrated irritated hopeless and apathetic because i feel it daily | **sadness** | anger (0.97) | sadness (1.00) |
| 5 | i was truly surprised and feel quite honored | **joy** | surprise (0.97) | joy (0.99) |
| 6 | i feel accepted and loved by a community of derby girls that i helped to create | **joy** | love (0.94) | joy (0.64) |
| 7 | i am truly unfortunate the majority of the time i m usually drained but i obtain it hard to get from bed i really feel restless and others | **fear** | sadness (0.91) | fear (1.00) |
| 8 | i realized what i am passionate about helping women feel accepted and appreciated | **joy** | love (0.89) | joy (0.57) |
| 9 | i feel so dirty but after spending a day at the mk show me and a buddy decided we would get the two player starter between us luckily for us both i liked the everblight and he liked the circle maybe a tad to much so it all worked out well | **sadness** | love (0.89) | sadness (1.00) |
| 10 | i started to feel so overwhelmed | **fear** | surprise (0.89) | fear (0.59) |

---

### B. DistilBERT Wrong, Baseline Right (Total: 56 instances)
Cases where classical lexical matching succeeded but the transformer misclassified:

| # | Text | True Label | Baseline Pred (Conf) | DistilBERT Pred (Conf) |
|---|---|---|---|---|
| 1 | i feel like the writer wants me to think so and proclaiming he no longer liked pulsars is a petty and hilarious bit of character | **love** | love (0.56) | anger (0.99) |
| 2 | i walked to school he felt the bounce in his step the overjoyed feelings of youth and the thrill of excitement of coming to school and meeting his beloved friends | **love** | love (0.50) | joy (0.99) |
| 3 | i feel affirmed gracious sensuous and will have less self doubt when a href http generations | **love** | love (0.44) | joy (0.98) |
| 4 | i cant help feeling this way | **sadness** | sadness (0.44) | joy (0.98) |
| 5 | i feel very saddened that the king whom i once quite respected as far as monarchs go was ineffectual at best | **joy** | joy (0.77) | sadness (0.92) |
| 6 | when we rearranged furniture in our flat and got stuck in a chair | **anger** | anger (0.25) | fear (0.92) |
| 7 | i don t feel betrayed coz the backstabber had no grounds for their accusation but i m just amazed at some people s ability to do such things | **surprise** | surprise (0.83) | fear (0.90) |
| 8 | i actually was in a meeting last week where someone yelled at an older lady because her phone rang i felt terrible for her your boss treats you unfairly or in this case someone makes you feel you are not worth anything is only allowing those who offended to steal your joy | **anger** | anger (0.46) | sadness (0.87) |
| 9 | i feel betrayed and angry and sad at the same time dammit | **anger** | anger (0.67) | fear (0.83) |
| 10 | i have reported feeling marginalized intimidated and or subjected to threats of retaliation | **fear** | fear (0.69) | sadness (0.83) |

---

### C. Both Models Wrong (Total: 85 instances)
Intrinsically ambiguous, overlapping, or multi-faceted emotion examples where both classifiers failed:

| # | Text | True Label | Baseline Pred (Conf) | DistilBERT Pred (Conf) |
|---|---|---|---|---|
| 1 | i didn t mean to get angry with you bommie i just can t control my feelings hellip i just hated myself why i am like this the dara who can t get over with that b | **sadness** | anger (0.96) | anger (0.82) |
| 2 | i feel hated in cempaka | **sadness** | anger (0.93) | anger (0.70) |
| 3 | i could feel his breath on me and smell the sweet scent of him | **joy** | love (0.89) | love (0.88) |
| 4 | i never feel like im not supporting | **joy** | love (0.89) | love (0.54) |
| 5 | i loved the feeling i got during an amazing slalom run whether it was in training or in a race | **joy** | surprise (0.87) | surprise (0.78) |
| 6 | i feel like some of you have pains and you cannot imagine becoming passionate about the group or the idea that is causing pain | **joy** | love (0.87) | love (0.56) |
| 7 | i don t know about you but i m feeling amp blessed | **joy** | love (0.87) | love (0.56) |
| 8 | i started walking again yesterday and it feels amazing | **joy** | surprise (0.86) | surprise (0.55) |
| 9 | i feel the need to pimp this since raini my beloved rocky casting director loves it so much | **joy** | love (0.85) | love (0.80) |
| 10 | i don t feel particularly agitated | **fear** | anger (0.84) | anger (0.77) |

---

## 2. Most Frequent Confusion Pairs

### Baseline Most Frequent Confusion Pairs (True -> Pred)
| true_label | pred_label | 0 |
| --- | --- | --- |
| joy | love | 48 |
| sadness | anger | 22 |
| love | joy | 18 |
| fear | surprise | 14 |
| sadness | fear | 13 |

### DistilBERT Most Frequent Confusion Pairs (True -> Pred)
| true_label | pred_label | 0 |
| --- | --- | --- |
| love | joy | 31 |
| joy | love | 19 |
| surprise | fear | 18 |
| anger | fear | 13 |
| anger | sadness | 12 |

### Common Failure Modes Where Both Models Agree on False Predictions
| true | base_pred | dist_pred | 0 |
| --- | --- | --- | --- |
| joy | love | love | 18 |
| love | joy | joy | 13 |
| sadness | anger | anger | 8 |
| surprise | fear | fear | 7 |
| anger | sadness | sadness | 6 |
