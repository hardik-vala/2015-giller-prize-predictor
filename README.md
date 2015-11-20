# 2015-giller-prize-predictor

My simple [2015 Giller Prize](http://www.scotiabankgillerprize.ca/) predictor.

### How it works

predict.py is well-documented and reading through the main method will show you how my method works. Running predict.py performs a type of grid-search through model parameters and spits out the prediction for both the 2014 and 2015 Giller Prize contests.

My model predicts Russell Smith's _Confidence_ as the winner, but if you consider a 4% margin of error, it's really a 5-way tie between Michael Christie's _If I Fall, If I Die_, Marina Endicott's _Close to Hugh_, Clifford Jackman's _The Winter Family_, and Anakana Schofield's _Martin John_. (There are 12 nominations in total so this doesn't quite bode well.)

UPDATE: Andre Alexis' _Fifteen Dogs_ was announced as the winner of the 2015 Giller Prize and surprisingly, my model predicted this novel as winning with the least confidence. So if you invert my model, you have yourself a Giller prize predictor! Jokes, it just entails I have alot more work to do for the 2016 Giller Prize.

### Data

Unfortunately the data, i.e. the set of training, validation, and test novel texts, is not included in the repository and cannot be released for copyright reasons. Regardless, hopefully this repository serves useful for your own computational predictors for literature prize winners.

### License

MIT
