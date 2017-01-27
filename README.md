# This seminar was created to get familiar with modern RL

### mipt_course 
contains solutions for tasks proposed by [RL course at MIPT](info.deephack.me), which is based on [David Silver's course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html). All tasks use OpenAI gym environments.

### deephack 
contains our attempts to solve [Skiing](https://gym.openai.com/envs/Skiing-v0) game, a problem for qualification round of [DeepHackLab hackathon](http://rl.deephack.me/). Core of model consists of training convolutional autoencoder with dense layers in bottleneck. Before trainig, we convert images from RGB to greys and compress it to 60x60. With autoencoder, we are obtaining ability to get low-dimensional features for images (64, basically). Code presented in autoencoder_simple_features.ipynb.

Then, we have 3 main directions of evolution:

1. parametrize agent's policy and use policy gradient algorithms, e.g. Monte-Carlo Policy Gradient (REINFORCE). Code presented in Skiing.ipynb
2. approximate value or action-value function and use epsilon-greedy policy. Code presented in linear_fa.ipynb
3. collect more features, e.g. via object detection. NB: due to competition rules, features should NOT be environment-specific. Code presented in features_demo.ipynb
