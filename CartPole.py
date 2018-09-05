import gym
import random
import numpy as np
import keras
from statistics import median, mean
from collections import Counter
from keras.layers import (Input,Dense, Dropout, Flatten,Reshape)
from keras.optimizers import adam
#from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential


LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500
score_requirement = 55
initial_games = 50000


def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0, 2)

            observation, reward, done, info = env.step(action)



            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done: break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:

                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]


                training_data.append([data[0], output])


        env.reset()

        scores.append(score)


    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data







def keras_model(input_size):

    model = Sequential()



    model.add(Dense(128, activation="relu" , input_shape= ( input_size ,1)))


    model.add(Dense(256, activation="relu"))



    model.add(Dense(512, activation="relu"))


    model.add(Dense(256, activation="relu"))



    model.add(Dense(512, activation="relu"))



    #model.add(Reshape((1, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation ="linear"))
    model.add(Dense(2, activation='sigmoid'))
    addam =  adam(lr= 0.001 )
    model.compile(loss = 'categorical_crossentropy' , optimizer= addam )
    model.summary()

    return model


def train_model(training_data):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = np.array([i[1] for i in training_data])
    print(X.shape)
    print(y.shape)
    print(y[1,:])


    model = keras_model(input_size= len(X[0]))

    #model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True)
    model.fit(X,y,nb_epoch=10)
    return model

training_data = initial_population()

model = train_model(training_data)

scores = []
choices = []
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    check = []
    for _ in range(goal_steps):
        env.render()

        if len(prev_obs) == 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
            check.append(model.predict(prev_obs.reshape(-1, len(prev_obs), 1)))
        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done: break

    scores.append(score)

print('Average Score:', sum(scores) / len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
print(score_requirement)
print(check)
