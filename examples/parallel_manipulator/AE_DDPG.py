import random
import imageio
import datetime
import numpy as np
from collections import deque
import threading
import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam

'''
Original paper: https://arxiv.org/pdf/1903.00827.pdf
'''

tf.keras.backend.set_floatx('float64')


def actor(state_shape, action_dim, action_bound, action_shift, units=(400, 300), num_actions=1):
    state = Input(shape=state_shape)
    x = Dense(units[0], name="L0", activation='relu')(state)
    for index in range(1, len(units)):
        x = Dense(units[index], name="L{}".format(index), activation='relu')(x)

    outputs = []  # for loop for discrete-continuous action space
    for i in range(num_actions):
        unscaled_output = Dense(action_dim, name="Out{}".format(i), activation='tanh')(x)
        scalar = action_bound * np.ones(action_dim)
        output = Lambda(lambda op: op * scalar)(unscaled_output)
        if np.sum(action_shift) != 0:
            output = Lambda(lambda op: op + action_shift)(output)  # for action range not centered at zero
        outputs.append(output)

    model = Model(inputs=state, outputs=outputs)

    return model


def critic(state_shape, action_dim, units=(48, 24), num_actions=1):
    inputs = [Input(shape=state_shape)]
    for i in range(num_actions):  # for loop for discrete-continuous action space
        inputs.append(Input(shape=(action_dim,)))
    concat = Concatenate(axis=-1)(inputs)
    x = Dense(units[0], name="L0", activation='relu')(concat)
    for index in range(1, len(units)):
        x = Dense(units[index], name="L{}".format(index), activation='relu')(x)
    output = Dense(1, name="Out")(x)
    model = Model(inputs=inputs, outputs=output)

    return model


def update_target_weights(model, target_model, tau=0.005):
    weights = model.get_weights()
    target_weights = target_model.get_weights()
    for i in range(len(target_weights)):  # set tau% of target model to be new weights
        target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
    target_model.set_weights(target_weights)


def is_threads_alive(threads):
    for t in threads:
        if t.isAlive():
            return 1
    return 0


class AE_DDPG:
    def __init__(
            self,
            env_name,
            num_envs=5,
            discrete=False,
            lr_actor=1e-4,
            lr_critic=1e-3,
            actor_units=(12, 8),
            critic_units=(12, 8),
            sigma=0.3,
            tau=0.125,
            gamma=0.85,
            rho=0.2,
            batch_size=128,
            memory_cap=1500,
            hmemory_cap=1000,
            cache_size=500
    ):
        self.env = [gym.make(env_name) for _ in range(num_envs)]
        self.num_envs = num_envs  # number of environments for async data collection
        self.state_shape = self.env[0].observation_space.shape  # shape of observations
        self.action_dim = self.env[0].action_space.n if discrete else self.env[0].action_space.shape[0]
        # number of actions
        self.discrete = discrete
        self.action_bound = (self.env[0].action_space.high - self.env[0].action_space.low) / 2 if not discrete else 1.
        self.action_shift = (self.env[0].action_space.high + self.env[0].action_space.low) / 2 if not discrete else 0.

        # Initialize memory buffers
        self.memory = deque(maxlen=memory_cap)
        self.hmemory = deque(maxlen=hmemory_cap)
        self.cache_buffer = [deque(maxlen=cache_size) for _ in range(num_envs)]

        # Define and initialize Actor network
        self.actor = actor(self.state_shape, self.action_dim, self.action_bound, self.action_shift, actor_units)
        self.actor_target = actor(self.state_shape, self.action_dim, self.action_bound, self.action_shift, actor_units)
        self.actor_optimizer = Adam(learning_rate=lr_actor)
        update_target_weights(self.actor, self.actor_target, tau=1.)

        # Define and initialize Critic network
        self.critic = critic(self.state_shape, self.action_dim, critic_units)
        self.critic_target = critic(self.state_shape, self.action_dim, critic_units)
        self.critic_optimizer = Adam(learning_rate=lr_critic)
        self.critic.compile(loss="mean_squared_error", optimizer=self.critic_optimizer)
        update_target_weights(self.critic, self.critic_target, tau=1.)

        # Set hyperparameters
        self.sigma = sigma  # stddev for mean-zero gaussian noise
        self.gamma = gamma  # discount factor
        self.tau = tau  # target model update
        self.rho = rho  # chance of sampling from hmemory, paper recommends [0.05, 0.25]
        self.batch_size = batch_size

        # Training log
        self.max_reward = 0
        self.rewards = []
        self.summaries = {}

    def act(self, state, random_walk):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        a = self.actor.predict(state)
        # add random walk noise
        a += random_walk * self.action_bound
        a = tf.clip_by_value(a, -self.action_bound + self.action_shift, self.action_bound + self.action_shift)

        q_val = self.critic.predict([state, a])

        return a, q_val[0][0]

    def save_model(self, a_fn, c_fn):
        self.actor.save(a_fn)
        self.critic.save(c_fn)

    def load_actor(self, a_fn):
        self.actor.load_weights(a_fn)
        self.actor_target.load_weights(a_fn)
        print(self.actor.summary())

    def load_critic(self, c_fn):
        self.critic.load_weights(c_fn)
        self.critic_target.load_weights(c_fn)
        print(self.critic.summary())

    def store_memory(self, state, action, reward, next_state, done, hmemory=False):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        if hmemory:
            self.hmemory.append([state, action, reward, next_state, done])
        else:
            self.memory.append([state, action, reward, next_state, done])

    def sample_batch(self):
        samples = []
        for i in range(self.batch_size):
            if np.random.random() < self.rho and len(self.hmemory) > 0:
                sample = random.sample(self.hmemory, 1)  # sample from highly rewarded trajectories
            else:
                sample = random.sample(self.memory, 1)
            samples.append(sample[0])

        return samples

    def replay(self):
        samples = self.sample_batch()
        s = np.array(samples).T
        states, actions, rewards, next_states, dones = [np.vstack(s[i, :]).astype(np.float) for i in range(5)]
        next_actions = self.actor_target.predict(next_states)
        q_future = self.critic_target.predict([next_states, next_actions])
        target_qs = rewards + q_future * self.gamma * (1. - dones)

        # train critic
        hist = self.critic.fit([states, actions], target_qs, epochs=1, batch_size=self.batch_size, verbose=0)

        # train actor
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic([states, actions]))

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)  # compute actor gradient
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        self.summaries['critic_loss'] = np.mean(hist.history['loss'])
        self.summaries['actor_loss'] = actor_loss

    def async_collection(self, index, log_dir):
        summary_writer = tf.summary.create_file_writer(log_dir + '/env{}'.format(index))
        episode = total_steps = 0
        while True:
            done, cur_state, total_reward, rand_walk_noise, step = False, self.env[index].reset(), 0, 0, 0
            while not done:
                a, q_val = self.act(cur_state, rand_walk_noise)  # model determine action given state
                action = np.argmax(a) if self.discrete else a[0]  # post process for discrete action space
                next_state, reward, done, _ = self.env[index].step(action)  # perform action on env

                self.cache_buffer[index].append([cur_state, a, reward, next_state, done])  # add to buffer
                self.store_memory(cur_state, a, reward, next_state, done)  # add to memory

                cur_state = next_state
                total_reward += reward
                rand_walk_noise += tf.random.normal(shape=a.shape, mean=0., stddev=self.sigma, dtype=tf.float32)
                with summary_writer.as_default():
                    tf.summary.scalar('Stats/q_val', q_val, step=total_steps)
                    tf.summary.scalar('Stats/action', action, step=total_steps)
                summary_writer.flush()

                step += 1
                total_steps += 1

            with summary_writer.as_default():
                tf.summary.scalar('Main/episode_reward', total_reward, step=episode)
                tf.summary.scalar('Main/episode_steps', step, step=episode)
            summary_writer.flush()

            if total_reward >= self.max_reward:
                self.max_reward = total_reward
                while len(self.cache_buffer[index]) > 0:
                    transitions = self.cache_buffer[index].pop()
                    self.store_memory(*transitions, hmemory=True)

            episode += 1
            self.rewards.append(total_reward)

    def train(self, max_epochs=8000, save_freq=20):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        collection_threads, epoch = [], 0
        self.actor.make_predict_function()  # from https://github.com/jaromiru/AI-blog/issues/2
        print("-- starting threads --")
        for i in range(self.num_envs):
            t = threading.Thread(target=self.async_collection, args=(i, train_log_dir), daemon=True)
            t.start()
            collection_threads.append(t)

        print("-- waiting for first batch of data --")
        while is_threads_alive(collection_threads) and epoch < max_epochs:
            start = time.time()
            if len(self.memory) < self.batch_size:
                continue

            self.replay()  # train models through memory replay
            update_target_weights(self.actor, self.actor_target, tau=self.tau)  # iterates target model
            update_target_weights(self.critic, self.critic_target, tau=self.tau)

            with summary_writer.as_default():
                if len(self.memory) > self.batch_size:
                    tf.summary.scalar('Loss/actor_loss', self.summaries['actor_loss'], step=epoch)
                    tf.summary.scalar('Loss/critic_loss', self.summaries['critic_loss'], step=epoch)

            summary_writer.flush()

            epoch += 1
            if epoch % save_freq == 0:
                self.save_model("aeddpg_actor_epoch{}.h5".format(epoch),
                                "aeddpg_critic_epoch{}.h5".format(epoch))

            dt = time.time() - start
            print("train epoch {}: {} episodes, {} mean reward, {} max reward, {} seconds".format(
                epoch, len(self.rewards), np.mean(self.rewards[-100:]), self.max_reward, dt))

        print("-- saving final model --")
        self.save_model("aeddpg_actor_final_epoch{}.h5".format(max_epochs),
                        "aeddpg_critic_final_epoch{}.h5".format(max_epochs))

    def test(self, render=True, fps=30, filename='test_render.mp4'):
        cur_state, done, rewards = self.env[0].reset(), False, 0
        video = imageio.get_writer(filename, fps=fps)
        while not done:
            a, _ = self.act(cur_state, 0)
            action = np.argmax(a) if self.discrete else a[0]  # post process for discrete action space
            next_state, reward, done, _ = self.env[0].step(action)
            cur_state = next_state
            rewards += reward
            if render:
                video.append_data(self.env[0].render(mode='rgb_array'))
        video.close()
        return rewards


if __name__ == "__main__":
    name = "CartPole-v1"
    gym_env = gym.make(name)
    try:
        # Ensure action bound is symmetric
        assert (gym_env.action_space.high == -gym_env.action_space.low)
        is_discrete = False
        print('Continuous Action Space')
    except AttributeError:
        is_discrete = True
        print('Discrete Action Space')

    ddpg = AE_DDPG(name, discrete=is_discrete)
    ddpg.load_critic("basic_models/aeddpg_critic_epoch2300.h5")
    ddpg.load_actor("basic_models/aeddpg_actor_epoch2300.h5")
    # ddpg.train(max_epochs=2500)
    rewards = ddpg.test()
    print("Total rewards: ", rewards)
