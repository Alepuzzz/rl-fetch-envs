import random
from ddpg import ddpg
from replay_buffer import HerReplayBuffer


def her(env, policy, steps_per_epoch=4000, epochs=200, replay_size=int(1e6), batch_size=128, prob_rand_action=0.2, 
        update_every=16, optimization_steps=40, act_noise=0.2, max_ep_len=50, save_freq=1, num_test_episodes=10):
    """
    steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
        for the agent and the environment in each epoch.

    epochs (int): Number of epochs to run and train agent.

    replay_size (int): Maximum length of replay buffer.

    batch_size (int): Minibatch size for SGD.

    prob_rand_action (int): Probability of sample a random action instead of
        the policy action. Helps exploration.

    update_every (int): Number of episodes that should elapse between gradient descent updates.

    optimization_steps (int): Number of optimization steps.

    act_noise (float): Stddev for Gaussian exploration noise added to 
        policy at training time. (At test time, no noise is added.)

    max_ep_len (int): Maximum length of trajectory / episode / rollout.

    save_freq (int): How often (in terms of gap between epochs) to save
        the current policy and value function.
    
    num_test_episodes (int): Number of episodes to test the deterministic
    policy at the end of each epoch.
    """

    obs_dim = env.observation_space['observation'].shape[0]
    act_dim = env.action_space.shape[0]
    goal_dim = env.observation_space['desired_goal'].shape[0]

    # Experience buffer
    replay_buffer = HerReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, goal_dim=goal_dim , size=replay_size)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len, episode_count = env.reset(), 0, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        if random.uniform(0,1) > prob_rand_action:
            a = policy.get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, info = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d, info)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            policy.logger['EpRet'].append(ep_ret)
            policy.logger['EpLen'].append(ep_len)
            policy.logger['SuccessRate'].append(int(d))
            o, ep_ret, ep_len, episode_count = env.reset(), 0, 0, episode_count+1
            replay_buffer.sample_additional_goals(env)

        # Update handling
        if episode_count % update_every == 0:
            for _ in range(optimization_steps):
                batch = replay_buffer.sample_batch(batch_size)
                policy.update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                policy.save_model('models.pt')

            # Test the performance of the deterministic version of the agent.
            policy.test_agent(num_test_episodes)

            policy.print_epoch_data(epoch, t, episode_count)
