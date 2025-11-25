import numpy as np
from blackjack_env import BlackjackEnv, Action
from q_learning import QLearningAgent


def test_environment():
    print("=" * 50)
    print("Testando o ambiente de Blackjack")
    print("=" * 50)
    
    env = BlackjackEnv()
    
    for episode in range(5):
        state = env.reset()
        print(f"\nEpisódio {episode + 1}")
        print(f"Estado inicial: {state} (valor da mão: {env.get_state_value(state)})")
        print(f"Estado: {env.render()}")
        
        steps = 0
        while not env.done:
            action = Action.HIT if env.player_hand < 17 else Action.STAND
            next_state, reward, done, info = env.step(action)
            
            print(f"  Passo {steps + 1}: Ação={action.name}, Próximo estado={next_state}, "
                  f"Recompensa={reward:.1f}, Done={done}")
            print(f"  {env.render()}")
            
            if done:
                print(f"  Resultado: {info.get('result', 'N/A')}")
            
            steps += 1
    
    print("\n" + "=" * 50)


def test_q_learning():
    print("\n" + "=" * 50)
    print("Testando Q-Learning (1000 episódios)")
    print("=" * 50)
    
    env = BlackjackEnv()
    agent = QLearningAgent(
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    num_episodes = 1000
    print(f"\nTreinando por {num_episodes} episódios...\n")
    
    for episode in range(num_episodes):
        agent.train_episode(env)
        
        if (episode + 1) % 200 == 0:
            stats = agent.get_stats()
            print(f"Episódio {episode + 1}/{num_episodes} | "
                  f"Recompensa média: {stats['avg_reward_recent']:.3f} | "
                  f"Taxa de vitória: {stats['win_rate']*100:.1f}% | "
                  f"Epsilon: {stats['epsilon']:.3f}")
    
    print("\n" + "=" * 50)
    print("Matriz Q final:")
    print("=" * 50)
    print("\nEstado | Valor Mão | Q(HIT)    | Q(STAND)  | Melhor Ação")
    print("-" * 55)
    
    for state in range(agent.num_states):
        hand_value = env.get_state_value(state)
        hit_q = agent.Q[state, Action.HIT.value]
        stand_q = agent.Q[state, Action.STAND.value]
        best_action_idx = np.argmax(agent.Q[state, :])
        best_action = "HIT" if best_action_idx == Action.HIT.value else "STAND"
        
        print(f"  {state:2d}   |    {hand_value:2d}     | {hit_q:8.3f} | {stand_q:8.3f} | {best_action}")
    
    print("\n" + "=" * 50)
    print("Estatísticas finais:")
    print("=" * 50)
    stats = agent.get_stats()
    print(f"Total de episódios: {stats['total_episodes']}")
    print(f"Recompensa média (últimos 1000): {stats['avg_reward_recent']:.3f}")
    print(f"Taxa de vitória: {stats['win_rate']*100:.1f}%")
    print(f"Epsilon final: {stats['epsilon']:.3f}")
    print("\n")


if __name__ == "__main__":
    test_environment()
    
    test_q_learning()

