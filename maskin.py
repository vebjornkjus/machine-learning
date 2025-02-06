import copy
import random
from collections import deque, defaultdict
import concurrent.futures

# -------------------------------
# Miljø og hjelpesfunksjoner
# -------------------------------

def get_connected_shapes(board, x, y):
    shape = board[y][x]
    to_visit = deque([(x, y)])
    connected = set()
    while to_visit:
        cx, cy = to_visit.pop()
        if (cx, cy) not in connected and board[cy][cx] == shape:
            connected.add((cx, cy))
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < len(board[0]) and 0 <= ny < len(board):
                    to_visit.append((nx, ny))
    return connected

def remove_shapes(board, connected):
    for x, y in connected:
        board[y][x] = 0

def apply_gravity(board):
    for col in range(len(board[0])):
        empty_row = len(board) - 1
        for row in range(len(board) - 1, -1, -1):
            if board[row][col] != 0:
                board[empty_row][col] = board[row][col]
                if empty_row != row:
                    board[row][col] = 0
                empty_row -= 1

def is_solved(board):
    return all(cell == 0 for row in board for cell in row)

# Miljøet for brettspillet
class BoardEnv:
    def __init__(self, board):
        self.initial_board = copy.deepcopy(board)
        self.board = copy.deepcopy(board)
        self.height = len(board)
        self.width = len(board[0])
        self.moves_taken = 0  # teller antall trekk brukt så langt
    
    def reset(self):
        self.board = copy.deepcopy(self.initial_board)
        self.moves_taken = 0  # nullstill trekk-teller
        return self.get_state()
    
    def get_state(self):
        return tuple(tuple(row) for row in self.board)
    
    def legal_actions(self):
        actions = []
        seen = set()
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] != 0 and (x, y) not in seen:
                    connected = get_connected_shapes(self.board, x, y)
                    seen.update(connected)
                    actions.append((x, y))
        return actions
    
    def step(self, action):
        # Øk trekk-teller
        self.moves_taken += 1

        x, y = action
        if self.board[y][x] == 0:
            # Meningsløst klikk: stor straff
            return self.get_state(), -5, False
        
        connected = get_connected_shapes(self.board, x, y)
        size_removed = len(connected)

        remove_shapes(self.board, connected)
        apply_gravity(self.board)

        # Belønning for å fjerne større klynger
        reward = size_removed * 0.5
        
        # Liten straff for hvert trekk
        reward -= 3

        # Sjekk om brettet er løst ELLER bare singletons igjen
        done = is_solved(self.board) or (size_removed > 1 and len(self.legal_actions()) == 0)
        
        if done:
            # Gi en grunnbelønning
            base_bonus = 50
            
            # Trekk-teller: Jo færre trekk, jo høyere bonus
            # Eks.: bonus = base_bonus + (50 - moves_taken)
            # Slik at bonusen avtar jo flere trekk man bruker
            bonus_for_few_moves = max(0, 50 - self.moves_taken)
            
            reward += base_bonus + bonus_for_few_moves
            
            # Alternativ: Bruk en brattere avtagende funksjon
            # f.eks. reward += base_bonus * (0.95 ** (self.moves_taken - 1))

        return self.get_state(), reward, done
    
    def render(self):
        for row in self.board:
            print(" ".join(map(str, row)))
        print()

# -------------------------------
# Q-Learning Agent
# -------------------------------

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.Q = defaultdict(float)  # nøkkel: (state, action)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
     
    def get_Q(self, state, action):
        return self.Q[(state, action)]
    
    def choose_action(self, state, legal_actions):
        # Epsilon-greedy strategi
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        else:
            q_values = [self.get_Q(state, a) for a in legal_actions]
            max_q = max(q_values) if q_values else 0
            best_actions = [a for a, q in zip(legal_actions, q_values) if q == max_q]
            return random.choice(best_actions) if best_actions else random.choice(legal_actions)
    
    def learn(self, state, action, reward, next_state, next_legal_actions, done):
        current_q = self.get_Q(state, action)
        if done:
            target = reward
        else:
            next_q = max([self.get_Q(next_state, a) for a in next_legal_actions], default=0)
            target = reward + self.gamma * next_q
        self.Q[(state, action)] = current_q + self.alpha * (target - current_q)

def convert_coord(action, board_height):
    x, y = action
    return (x + 1, board_height - y)

# -------------------------------
# Simulering av én episode (kjøres i en separat prosess)
# -------------------------------

def simulate_episode(agent_params, board, max_steps):
    env = BoardEnv(board)
    agent = QLearningAgent(alpha=agent_params["alpha"],
                           gamma=agent_params["gamma"],
                           epsilon=agent_params["epsilon"])
    
    experiences = []
    state = env.reset()
    for _ in range(max_steps):
        legal_actions = env.legal_actions()
        if not legal_actions:
            break
        action = agent.choose_action(state, legal_actions)
        next_state, reward, done = env.step(action)
        # Få neste lovlige handlinger med en gang (unngå reconstruct senere)
        next_legal_actions = env.legal_actions() if not done else []
        
        experiences.append((state, action, reward, next_state, next_legal_actions, done))
        state = next_state
        
        if done:
            break
    return experiences

# -------------------------------
# Parallel trening
# -------------------------------

def parallel_train(agent, board, total_episodes=80000, max_steps=50, workers=4):
    """
    Eksempel: Kjør parallel trening i flere batcher, og reduser epsilon mellom hver batch.
    """
    batch_size = 1000  # antall episoder per batch
    num_batches = total_episodes // batch_size
    
    for b in range(num_batches):
        # Lag en kopi av agent-parametere som brukes i parallell-simuleringer
        agent_params = {
            "alpha": agent.alpha,
            "gamma": agent.gamma,
            "epsilon": agent.epsilon  # Startverdi for batchen
        }
        
        all_experiences = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(simulate_episode, agent_params, board, max_steps)
                for _ in range(batch_size)
            ]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                experiences = future.result()
                all_experiences.extend(experiences)

        # Oppdater Q-tabell med alle erfaringene fra denne batchen
        for state, action, reward, next_state, next_legal_actions, done in all_experiences:
            agent.learn(state, action, reward, next_state, next_legal_actions, done)

        # Etter at batchen er ferdig, decay epsilon
        old_epsilon = agent.epsilon
        agent.epsilon = max(0.01, agent.epsilon * 0.97)  
        # Print ut at hver femte batch er fullført
        if b % 5 == 0:
            print(f"Batch {b+1}/{num_batches} fullført. Epsilon: {old_epsilon:.3f} -> {agent.epsilon:.3f}")

    for state, action, reward, next_state, next_legal_actions, done in all_experiences:
            agent.learn(state, action, reward, next_state, next_legal_actions, done)

# -------------------------------
# Eksempelbrett og hovedprogram
# -------------------------------

board = [
    [4, 1, 4, 2, 1, 4, 2],
    [2, 2, 4, 4, 2, 3, 3],
    [4, 1, 1, 4, 3, 3, 1],
    [1, 1, 4, 2, 1, 3, 4],
    [3, 1, 2, 4, 1, 1, 1],
    [3, 3, 1, 2, 4, 2, 4],
    [4, 3, 4, 2, 2, 1, 2],
    [2, 4, 4, 1, 2, 4, 1],
    [2, 1, 1, 1, 3, 4, 4],
]

def test_agent(env, agent, max_steps=50):
    state = env.reset()
    env.render()
    moves = []
    for _ in range(max_steps):
        legal_actions = env.legal_actions()
        if not legal_actions:
            break
        action = agent.choose_action(state, legal_actions)
        moves.append(action)
        printed_action = convert_coord(action, env.height)
        print("Trekk utført:", printed_action)
        state, reward, done = env.step(action)
        env.render()
        if done:
            print("Brettet er løst!")
            break
    print("Alle trekk:", [convert_coord(a, env.height) for a in moves])
    return moves

if __name__ == '__main__':
    # Opprett agenten
    agent = QLearningAgent(alpha=0.2, gamma=0.9, epsilon = 1.0)
    
    # Start parallel trening
    print("Starter parallel trening...")
    parallel_train(agent, board, total_episodes=80000, max_steps=50, workers=4)
    
    # Test agenten etter parallel trening
    print("\nTester trent agent:")
    env = BoardEnv(board)
    test_agent(env, agent, max_steps=50)