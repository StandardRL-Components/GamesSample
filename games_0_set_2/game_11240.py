import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:40:27.601887
# Source Brief: brief_01240.md
# Brief Index: 1240
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Network Sync'.

    In this puzzle game, the player's goal is to synchronize a network of nodes
    to a single target frequency (color). The player selects nodes and cycles
    their frequency. Changing a node's frequency can trigger a chain reaction
    that synchronizes other nodes of the same color. The network degrades over
    time, making synchronization harder and eventually leading to failure if
    not managed.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]` (Movement): Selects a node. 0: none, 1: up, 2: down, 3: left, 4: right.
    - `action[1]` (Space): Cycles the selected node's frequency forward.
    - `action[2]` (Shift): Cycles the selected node's frequency backward.

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - +1 for each node synchronized in a chain reaction.
    - +100 for winning (all nodes synchronized).
    - -100 for losing (network degradation at 100%).
    - -0.02 per step to encourage efficiency.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Synchronize a network of nodes to a single color. Select nodes and change their color to trigger chain reactions, but act quickly before the network degrades."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to select a node. Press space to cycle its color forward and shift to cycle it backward."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000
    DEGRADATION_PER_STEP = 1.0 / (5 * FPS * 10) # 100% degradation in 50 seconds

    # --- Colors ---
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (20, 30, 50)
    COLORS = [
        (255, 80, 80),   # 0: Red
        (80, 255, 80),   # 1: Green
        (80, 150, 255),  # 2: Blue
        (255, 255, 80),  # 3: Yellow
        (80, 255, 255),  # 4: Cyan
        (255, 80, 255),  # 5: Magenta
    ]
    TARGET_COLOR = (255, 255, 255) # 6: White
    DEGRADED_COLOR = (90, 90, 90)
    UI_COLOR = (220, 220, 240)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 32)

        # Game state variables
        self.network_size = 5
        self.nodes = []
        self.connections = []
        self.selected_node_index = 0
        self.global_degradation = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.space_was_held = False
        self.shift_was_held = False
        self.particles = []
        self.active_connections = [] # For glowing effect

        # self.reset() is called by the environment wrapper
        # self.validate_implementation() is for debugging, not needed in final version

    def _generate_network(self):
        """Generates a new network layout with nodes and connections."""
        self.nodes = []
        node_radius = 20
        # 1. Create nodes with random positions
        for i in range(self.network_size):
            self.nodes.append({
                'pos': np.array([
                    self.np_random.uniform(node_radius * 2, self.WIDTH - node_radius * 2),
                    self.np_random.uniform(node_radius * 2, self.HEIGHT - node_radius * 2)
                ]),
                'color_index': self.np_random.integers(0, len(self.COLORS)),
                'pulse': self.np_random.uniform(0, math.pi * 2),
            })

        # 2. Iteratively push nodes apart to avoid overlap
        for _ in range(50):
            for i in range(self.network_size):
                for j in range(i + 1, self.network_size):
                    p1, p2 = self.nodes[i]['pos'], self.nodes[j]['pos']
                    vec = p1 - p2
                    dist = np.linalg.norm(vec)
                    if 0 < dist < node_radius * 4:
                        repel_force = (vec / dist) * (1 / (dist + 1e-6)) * 5
                        self.nodes[i]['pos'] = np.clip(p1 + repel_force, node_radius, [self.WIDTH-node_radius, self.HEIGHT-node_radius])
                        self.nodes[j]['pos'] = np.clip(p2 - repel_force, node_radius, [self.WIDTH-node_radius, self.HEIGHT-node_radius])

        # 3. Create connections to 2-3 nearest neighbors
        self.connections = [[] for _ in range(self.network_size)]
        for i in range(self.network_size):
            distances = [(np.linalg.norm(self.nodes[i]['pos'] - self.nodes[j]['pos']), j) for j in range(self.network_size) if i != j]
            distances.sort()
            num_connections = self.np_random.integers(2, 4)
            for _, j in distances[:num_connections]:
                if j not in self.connections[i] and i not in self.connections[j]:
                    self.connections[i].append(j)
                    self.connections[j].append(i)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if 'network_size' in (options or {}):
            self.network_size = options['network_size']

        self._generate_network()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.global_degradation = 0.0
        self.selected_node_index = 0
        self.space_was_held = False
        self.shift_was_held = False
        self.particles = []
        self.active_connections = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.02  # Small penalty per step
        self.active_connections.clear()

        # --- 1. Unpack and process actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Handle node selection via movement
        if movement > 0:
            self.selected_node_index = self._find_next_node(movement)

        # Handle color cycling
        space_pressed = space_held and not self.space_was_held
        shift_pressed = shift_held and not self.shift_was_held
        
        if space_pressed or shift_pressed:
            node = self.nodes[self.selected_node_index]
            num_colors = len(self.COLORS)
            
            if space_pressed: # Cycle forward
                node['color_index'] = (node['color_index'] + 1) % num_colors
            elif shift_pressed: # Cycle backward
                node['color_index'] = (node['color_index'] - 1 + num_colors) % num_colors
            
            # --- Trigger chain reaction and calculate reward ---
            # sfx: positive_action.wav
            chain_reward, activated_nodes = self._run_chain_reaction(self.selected_node_index)
            reward += chain_reward
            self.score += chain_reward
            # Create particles for visual feedback
            for node_idx in activated_nodes:
                self._create_particles(node_idx)

        self.space_was_held = space_held
        self.shift_was_held = shift_held

        # --- 2. Update game state ---
        self.global_degradation = min(1.0, self.global_degradation + self.DEGRADATION_PER_STEP)
        
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # --- 3. Check for termination conditions ---
        terminated = False
        is_win = all(n['color_index'] == self.nodes[0]['color_index'] for n in self.nodes)
        
        if is_win:
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
            # sfx: win_sound.wav
            # On win, increase difficulty for the next round
            self.network_size += 1

        if self.global_degradation >= 1.0:
            reward -= 100
            self.score -= 100
            terminated = True
            self.game_over = True
            # sfx: lose_sound.wav
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _run_chain_reaction(self, start_node_idx):
        """
        Propagates a signal from the start node. Nodes of the same color
        also activate, creating a chain. Returns the reward for this chain.
        """
        start_node = self.nodes[start_node_idx]
        target_color_idx = start_node['color_index']
        
        q = [start_node_idx]
        visited = {start_node_idx}
        activated_count = 0
        
        while q:
            current_idx = q.pop(0)
            
            # Degraded nodes have a chance to fail propagation
            node_degradation = self.global_degradation * self.np_random.uniform(0.8, 1.2)
            if self.np_random.random() < node_degradation:
                continue # sfx: fizzle.wav
            
            activated_count += 1
            # sfx: chain_propagate.wav
            
            for neighbor_idx in self.connections[current_idx]:
                if neighbor_idx not in visited and self.nodes[neighbor_idx]['color_index'] == target_color_idx:
                    visited.add(neighbor_idx)
                    q.append(neighbor_idx)
                    self.active_connections.append(tuple(sorted((current_idx, neighbor_idx))))

        return activated_count, visited

    def _find_next_node(self, direction):
        """Finds the best candidate node in a given direction."""
        current_pos = self.nodes[self.selected_node_index]['pos']
        best_candidate = -1
        min_score = float('inf')

        # Direction vectors: 1=up, 2=down, 3=left, 4=right
        dir_vectors = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        target_dir = np.array(dir_vectors[direction])

        for i, node in enumerate(self.nodes):
            if i == self.selected_node_index:
                continue
            
            vec_to_node = node['pos'] - current_pos
            dist = np.linalg.norm(vec_to_node)
            if dist < 1e-6: continue
            
            dir_to_node = vec_to_node / dist
            
            # Cosine similarity to check angle
            dot_product = np.dot(target_dir, dir_to_node)
            
            # We only want nodes in the correct general direction
            if dot_product > 0.1: # Allow a wide cone
                # Score prioritizes nodes aligned with the direction and penalizes distance
                angle_penalty = (1 - dot_product) * 500
                score = dist + angle_penalty
                if score < min_score:
                    min_score = score
                    best_candidate = i

        return best_candidate if best_candidate != -1 else self.selected_node_index

    def _create_particles(self, node_idx):
        """Create a burst of particles at a node's location."""
        pos = self.nodes[node_idx]['pos']
        color = self.COLORS[self.nodes[node_idx]['color_index']]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(10, 20),
                'color': color,
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw connections
        for i, neighbors in enumerate(self.connections):
            for j in neighbors:
                if i < j:
                    is_active = tuple(sorted((i, j))) in self.active_connections
                    color = (150, 200, 255) if is_active else (50, 70, 100)
                    width = 3 if is_active else 1
                    pygame.draw.aaline(self.screen, color, self.nodes[i]['pos'], self.nodes[j]['pos'], width)

        # Draw nodes
        for i, node in enumerate(self.nodes):
            pos = (int(node['pos'][0]), int(node['pos'][1]))
            
            # Calculate dynamic properties
            node['pulse'] = (node['pulse'] + 0.1) % (2 * math.pi)
            pulse_size = int(math.sin(node['pulse']) * 2)
            base_radius = 15
            radius = base_radius + pulse_size
            
            node_degradation = self.global_degradation * self.np_random.uniform(0.8, 1.2)
            node_color = self.COLORS[node['color_index']]
            
            # Blend color with gray based on degradation
            final_color = tuple(int(c * (1 - node_degradation) + gc * node_degradation) for c, gc in zip(node_color, self.DEGRADED_COLOR))
            
            # Flickering effect for high degradation
            if self.global_degradation > 0.5 and self.np_random.random() < self.global_degradation * 0.2:
                final_color = self.DEGRADED_COLOR

            # Draw glow
            glow_radius = int(radius * 1.8)
            glow_color = tuple(c * 0.4 for c in final_color)
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*glow_color, 80), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius))

            # Draw main node
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, final_color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, final_color)
        
        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = int(255 * (p['life'] / 20.0))
            s = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p['color'], alpha), (2,2), 2)
            self.screen.blit(s, (pos[0]-2, pos[1]-2))

        # Draw selection cursor
        if not self.game_over:
            sel_pos = self.nodes[self.selected_node_index]['pos']
            sel_pos_int = (int(sel_pos[0]), int(sel_pos[1]))
            cursor_radius = 25
            angle = (pygame.time.get_ticks() / 200) % (2 * math.pi)
            for i in range(3):
                a = angle + i * (2 * math.pi / 3)
                p1 = (sel_pos_int[0] + cursor_radius * math.cos(a), sel_pos_int[1] + cursor_radius * math.sin(a))
                p2 = (sel_pos_int[0] + cursor_radius * math.cos(a + 0.5), sel_pos_int[1] + cursor_radius * math.sin(a + 0.5))
                pygame.draw.line(self.screen, self.UI_COLOR, p1, p2, 2)

    def _render_ui(self):
        # Sync percentage
        if self.network_size > 0:
            unique_colors = len(set(n['color_index'] for n in self.nodes))
            sync_percent = 100 * (self.network_size - unique_colors + 1) / self.network_size if self.network_size > 1 else 100
        else:
            sync_percent = 100
        sync_text = self.font_ui.render(f"SYNC: {sync_percent:.0f}%", True, self.UI_COLOR)
        self.screen.blit(sync_text, (10, 10))
        
        # Degradation bar
        bar_width, bar_height = 150, 15
        bar_x, bar_y = self.WIDTH - bar_width - 10, 10
        pygame.draw.rect(self.screen, (50, 50, 80), (bar_x, bar_y, bar_width, bar_height))
        fill_width = bar_width * self.global_degradation
        pygame.draw.rect(self.screen, (255, 80, 80), (bar_x, bar_y, fill_width, bar_height))
        degr_text = self.font_ui.render("DEGRADATION", True, self.UI_COLOR)
        self.screen.blit(degr_text, (bar_x, bar_y + bar_height + 2))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.UI_COLOR)
        self.screen.blit(score_text, (10, self.HEIGHT - 30))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "degradation": self.global_degradation,
            "network_size": self.network_size,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    if os.environ.get("SDL_VIDEODRIVER") != "dummy":
        env = GameEnv()
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Network Sync")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            movement, space, shift = 0, 0, 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            # Only register movement if a key is newly pressed to avoid rapid selection
            # This part is simplified for manual play; the agent gets one action per step.
            # A better manual play would handle key-down events.
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
                
            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Draw the observation from the environment to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                # Wait a moment then reset
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0

            clock.tick(GameEnv.FPS)
            
        env.close()
    else:
        print("Running in headless mode. Skipping manual play.")
        env = GameEnv()
        env.reset()
        env.step(env.action_space.sample())
        env.validate_implementation()
        env.close()