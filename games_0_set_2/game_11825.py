import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:39:42.104064
# Source Brief: brief_01825.md
# Brief Index: 1825
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Draw paths and place cells to cover nodes. Activate a magnetic field to create connections and complete the circuit before time runs out."
    )
    user_guide = (
        "Use arrow keys to move the cursor and draw paths. Press space to place a cell and shift to activate the magnetic field."
    )
    auto_advance = True

    # --- Constants ---
    GRID_SIZE = 40
    GRID_W = 16
    GRID_H = 10
    WIDTH, HEIGHT = GRID_W * GRID_SIZE, GRID_H * GRID_SIZE

    MAX_EPISODE_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_PATH = (0, 100, 120)
    COLOR_NODE_INACTIVE = (200, 50, 50)
    COLOR_NODE_ACTIVE = (50, 255, 50)
    COLOR_CELL = (255, 0, 255)
    COLOR_CONNECTION = (255, 255, 255)
    COLOR_MAGNETIC_FIELD = (0, 100, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SKILL = (0, 200, 200)

    # --- Rewards ---
    REWARD_STEP = -0.01
    REWARD_NEW_CONNECTION = 1.0
    REWARD_WIN = 100.0
    REWARD_LOSE = -100.0
    REWARD_SKILL_UNLOCK = 5.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # --- Persistent State (across episodes) ---
        self.all_skills = {
            "RANGE_1": {"type": "magnetic_range", "value": 1.0, "name": "Magnetic Range +1"},
            "RANGE_2": {"type": "magnetic_range", "value": 1.0, "name": "Magnetic Range +2"},
            "RANGE_3": {"type": "magnetic_range", "value": 1.5, "name": "Magnetic Range +3.5"},
        }
        self.unlocked_skills = []
        self.num_nodes_to_win = 3
        self.consecutive_wins = 0

        # --- Episode State ---
        self.cursor_pos = [0, 0]
        self.paths = set()
        self.cells = []
        self.nodes = []
        self.connections = set()
        self.magnetism_active = False
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Episode-Specific State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.paths = set()
        self.cells = []
        self.connections = set()
        self.magnetism_active = False
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False

        # --- Generate New Level ---
        self._generate_nodes()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = self.REWARD_STEP
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Actions (Press Detection) ---
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        # 1. Movement and Path Drawing
        if movement > 0:
            # Add current cursor position to path before moving
            self.paths.add(tuple(self.cursor_pos))
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_W - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_H - 1)
            self.paths.add(tuple(self.cursor_pos)) # Add new position as well

        # 2. Place Cell
        if space_pressed:
            cursor_tuple = tuple(self.cursor_pos)
            is_on_path = cursor_tuple in self.paths
            is_occupied = any(cell['pos'] == cursor_tuple for cell in self.cells)
            if is_on_path and not is_occupied:
                self.cells.append({'pos': cursor_tuple, 'id': len(self.cells)})
                # sfx: place_cell.wav
                self._create_particles(self.cursor_pos, self.COLOR_CELL, 15)

        # 3. Activate Magnetism
        if shift_pressed and not self.magnetism_active:
            self.magnetism_active = True
            # sfx: magnetism_activate.wav
            reward += self._update_connections()
            self._create_particles(self.cursor_pos, self.COLOR_MAGNETIC_FIELD, 30, is_burst=True)

        # --- Update Game State ---
        self._update_particles()

        # --- Check for Termination ---
        terminated = False
        truncated = False
        win = self._check_win_condition()
        if win:
            reward += self.REWARD_WIN
            terminated = True
            self.game_over = True
            self.consecutive_wins += 1
            # sfx: win_fanfare.wav
            
            # Handle progression
            if self.consecutive_wins % 5 == 0 and self.num_nodes_to_win < self.GRID_W * self.GRID_H / 4:
                self.num_nodes_to_win += 1
            
            if len(self.unlocked_skills) < len(self.all_skills):
                next_skill_key = list(self.all_skills.keys())[len(self.unlocked_skills)]
                self.unlocked_skills.append(next_skill_key)
                reward += self.REWARD_SKILL_UNLOCK

        elif self.steps >= self.MAX_EPISODE_STEPS:
            reward += self.REWARD_LOSE
            terminated = True # Per Gymnasium API, this is termination, not truncation
            self.game_over = True
            self.consecutive_wins = 0 # Reset win streak on loss
            # sfx: lose_buzzer.wav

        self.score += reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _generate_nodes(self):
        self.nodes = []
        occupied_coords = set()
        for _ in range(self.num_nodes_to_win):
            pos = None
            for _ in range(100): # Max attempts to find a free spot
                candidate_pos = (self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H))
                
                # Ensure minimum distance from other nodes
                too_close = False
                for existing_node in self.nodes:
                    dist = math.hypot(candidate_pos[0] - existing_node[0], candidate_pos[1] - existing_node[1])
                    if dist < 3:
                        too_close = True
                        break
                
                if not too_close:
                    pos = candidate_pos
                    break

            if pos is None: # Fallback if no good spot found
                pos = (self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H))

            self.nodes.append(pos)
            occupied_coords.add(pos)

    def _get_magnetic_range(self):
        base_range = 2.5
        for skill_key in self.unlocked_skills:
            skill = self.all_skills[skill_key]
            if skill["type"] == "magnetic_range":
                base_range += skill["value"]
        return base_range

    def _update_connections(self):
        if not self.magnetism_active:
            return 0

        new_connections = set()
        magnetic_range = self._get_magnetic_range()

        for i in range(len(self.cells)):
            for j in range(i + 1, len(self.cells)):
                cell1 = self.cells[i]
                cell2 = self.cells[j]
                dist = math.hypot(cell1['pos'][0] - cell2['pos'][0], cell1['pos'][1] - cell2['pos'][1])

                if dist <= magnetic_range:
                    # Check for a valid path between cells using BFS
                    q = deque([cell1['pos']])
                    visited = {cell1['pos']}
                    path_found = False
                    while q:
                        curr = q.popleft()
                        if curr == cell2['pos']:
                            path_found = True
                            break
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            neighbor = (curr[0] + dx, curr[1] + dy)
                            if neighbor in self.paths and neighbor not in visited:
                                visited.add(neighbor)
                                q.append(neighbor)
                    
                    if path_found:
                        new_connections.add(tuple(sorted((cell1['id'], cell2['id']))))

        new_connection_count = len(new_connections - self.connections)
        if new_connection_count > 0:
            # sfx: connection_zap.wav
            self.connections = new_connections
            # Create sparks on new connections
            for c1_id, c2_id in new_connections:
                p1 = self.cells[c1_id]['pos']
                p2 = self.cells[c2_id]['pos']
                self._create_particles(((p1[0]+p2[0])/2, (p1[1]+p2[1])/2), self.COLOR_CONNECTION, 5)

        return new_connection_count * self.REWARD_NEW_CONNECTION

    def _check_win_condition(self):
        if not self.magnetism_active or not self.nodes:
            return False

        # 1. Check if all nodes are covered by a cell
        node_to_cell_id = {}
        for node_pos in self.nodes:
            found_cell = False
            for cell in self.cells:
                if cell['pos'] == node_pos:
                    node_to_cell_id[node_pos] = cell['id']
                    found_cell = True
                    break
            if not found_cell:
                return False # An uncovered node exists

        # 2. Check if all node-covering cells are in the same connected component
        if not node_to_cell_id: return False
        
        adj = {i: [] for i in range(len(self.cells))}
        for c1_id, c2_id in self.connections:
            adj[c1_id].append(c2_id)
            adj[c2_id].append(c1_id)

        start_cell_id = list(node_to_cell_id.values())[0]
        q = deque([start_cell_id])
        visited_component = {start_cell_id}
        while q:
            curr_id = q.popleft()
            for neighbor_id in adj[curr_id]:
                if neighbor_id not in visited_component:
                    visited_component.add(neighbor_id)
                    q.append(neighbor_id)

        all_node_cells = set(node_to_cell_id.values())
        return all_node_cells.issubset(visited_component)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "unlocked_skills": len(self.unlocked_skills)}

    def _render_game(self):
        self._render_grid()
        self._render_paths()
        self._render_connections()
        self._render_nodes()
        self._render_cells()
        self._render_particles()
        self._render_cursor()

    def _render_grid(self):
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_paths(self):
        for x, y in self.paths:
            rect = pygame.Rect(x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_PATH, rect, border_radius=4)

    def _render_nodes(self):
        active_nodes = self._get_active_nodes()
        for i, (x, y) in enumerate(self.nodes):
            is_active = (x, y) in active_nodes
            color = self.COLOR_NODE_ACTIVE if is_active else self.COLOR_NODE_INACTIVE
            center_px = (int(x * self.GRID_SIZE + self.GRID_SIZE / 2), int(y * self.GRID_SIZE + self.GRID_SIZE / 2))
            radius = self.GRID_SIZE // 3
            
            if is_active:
                # Glow effect
                glow_radius = int(radius * (1.5 + 0.2 * math.sin(self.steps * 0.1 + i)))
                pygame.gfxdraw.filled_circle(self.screen, center_px[0], center_px[1], glow_radius, (*color, 50))
                pygame.gfxdraw.filled_circle(self.screen, center_px[0], center_px[1], int(glow_radius * 0.7), (*color, 80))

            pygame.gfxdraw.filled_circle(self.screen, center_px[0], center_px[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, center_px[0], center_px[1], radius, color)

    def _get_active_nodes(self):
        if not self._check_win_condition():
            return set()
        return set(self.nodes)

    def _render_cells(self):
        magnetic_range_px = self._get_magnetic_range() * self.GRID_SIZE
        for cell in self.cells:
            x, y = cell['pos']
            center_px = (int(x * self.GRID_SIZE + self.GRID_SIZE / 2), int(y * self.GRID_SIZE + self.GRID_SIZE / 2))
            
            # Magnetic field visualization
            if self.magnetism_active:
                pygame.gfxdraw.filled_circle(self.screen, center_px[0], center_px[1], int(magnetic_range_px), (*self.COLOR_MAGNETIC_FIELD, 30))
                pygame.gfxdraw.aacircle(self.screen, center_px[0], center_px[1], int(magnetic_range_px), (*self.COLOR_MAGNETIC_FIELD, 100))

            # Cell body
            size = self.GRID_SIZE // 2
            rect = pygame.Rect(center_px[0] - size/2, center_px[1] - size/2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_CELL, rect, border_radius=3)

    def _render_connections(self):
        if not self.magnetism_active:
            return
        
        for c1_id, c2_id in self.connections:
            pos1 = self.cells[c1_id]['pos']
            pos2 = self.cells[c2_id]['pos']
            p1_px = (int(pos1[0] * self.GRID_SIZE + self.GRID_SIZE / 2), int(pos1[1] * self.GRID_SIZE + self.GRID_SIZE / 2))
            p2_px = (int(pos2[0] * self.GRID_SIZE + self.GRID_SIZE / 2), int(pos2[1] * self.GRID_SIZE + self.GRID_SIZE / 2))
            
            # Pulsating width
            width = int(2 + math.sin(self.steps * 0.2 + c1_id))
            pygame.draw.line(self.screen, self.COLOR_CONNECTION, p1_px, p2_px, width)

    def _render_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2, border_radius=4)

    def _render_ui(self):
        # Score and Steps
        score_text = self.font_large.render(f"SCORE: {self.score:.2f}", True, self.COLOR_TEXT)
        steps_text = self.font_large.render(f"STEPS: {self.steps}/{self.MAX_EPISODE_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 5))

        # Unlocked Skills
        y_offset = 35
        for i, skill_key in enumerate(self.unlocked_skills):
            skill_name = self.all_skills[skill_key]['name']
            skill_text = self.font_small.render(f"✓ {skill_name}", True, self.COLOR_SKILL)
            self.screen.blit(skill_text, (10, y_offset + i * 20))

        # Game Over Message
        if self.game_over:
            win = self._check_win_condition()
            msg = "CIRCUIT COMPLETE" if win else "TIME LIMIT EXCEEDED"
            color = self.COLOR_NODE_ACTIVE if win else self.COLOR_NODE_INACTIVE
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            pygame.draw.rect(self.screen, (*self.COLOR_BG, 200), text_rect.inflate(20, 20))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, grid_pos, color, count, is_burst=False):
        px_pos = (grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE/2, grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE/2)
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4) if is_burst else random.uniform(0.5, 2)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(15, 30)
            self.particles.append({'pos': list(px_pos), 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            radius = int(3 * (p['life'] / p['max_life']))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Set the video driver to a real one for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv()
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # --- Manual Control Mapping ---
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Use a separate screen for rendering if playing manually
    manual_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Magnetic Cell Circuit")
    
    running = True
    while running:
        # --- Default Action ---
        movement = 0
        space_held = 0
        shift_held = 0

        # --- Pygame Event Loop ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        for key, move_action in key_to_action.items():
            if keys[key]:
                movement = move_action
                break # Prioritize first key found
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
        if keys[pygame.K_r]: # Press R to reset
            obs, info = env.reset()
            total_reward = 0
            print("--- Environment Reset ---")

        action = [movement, space_held, shift_held]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render to Display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        manual_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode Finished! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            print("--- New Episode Started ---")

        env.clock.tick(30) # Limit to 30 FPS

    env.close()