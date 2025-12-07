import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T19:40:17.210842
# Source Brief: brief_03197.md
# Brief Index: 3197
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
        "Navigate a dynamic maze by playing movement cards. Manage your momentum to reach the exit before it's depleted."
    )
    user_guide = (
        "Use the ← and → arrow keys to select a movement card. Press space to play the card and execute the move."
    )
    auto_advance = True
    
    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    CELL_SIZE = 40
    
    MAX_MOMENTUM = 100
    MAX_STEPS = 2000
    
    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 35, 64)
    COLOR_PLAYER = (0, 192, 255)
    COLOR_PLAYER_GLOW = (0, 192, 255, 50)
    COLOR_EXIT = (100, 255, 150)
    COLOR_EXIT_GLOW = (100, 255, 150, 60)
    COLOR_OBSTACLE = (255, 80, 80)
    COLOR_OBSTACLE_GLOW = (255, 80, 80, 70)
    COLOR_PATH_PREVIEW = (170, 100, 255, 200)
    COLOR_TRAIL = (0, 192, 255, 150)
    COLOR_MOMENTUM_BAR = (255, 150, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_CARD_BG = (40, 45, 80)
    COLOR_CARD_BG_SELECTED = (80, 90, 160)
    COLOR_CARD_TEXT = (200, 200, 220)

    # Game States
    STATE_IDLE = 0
    STATE_MOVING = 1
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont('Consolas', 14)
        self.font_medium = pygame.font.SysFont('Consolas', 18, bold=True)
        self.font_large = pygame.font.SysFont('Consolas', 24, bold=True)
        
        # Card definitions
        self._define_cards()
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0, 0])
        self.player_visual_pos = np.array([0.0, 0.0])
        self.exit_pos = np.array([0, 0])
        self.obstacles = []
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT))
        self.momentum = 0
        self.game_state = self.STATE_IDLE
        self.selected_card_index = 0
        self.hand = []
        self.unlocked_card_indices = []
        self.move_path = []
        self.move_progress = 0.0
        self.move_duration = 0
        self.particles = []
        self.total_steps_persistent = 0

    def _define_cards(self):
        self.all_cards = [
            {"name": "Step R", "path": [(1, 0)], "cost": 5},
            {"name": "Step L", "path": [(-1, 0)], "cost": 5},
            {"name": "Step D", "path": [(0, 1)], "cost": 5},
            {"name": "Step U", "path": [(0, -1)], "cost": 5},
            {"name": "Hop R", "path": [(2, 0)], "cost": 8},
            {"name": "Hop D", "path": [(0, 2)], "cost": 8},
            {"name": "Slide DR", "path": [(1, 1)], "cost": 7},
            {"name": "Slide UL", "path": [(-1, -1)], "cost": 7},
            {"name": "Knight Fwd", "path": [(1, -2)], "cost": 12},
            {"name": "Knight Bwd", "path": [(-1, 2)], "cost": 12},
            {"name": "Blink", "path": [(3, 0)], "cost": 15},
            {"name": "Phase", "path": [(0, 3)], "cost": 15},
        ]
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([1, self.GRID_HEIGHT // 2])
        self._generate_maze()
        
        self.player_visual_pos = self.player_pos.astype(float) * self.CELL_SIZE + self.CELL_SIZE / 2
        self.momentum = self.MAX_MOMENTUM
        self.game_state = self.STATE_IDLE
        self.selected_card_index = 0
        
        self.unlocked_card_indices = list(range(4)) # Start with 4 basic cards
        self.hand = []
        for _ in range(4):
            self._draw_card()
            
        self.move_path = []
        self.move_progress = 0.0
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _generate_maze(self):
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        # Place exit and ensure path
        while True:
            self.exit_pos = np.array([
                self.np_random.integers(self.GRID_WIDTH - 3, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            ])
            
            obstacle_density = 0.15 + 0.05 * (self.total_steps_persistent // 500)
            self.obstacles = []
            self.grid.fill(0)
            self.grid[self.exit_pos[0], self.exit_pos[1]] = 2 # Mark exit
            
            for _ in range(int(self.GRID_WIDTH * self.GRID_HEIGHT * obstacle_density)):
                pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
                if pos[0] == self.player_pos[0] and pos[1] == self.player_pos[1]: continue
                if pos[0] == self.exit_pos[0] and pos[1] == self.exit_pos[1]: continue
                self.grid[pos[0], pos[1]] = 1
                self.obstacles.append(np.array(pos))
            
            if self._is_solvable():
                break

    def _is_solvable(self):
        q = deque([tuple(self.player_pos)])
        visited = {tuple(self.player_pos)}
        while q:
            x, y = q.popleft()
            if x == self.exit_pos[0] and y == self.exit_pos[1]:
                return True
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[nx, ny] != 1 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self.total_steps_persistent += 1
        reward = 0
        
        prev_dist_to_exit = np.linalg.norm(self.player_pos - self.exit_pos)
        prev_momentum = self.momentum
        
        # --- Handle player action ---
        movement_action = action[0]
        play_card_action = action[1] == 1
        
        if self.game_state == self.STATE_IDLE:
            # Handle card selection
            if movement_action == 3: # Left
                self.selected_card_index = max(0, self.selected_card_index - 1)
            elif movement_action == 4: # Right
                self.selected_card_index = min(len(self.hand) - 1, self.selected_card_index + 1)
            
            # Handle card play
            if play_card_action and self.selected_card_index < len(self.hand):
                card = self.hand[self.selected_card_index]
                
                # Check if move is valid
                final_pos = self.player_pos + np.sum(card["path"], axis=0)
                if not (0 <= final_pos[0] < self.GRID_WIDTH and 0 <= final_pos[1] < self.GRID_HEIGHT):
                    # Invalid move, penalty
                    reward -= 1 
                else:
                    # Start movement
                    self.game_state = self.STATE_MOVING
                    self.move_path = [self.player_pos.copy()]
                    current_pos = self.player_pos.copy()
                    for segment in card["path"]:
                        current_pos = current_pos + segment
                        self.move_path.append(current_pos.copy())
                    
                    self.move_duration = 15 * len(card["path"]) # 15 frames per segment
                    self.move_progress = 0.0
                    self.momentum -= card["cost"]
                    
                    # Consume card and draw new one
                    self.hand.pop(self.selected_card_index)
                    self._draw_card()
                    if self.selected_card_index >= len(self.hand):
                        self.selected_card_index = len(self.hand) - 1
                    
                    # Sound placeholder
                    # play_sound("card_play")
                    self._create_particles(self.player_visual_pos, 20, self.COLOR_PATH_PREVIEW)
            else:
                 # Idle penalty
                 self.momentum -= 0.1
        
        # --- Update game state ---
        if self.game_state == self.STATE_MOVING:
            self.move_progress += 1.0 / self.move_duration
            
            if len(self.move_path) > 1:
                path_segment_index = math.floor(self.move_progress * (len(self.move_path) - 1))
                path_segment_index = min(path_segment_index, len(self.move_path) - 2)
                segment_progress = (self.move_progress * (len(self.move_path) - 1)) - path_segment_index
                
                start_node = self.move_path[path_segment_index]
                end_node = self.move_path[path_segment_index + 1]
                
                interp_pos = start_node + (end_node - start_node) * segment_progress
                self.player_visual_pos = interp_pos * self.CELL_SIZE + self.CELL_SIZE / 2
            
            if self.move_progress % 0.1 < 0.05: # Trail particles
                self._create_particles(self.player_visual_pos, 1, self.COLOR_TRAIL, life=15, size=3)

            if self.move_progress >= 1.0:
                self.player_pos = self.move_path[-1]
                self.player_visual_pos = self.player_pos.astype(float) * self.CELL_SIZE + self.CELL_SIZE / 2
                self.game_state = self.STATE_IDLE
                self.move_path = []
                
                # Check for collisions after move
                if self.grid[self.player_pos[0], self.player_pos[1]] == 1:
                    self.momentum = 0 # Instant failure on obstacle
                    # play_sound("collision")
                elif tuple(self.player_pos) == tuple(self.exit_pos):
                    self.game_over = True
                    self.score += 100
                    reward += 100
                    # play_sound("win")

        # Update particles
        self._update_particles()
        
        # Unlock new cards
        if self.steps > 0 and self.steps % 200 == 0:
            if len(self.unlocked_card_indices) < len(self.all_cards):
                self.unlocked_card_indices.append(len(self.unlocked_card_indices))
                reward += 5
                # play_sound("unlock")

        # --- Calculate reward ---
        new_dist_to_exit = np.linalg.norm(self.player_pos - self.exit_pos)
        reward += (prev_dist_to_exit - new_dist_to_exit) * 1.0
        
        momentum_loss = prev_momentum - self.momentum
        if momentum_loss > 0:
            reward -= momentum_loss * 0.1
            
        # --- Check termination ---
        if self.momentum <= 0:
            self.momentum = 0
            if not self.game_over: # Avoid double penalty
                reward -= 50
                self.score -= 50
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        terminated = self.game_over
        
        return (
            self._get_observation(),
            np.clip(reward, -100, 100),
            terminated,
            False,
            self._get_info()
        )

    def _draw_card(self):
        if len(self.hand) < 4:
            card_idx = self.np_random.choice(self.unlocked_card_indices)
            self.hand.append(self.all_cards[card_idx])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py))

        # Draw obstacles
        pulse = (math.sin(pygame.time.get_ticks() * 0.002) + 1) / 2 * 5
        for pos in self.obstacles:
            center_x, center_y = pos * self.CELL_SIZE + self.CELL_SIZE // 2
            size = self.CELL_SIZE * 0.6
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (center_x - size/2, center_y - size/2, size, size))
            pygame.gfxdraw.rectangle(self.screen, (int(center_x - size/2 - pulse), int(center_y - size/2 - pulse), int(size + pulse*2), int(size + pulse*2)), (*self.COLOR_OBSTACLE_GLOW[:3], int(100 - pulse * 10)))

        # Draw exit
        exit_pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2 * 8
        center_x, center_y = self.exit_pos * self.CELL_SIZE + self.CELL_SIZE // 2
        size = self.CELL_SIZE * 0.8
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (center_x - size/2, center_y - size/2, size, size), border_radius=4)
        pygame.gfxdraw.box(self.screen, (int(center_x - size/2 - exit_pulse), int(center_y - size/2 - exit_pulse), int(size + exit_pulse*2), int(size + exit_pulse*2)), (*self.COLOR_EXIT_GLOW[:3], int(80 - exit_pulse * 5)))

        # Draw path preview
        if self.game_state == self.STATE_IDLE and self.selected_card_index < len(self.hand):
            card = self.hand[self.selected_card_index]
            current_pos = self.player_visual_pos.copy()
            path_points = [current_pos]
            
            simulated_grid_pos = self.player_pos.copy()
            valid_path = True
            for segment in card['path']:
                simulated_grid_pos += segment
                if not (0 <= simulated_grid_pos[0] < self.GRID_WIDTH and 0 <= simulated_grid_pos[1] < self.GRID_HEIGHT):
                    valid_path = False
                    break
                path_points.append(simulated_grid_pos * self.CELL_SIZE + self.CELL_SIZE / 2)
            
            if valid_path and len(path_points) > 1:
                pygame.draw.aalines(self.screen, self.COLOR_PATH_PREVIEW, False, path_points, 2)

        # Draw particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            size = p['size'] * (p['life'] / p['max_life'])
            if size > 1:
                color = (*p['color'][:3], int(p['color'][3] * (p['life'] / p['max_life'])))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(size), color)
        
        # Draw player
        px, py = int(self.player_visual_pos[0]), int(self.player_visual_pos[1])
        player_glow = (math.sin(pygame.time.get_ticks() * 0.003) + 1) / 2 * 10 + 10
        pygame.gfxdraw.filled_circle(self.screen, px, py, int(self.CELL_SIZE * 0.3 + player_glow), self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, px, py, int(self.CELL_SIZE * 0.3), self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, px, py, int(self.CELL_SIZE * 0.3), self.COLOR_PLAYER)

    def _render_ui(self):
        # Momentum Bar
        bar_width = 200
        bar_height = 15
        momentum_ratio = max(0, self.momentum / self.MAX_MOMENTUM)
        pygame.draw.rect(self.screen, self.COLOR_CARD_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_MOMENTUM_BAR, (10, 10, bar_width * momentum_ratio, bar_height))
        momentum_text = self.font_small.render(f"MOMENTUM", True, self.COLOR_UI_TEXT)
        self.screen.blit(momentum_text, (15, 11))

        # Step Counter
        step_text = self.font_medium.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(step_text, (self.SCREEN_WIDTH - step_text.get_width() - 10, 10))
        
        # Card Hand
        card_w, card_h = 100, 50
        hand_y = self.SCREEN_HEIGHT - card_h - 10
        total_hand_width = len(self.hand) * (card_w + 10) - 10
        hand_x_start = (self.SCREEN_WIDTH - total_hand_width) / 2
        
        for i, card in enumerate(self.hand):
            card_x = hand_x_start + i * (card_w + 10)
            bg_color = self.COLOR_CARD_BG_SELECTED if i == self.selected_card_index and self.game_state == self.STATE_IDLE else self.COLOR_CARD_BG
            pygame.draw.rect(self.screen, bg_color, (card_x, hand_y, card_w, card_h), border_radius=5)
            
            name_text = self.font_small.render(card['name'], True, self.COLOR_CARD_TEXT)
            self.screen.blit(name_text, (card_x + (card_w - name_text.get_width())/2, hand_y + 10))
            
            cost_text = self.font_small.render(f"Cost: {card['cost']}", True, self.COLOR_MOMENTUM_BAR)
            self.screen.blit(cost_text, (card_x + (card_w - cost_text.get_width())/2, hand_y + 30))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            msg = "EXIT REACHED" if tuple(self.player_pos) == tuple(self.exit_pos) else "MOMENTUM DEPLETED"
            end_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            self.screen.blit(end_text, ((self.SCREEN_WIDTH - end_text.get_width())/2, self.SCREEN_HEIGHT/2 - 20))
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "momentum": self.momentum,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
        }

    def _create_particles(self, pos, count, color, life=30, size=5):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': self.np_random.integers(life // 2, life),
                'max_life': life,
                'color': color,
                'size': size
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Quantum Maze")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    action = [0, 0, 0] # no-op, no-space, no-shift
    
    while not done:
        # --- Human Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action[0] = 3 # Left
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4 # Right
                elif event.key == pygame.K_SPACE:
                    action[1] = 1 # Play card
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
                    
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # --- Reset action after one step to avoid sticky keys ---
        action = [0, 0, 0]
        
        # --- Render ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
    print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()