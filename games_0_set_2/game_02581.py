import os
import os
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move. Push gems of the same color together to merge and score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic grid-based puzzle game. Clear the board by merging matching gems. Plan your moves carefully to avoid getting stuck!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Grid Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 5
        self.CELL_SIZE = 60
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (60, 60, 70)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_BONUS = (255, 223, 0)
        self.GEM_COLORS = [
            (255, 71, 87),    # Red
            (46, 213, 115),   # Green
            (30, 144, 255),   # Blue
            (255, 165, 2),    # Orange
        ]
        self.COLOR_TEXT = (240, 240, 240)

        # Game constants
        self.MAX_STEPS = 1000
        self.TOTAL_GEMS_INITIAL = 20
        self.NUM_GEM_TYPES = 4

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 18)
        self.font_game_over = pygame.font.SysFont("Arial", 48, bold=True)
        
        # Etc...        
        self.grid = None
        self.player_pos = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.gems_remaining = 0
        self.bonus_tiles = []
        self.particles = []
        self.last_action_info = ""
        
        # Initialize state variables
        # self.reset() is called by the wrapper or user
    
    def _cell_to_pixel(self, r, c):
        x = self.GRID_X_OFFSET + c * self.CELL_SIZE
        y = self.GRID_Y_OFFSET + r * self.CELL_SIZE
        return x, y

    def _get_empty_cells(self):
        empty = []
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == 0:
                    empty.append((r, c))
        return empty

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.bonus_tiles = []
        self.particles = []
        self.last_action_info = "Game Start!"

        # Regenerate board until it's not an immediate loss state
        while True:
            self._setup_board()
            if not self._is_stuck():
                break
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _setup_board(self):
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        
        # Define entities: 1 player, 20 gems (5 of each of 4 types)
        entities = [1] # 1 is player
        for i in range(self.NUM_GEM_TYPES):
            entities.extend([i + 2] * (self.TOTAL_GEMS_INITIAL // self.NUM_GEM_TYPES)) # Gems are 2, 3, 4, 5
        
        self.gems_remaining = self.TOTAL_GEMS_INITIAL

        # Place entities randomly
        positions = [(r, c) for r in range(self.GRID_SIZE) for c in range(self.GRID_SIZE)]
        self.np_random.shuffle(positions)
        
        for i, entity_id in enumerate(entities):
            r, c = positions[i]
            self.grid[r, c] = entity_id
            if entity_id == 1:
                self.player_pos = (r, c)

    def step(self, action):
        if self.game_over:
            # On subsequent steps after termination, return the last observation.
            # The reward is 0 and info is the final info.
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held and shift_held are not used in this game
        
        reward = -0.1  # Cost of taking a step
        self.last_action_info = ""
        
        moved = False
        if movement != 0: # 0 is no-op
            dr = [-1, 1, 0, 0] # Corresponds to up, down
            dc = [0, 0, -1, 1] # Corresponds to left, right
            
            move_map = {1: 0, 2: 1, 3: 2, 4: 3} # map action to dr/dc index
            direction_idx = move_map[movement]

            pr, pc = self.player_pos
            tr, tc = pr + dr[direction_idx], pc + dc[direction_idx] # Target cell

            if 0 <= tr < self.GRID_SIZE and 0 <= tc < self.GRID_SIZE:
                target_content = self.grid[tr, tc]

                if target_content == 0: # Move to empty cell
                    self.grid[pr, pc] = 0
                    self.grid[tr, tc] = 1
                    self.player_pos = (tr, tc)
                    moved = True
                    self.last_action_info = "Move"
                
                elif target_content >= 2: # Push a gem
                    br, bc = tr + dr[direction_idx], tc + dc[direction_idx] # Cell behind gem

                    if 0 <= br < self.GRID_SIZE and 0 <= bc < self.GRID_SIZE:
                        behind_content = self.grid[br, bc]
                        
                        if behind_content == 0: # Push gem into empty space
                            reward += self._calculate_centroid_reward(target_content, (tr, tc), (br, bc))
                            
                            self.grid[br, bc] = target_content
                            self.grid[tr, tc] = 1
                            self.grid[pr, pc] = 0
                            self.player_pos = (tr, tc)
                            moved = True
                            self.last_action_info = "Push"

                            if (br, bc) in self.bonus_tiles:
                                reward += 20
                                self.bonus_tiles.remove((br, bc))
                                self.last_action_info = "Bonus Push! +20"
                                # Bonus particle effect
                                for _ in range(30):
                                    self.particles.append(Particle(self, self._cell_to_pixel(br, bc), self.COLOR_BONUS))

                        elif behind_content == target_content: # Push gem into matching gem (merge)
                            self.grid[br, bc] = 0
                            self.grid[tr, tc] = 1
                            self.grid[pr, pc] = 0
                            self.player_pos = (tr, tc)
                            
                            reward += 10
                            self.score += 5 # Per-gem collection reward as per brief
                            self.gems_remaining -= 2
                            moved = True
                            self.last_action_info = "Merge! +10"

                            # Particle effect for merge
                            for _ in range(50):
                                self.particles.append(Particle(self, self._cell_to_pixel(br, bc), self.GEM_COLORS[target_content - 2]))

                            empty_cells = self._get_empty_cells()
                            if empty_cells:
                                bonus_pos_idx = self.np_random.integers(0, len(empty_cells))
                                self.bonus_tiles.append(empty_cells[bonus_pos_idx])
        
        if not moved and movement != 0:
            reward -= 1 # Penalty for invalid move
            self.last_action_info = "Invalid Move"

        # Update game logic
        self.steps += 1
        self.score += reward # The score is the cumulative reward
        
        terminated = self._check_termination()
        if terminated and not self.game_over: # First frame of termination
            if self.gems_remaining <= 0:
                reward += 100
                self.score += 100
                self.last_action_info = "You Win! +100"
            else: # Got stuck or max steps
                reward -= 50
                self.score -= 50
                self.last_action_info = "Stuck! -50"
        
        self.game_over = terminated
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _calculate_centroid_reward(self, gem_type, gem_pos, new_gem_pos):
        gem_locations = []
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == gem_type and (r, c) != gem_pos:
                    gem_locations.append((r, c))
        
        if not gem_locations:
            return 0
        
        centroid_r = sum(r for r, c in gem_locations) / len(gem_locations)
        centroid_c = sum(c for r, c in gem_locations) / len(gem_locations)
        
        old_dist = math.hypot(gem_pos[0] - centroid_r, gem_pos[1] - centroid_c)
        new_dist = math.hypot(new_gem_pos[0] - centroid_r, new_gem_pos[1] - centroid_c)
        
        if new_dist < old_dist:
            return 1.0
        else:
            return -0.2

    def _is_stuck(self):
        pr, pc = self.player_pos
        dr = [-1, 1, 0, 0]
        dc = [0, 0, -1, 1]

        for i in range(4):
            tr, tc = pr + dr[i], pc + dc[i]

            if not (0 <= tr < self.GRID_SIZE and 0 <= tc < self.GRID_SIZE):
                continue

            target_content = self.grid[tr, tc]
            if target_content == 0: return False

            if target_content >= 2:
                br, bc = tr + dr[i], tc + dc[i]
                if not (0 <= br < self.GRID_SIZE and 0 <= bc < self.GRID_SIZE):
                    continue
                
                behind_content = self.grid[br, bc]
                if behind_content == 0: return False
                if behind_content == target_content: return False
        
        return True

    def _check_termination(self):
        if self.gems_remaining <= 0: return True
        if self.steps >= self.MAX_STEPS: return True
        if self._is_stuck(): return True
        return False
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET), (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT), 2)
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + i * self.CELL_SIZE), (self.GRID_X_OFFSET + self.GRID_WIDTH, self.GRID_Y_OFFSET + i * self.CELL_SIZE), 2)
        
        for r, c in self.bonus_tiles:
            x, y = self._cell_to_pixel(r, c)
            alpha = 128 + 127 * math.sin(self.steps * 0.3)
            flash_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(flash_surface, (*self.COLOR_BONUS, alpha), (0,0,self.CELL_SIZE,self.CELL_SIZE))
            self.screen.blit(flash_surface, (x, y))

        padding, radius = 5, 8
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                entity = self.grid[r, c]
                if entity == 0: continue
                
                x, y = self._cell_to_pixel(r, c)
                rect = pygame.Rect(x + padding, y + padding, self.CELL_SIZE - 2 * padding, self.CELL_SIZE - 2 * padding)
                
                if entity == 1: # Player
                    glow_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                    for i in range(10, 0, -2):
                        pygame.gfxdraw.filled_circle(glow_surf, self.CELL_SIZE//2, self.CELL_SIZE//2, self.CELL_SIZE//2 - padding + i, (100, 150, 255, 10))
                    self.screen.blit(glow_surf, (x, y))
                    pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=radius)

                elif entity >= 2: # Gem
                    color = self.GEM_COLORS[entity - 2]
                    shadow_rect = rect.copy(); shadow_rect.move_ip(2, 2)
                    pygame.draw.rect(self.screen, (0,0,0,50), shadow_rect, border_radius=radius)
                    pygame.draw.rect(self.screen, color, rect, border_radius=radius)
                    highlight_rect = pygame.Rect(rect.x+2, rect.y+2, rect.width-4, rect.height/2-2)
                    pygame.draw.rect(self.screen, (255,255,255,60), highlight_rect, border_radius=radius-2)

        for p in self.particles[:]:
            p.update()
            p.draw(self.screen)
            if p.lifetime <= 0:
                self.particles.remove(p)

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        gems_text = self.font_small.render(f"Gems: {self.gems_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(gems_text, (15, 40))

        steps_text = self.font_large.render(f"Moves: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        text_rect = steps_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 10))
        self.screen.blit(steps_text, text_rect)

        action_text = self.font_small.render(self.last_action_info, True, self.COLOR_TEXT)
        action_rect = action_text.get_rect(midbottom=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 10))
        self.screen.blit(action_text, action_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "You Win!" if self.gems_remaining <= 0 else "Game Over"
            end_text = self.font_game_over.render(msg, True, self.COLOR_PLAYER)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_remaining": self.gems_remaining,
            "player_pos": self.player_pos,
        }
    
    def close(self):
        pygame.quit()

class Particle:
    def __init__(self, env, pos, color):
        self.env = env
        self.x, self.y = pos[0] + env.CELL_SIZE / 2, pos[1] + env.CELL_SIZE / 2
        self.color = color
        angle = env.np_random.uniform(0, 2 * math.pi)
        speed = env.np_random.uniform(2, 6)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifetime = env.np_random.integers(20, 40)
        self.radius = env.np_random.uniform(3, 7)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.95
        self.vy *= 0.95
        self.lifetime -= 1
        self.radius -= 0.1
    
    def draw(self, screen):
        if self.lifetime > 0 and self.radius > 0:
            alpha = max(0, min(255, int(255 * (self.lifetime / 30))))
            try:
                pygame.gfxdraw.filled_circle(
                    screen, int(self.x), int(self.y), int(self.radius), 
                    (*self.color, alpha)
                )
            except OverflowError: # Can happen if radius becomes negative float
                pass