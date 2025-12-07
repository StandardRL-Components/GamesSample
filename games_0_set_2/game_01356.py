
# Generated: 2025-08-27T16:52:13.760900
# Source Brief: brief_01356.md
# Brief Index: 1356

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set a dummy video driver for headless operation
os.environ['SDL_VIDEODRIVER'] = 'dummy'

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to move the selector. Press Space to select a block and attempt a match."
    )

    # User-facing game description
    game_description = (
        "A fast-paced, grid-based color-matching puzzle. Chain matches to maximize your score before the timer runs out."
    )

    # Frames auto-advance for real-time timer
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 6
        self.GRID_OFFSET_X, self.GRID_OFFSET_Y = 160, 40
        self.BLOCK_SIZE = 50
        self.GRID_WIDTH = self.GRID_SIZE * self.BLOCK_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.BLOCK_SIZE
        
        self.FPS = 30
        self.MAX_TIME = 60 * self.FPS  # 60 seconds
        self.SCORE_GOAL = 1000

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID_BG = (30, 35, 50)
        self.COLOR_GRID_LINE = (50, 55, 70)
        self.BLOCK_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_TIMER_START = (0, 200, 0)
        self.COLOR_TIMER_END = (200, 0, 0)

        # Action/Observation Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont('Consolas', 36, bold=True)
        self.font_small = pygame.font.SysFont('Consolas', 20)
        
        # Game state variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.time_left = None
        self.game_over = None
        self.steps = None
        self.particles = None
        self.game_phase = None
        self.animation_timer = None
        self.blocks_in_animation = None
        self.invalid_move_shake = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_TIME
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.particles = []
        
        self._create_grid()
        self.game_phase = "IDLE" # "IDLE", "MATCH", "FALL"
        self.animation_timer = 0
        self.blocks_in_animation = []
        self.invalid_move_shake = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_left -= 1
        reward = 0

        # Unpack factorized action
        movement, space_press, _ = action[0], action[1] == 1, action[2] == 1
        
        self._update_animations()

        if self.game_phase == "IDLE":
            if space_press:
                # Sound: select_block.wav
                matches = self._find_matches(self.cursor_pos[0], self.cursor_pos[1])
                if len(matches) >= 3:
                    # Sound: match_success.wav
                    self.game_phase = "MATCH"
                    self.animation_timer = 10 # frames for match anim
                    self.blocks_in_animation = [{"pos": pos, "start_frame": self.steps} for pos in matches]
                    self._spawn_particles(matches)

                    reward += len(matches) # +1 per block
                    if len(matches) == 4: reward += 5
                    if len(matches) >= 5: reward += 10
                else:
                    # Sound: invalid_move.wav
                    reward -= 0.5
                    self.invalid_move_shake = 5
            elif movement != 0:
                self._move_cursor(movement)
            else: # no-op
                reward -= 0.1
        
        self._update_particles()
        
        terminated = self.time_left <= 0 or self.score >= self.SCORE_GOAL
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.SCORE_GOAL:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Timeout penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_animations(self):
        if self.invalid_move_shake > 0:
            self.invalid_move_shake -= 1

        if self.game_phase == "IDLE":
            return
        
        self.animation_timer -= 1
        if self.animation_timer > 0:
            return

        if self.game_phase == "MATCH":
            # Sound: blocks_fall.wav
            self._remove_and_shift_blocks()
            self.game_phase = "FALL"
            self.animation_timer = 10 # frames for fall anim
        elif self.game_phase == "FALL":
            self._finalize_fall()
            self.game_phase = "IDLE"
            # Simplification: No automatic chain reactions. Agent must trigger next match.

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_left": self.time_left}

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=10)

        # Draw blocks
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                block_info = self.grid[y][x]
                if block_info is None:
                    continue
                
                color_index = block_info["color"]
                block_color = self.BLOCK_COLORS[color_index]
                
                # Calculate position with fall animation
                fall_offset = 0
                if self.game_phase == "FALL" and block_info["fall_dist"] > 0:
                    progress = 1.0 - (self.animation_timer / 10.0)
                    fall_offset = -block_info["fall_dist"] * self.BLOCK_SIZE * (1.0 - progress)
                
                px = self.GRID_OFFSET_X + x * self.BLOCK_SIZE
                py = self.GRID_OFFSET_Y + y * self.BLOCK_SIZE + fall_offset
                
                block_rect = pygame.Rect(px + 3, py + 3, self.BLOCK_SIZE - 6, self.BLOCK_SIZE - 6)

                # Handle match animation
                is_matching = any(b["pos"] == (x, y) for b in self.blocks_in_animation)
                if self.game_phase == "MATCH" and is_matching:
                    progress = 1.0 - (self.animation_timer / 10.0)
                    flash_color = (255, 255, 255)
                    lerp_color = tuple(int(block_color[i] + (flash_color[i] - block_color[i]) * progress) for i in range(3))
                    pygame.draw.rect(self.screen, lerp_color, block_rect, border_radius=8)
                    size = int((self.BLOCK_SIZE - 6) * (1.0 - progress))
                    shrink_rect = pygame.Rect(px + 3 + (self.BLOCK_SIZE - 6 - size)//2, 
                                              py + 3 + (self.BLOCK_SIZE - 6 - size)//2, 
                                              size, size)
                    pygame.draw.rect(self.screen, lerp_color, shrink_rect, border_radius=int(8 * (1.0 - progress)))
                else:
                    highlight_color = tuple(min(255, c + 40) for c in block_color)
                    shadow_color = tuple(max(0, c - 40) for c in block_color)
                    pygame.draw.rect(self.screen, shadow_color, block_rect, border_radius=8)
                    pygame.draw.rect(self.screen, block_color, block_rect.inflate(-4, -4), border_radius=6)
                    pygame.draw.rect(self.screen, highlight_color, block_rect.inflate(-10, -10), width=1, border_radius=4)


        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, 
                             (self.GRID_OFFSET_X + i * self.BLOCK_SIZE, self.GRID_OFFSET_Y), 
                             (self.GRID_OFFSET_X + i * self.BLOCK_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT))
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, 
                             (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.BLOCK_SIZE), 
                             (self.GRID_OFFSET_X + self.GRID_WIDTH, self.GRID_OFFSET_Y + i * self.BLOCK_SIZE))

        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), p['color'])

        # Draw cursor
        if self.game_phase == "IDLE":
            shake_x = (self.np_random.random() - 0.5) * self.invalid_move_shake if self.invalid_move_shake > 0 else 0
            shake_y = (self.np_random.random() - 0.5) * self.invalid_move_shake if self.invalid_move_shake > 0 else 0
            cursor_rect = pygame.Rect(
                self.GRID_OFFSET_X + self.cursor_pos[0] * self.BLOCK_SIZE + shake_x,
                self.GRID_OFFSET_Y + self.cursor_pos[1] * self.BLOCK_SIZE + shake_y,
                self.BLOCK_SIZE, self.BLOCK_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=3, border_radius=5)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer bar
        timer_width = self.WIDTH - self.GRID_OFFSET_X - self.GRID_WIDTH - 30
        timer_x = self.GRID_OFFSET_X + self.GRID_WIDTH + 10
        time_ratio = max(0, self.time_left / self.MAX_TIME)
        
        timer_color = tuple(int(self.COLOR_TIMER_END[i] + (self.COLOR_TIMER_START[i] - self.COLOR_TIMER_END[i]) * time_ratio) for i in range(3))
        
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (timer_x, 10, timer_width, 25), border_radius=5)
        if time_ratio > 0:
            pygame.draw.rect(self.screen, timer_color, (timer_x, 10, timer_width * time_ratio, 25), border_radius=5)

    def _create_grid(self):
        self.grid = [[None for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                self.grid[y][x] = {"color": self.np_random.integers(0, len(self.BLOCK_COLORS)), "fall_dist": 0}

        # Ensure at least one match is possible by forcing one
        if not self._has_possible_moves():
            y, x = self.np_random.integers(0, self.GRID_SIZE, size=2)
            color = self.grid[y][x]["color"]
            # Force a horizontal match
            if x < self.GRID_SIZE - 2:
                self.grid[y][x+1]["color"] = color
                self.grid[y][x+2]["color"] = color
            # Or a vertical match
            elif y < self.GRID_SIZE - 2:
                self.grid[y+1][x]["color"] = color
                self.grid[y+2][x]["color"] = color

    def _has_possible_moves(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if len(self._find_matches(c, r)) >= 3:
                    return True
        return False
        
    def _move_cursor(self, movement):
        # Sound: cursor_move.wav
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Up
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1) # Down
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Left
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1) # Right
    
    def _find_matches(self, start_x, start_y):
        if self.grid[start_y][start_x] is None:
            return []
        
        target_color = self.grid[start_y][start_x]["color"]
        q = [(start_x, start_y)]
        visited = set(q)
        matches = []

        while q:
            x, y = q.pop(0)
            matches.append((x, y))
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and (nx, ny) not in visited:
                    if self.grid[ny][nx] is not None and self.grid[ny][nx]["color"] == target_color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return matches

    def _spawn_particles(self, matched_blocks):
        center_x, center_y = 0, 0
        for x, y in matched_blocks:
            center_x += self.GRID_OFFSET_X + (x + 0.5) * self.BLOCK_SIZE
            center_y += self.GRID_OFFSET_Y + (y + 0.5) * self.BLOCK_SIZE
        center_x /= len(matched_blocks)
        center_y /= len(matched_blocks)

        color_index = self.grid[matched_blocks[0][1]][matched_blocks[0][0]]["color"]
        particle_color = self.BLOCK_COLORS[color_index]

        for _ in range(len(matched_blocks) * 5):
            angle = self.np_random.random() * 2 * math.pi
            speed = 2 + self.np_random.random() * 3
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'size': 3 + self.np_random.random() * 3,
                'lifespan': 15 + self.np_random.integers(0, 10),
                'color': particle_color
            })
    
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['size'] *= 0.95
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
    
    def _remove_and_shift_blocks(self):
        # Remove matched blocks
        for x, y in [b["pos"] for b in self.blocks_in_animation]:
            self.grid[y][x] = None
            self.score += 1

        # Reset fall distances and apply gravity
        for x in range(self.GRID_SIZE):
            empty_count = 0
            for y in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[y][x] is None:
                    empty_count += 1
                elif empty_count > 0:
                    self.grid[y + empty_count][x] = self.grid[y][x]
                    self.grid[y + empty_count][x]["fall_dist"] = empty_count
                    self.grid[y][x] = None
        
        # Refill top rows
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                if self.grid[y][x] is None:
                    self.grid[y][x] = {"color": self.np_random.integers(0, len(self.BLOCK_COLORS)), "fall_dist": y + 1}
    
    def _finalize_fall(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y][x]:
                    self.grid[y][x]["fall_dist"] = 0
        self.blocks_in_animation = []

    def close(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")