
# Generated: 2025-08-27T16:22:48.470126
# Source Brief: brief_01210.md
# Brief Index: 1210

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the selector. Press Space to clear a selected group of blocks. "
        "Hold Shift to forfeit the current game."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game. Match groups of 2 or more same-colored blocks to clear them from the board. "
        "Clear the entire board before the 60-second timer runs out to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 6
        self.GRID_OFFSET_X, self.GRID_OFFSET_Y = 120, 20
        self.CELL_SIZE = (self.HEIGHT - 40) // self.GRID_SIZE
        self.BLOCK_SIZE = int(self.CELL_SIZE * 0.9)
        self.BLOCK_OFFSET = (self.CELL_SIZE - self.BLOCK_SIZE) // 2
        
        self.FPS = 30
        self.MAX_TIME = 60.0
        self.MAX_STEPS = 1800 # 60 seconds * 30 FPS

        # --- Colors ---
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_EMPTY = (40, 50, 60)
        self.BLOCK_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
        ]
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_CURSOR = (255, 255, 0)
        
        # --- Gymnasium API Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State ---
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.timer = None
        self.steps = None
        self.game_over = None
        self.particles = None
        self.last_space_held = None
        self.score_popups = None
        
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self.np_random.integers(1, len(self.BLOCK_COLORS) + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        
        self.score = 0
        self.timer = self.MAX_TIME
        self.steps = 0
        self.game_over = False
        
        self.particles = []
        self.score_popups = []
        self.last_space_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1
        shift_held = action[2] == 1

        reward = 0
        self.steps += 1
        self.timer = max(0, self.timer - 1.0 / self.FPS)

        # --- Handle Input and Game Logic ---
        reward += self._handle_input(movement, space_held)
        self._update_particles()
        self._update_score_popups()

        # --- Check Termination Conditions ---
        terminated = False
        terminal_reward = 0
        
        # 1. Shift key pressed (forfeit)
        if shift_held:
            terminated = True
            # No specific reward/penalty for forfeiting
        
        # 2. Timer runs out (loss)
        if self.timer <= 0:
            terminated = True
            terminal_reward = -50
            self.game_over = True
            
        # 3. Board is cleared (win)
        if np.sum(self.grid) == 0:
            terminated = True
            terminal_reward = 100
            self.game_over = True
            
        # 4. Max steps reached
        if self.steps >= self.MAX_STEPS:
            terminated = True
            # No terminal reward if just max steps
            
        self.score += terminal_reward
        reward += terminal_reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)

        # --- Block Clearing (on space press) ---
        reward = 0
        space_pressed = space_held and not self.last_space_held
        if space_pressed:
            reward = self._clear_selected_group()
            
        self.last_space_held = space_held
        return reward

    def _clear_selected_group(self):
        x, y = self.cursor_pos
        if self.grid[y, x] == 0:
            return 0  # Cannot clear an empty space

        connected_blocks = self._find_connected_blocks(x, y)
        
        if len(connected_blocks) < 2:
            return 0 # Not a large enough group

        # --- Calculate Reward ---
        reward = 0
        # +1 for every block cleared
        reward += len(connected_blocks)
        # -0.2 for selecting a group of only 2 blocks
        if len(connected_blocks) == 2:
            reward -= 0.2
        # +5 bonus for clearing 4+ blocks
        if len(connected_blocks) >= 4:
            reward += 5
        
        # --- Update Score and Visuals ---
        self.score += reward
        self._create_score_popup(f"+{reward:.1f}", (x, y))

        # --- Clear Blocks and Create Particles ---
        for block_y, block_x in connected_blocks:
            color_index = self.grid[block_y, block_x] - 1
            self._create_particles(block_x, block_y, self.BLOCK_COLORS[color_index])
            self.grid[block_y, block_x] = 0
            # sfx: block_clear.wav
        
        # --- Apply Gravity and Refill ---
        self._apply_gravity_and_refill()
        # sfx: blocks_fall.wav

        return reward

    def _find_connected_blocks(self, start_x, start_y):
        target_color = self.grid[start_y, start_x]
        if target_color == 0:
            return set()

        q = deque([(start_y, start_x)])
        visited = set([(start_y, start_x)])
        
        while q:
            y, x = q.popleft()
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.GRID_SIZE and 0 <= nx < self.GRID_SIZE:
                    if (ny, nx) not in visited and self.grid[ny, nx] == target_color:
                        visited.add((ny, nx))
                        q.append((ny, nx))
        return visited

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_SIZE):
            empty_slots = 0
            for y in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[y, x] == 0:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.grid[y + empty_slots, x] = self.grid[y, x]
                    self.grid[y, x] = 0
            
            # Refill from the top
            for y in range(empty_slots):
                self.grid[y, x] = self.np_random.integers(1, len(self.BLOCK_COLORS) + 1)
    
    def _create_particles(self, grid_x, grid_y, color):
        cx = self.GRID_OFFSET_X + grid_x * self.CELL_SIZE + self.CELL_SIZE // 2
        cy = self.GRID_OFFSET_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE // 2
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': [cx, cy], 'vel': vel, 'life': lifespan, 'max_life': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _create_score_popup(self, text, grid_pos):
        x = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE
        self.score_popups.append({'text': text, 'pos': [x, y], 'life': 30, 'color': (255, 255, 100)})

    def _update_score_popups(self):
        for p in self.score_popups:
            p['pos'][1] -= 1
            p['life'] -= 1
        self.score_popups = [p for p in self.score_popups if p['life'] > 0]

    def _get_observation(self):
        # --- Clear screen ---
        self.screen.fill(self.COLOR_BG)
        
        # --- Render Game Elements ---
        self._render_grid()
        self._render_blocks()
        self._render_cursor()
        self._render_particles()
        
        # --- Render UI ---
        self._render_ui()
        self._render_score_popups()

        if self.game_over:
            self._render_game_over()
        
        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_grid(self):
        grid_width = self.GRID_SIZE * self.CELL_SIZE
        grid_height = self.GRID_SIZE * self.CELL_SIZE
        # Draw background for the grid area
        pygame.draw.rect(self.screen, self.COLOR_EMPTY, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y, grid_width, grid_height))
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, 
                             (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE),
                             (self.GRID_OFFSET_X + grid_width, self.GRID_OFFSET_Y + i * self.CELL_SIZE), 2)
            pygame.draw.line(self.screen, self.COLOR_GRID, 
                             (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y),
                             (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + grid_height), 2)

    def _render_blocks(self):
        wobble_phase = (self.steps % 30) / 30.0 * 2 * math.pi
        
        # Find the potential group to apply wobble effect
        hover_group = self._find_connected_blocks(self.cursor_pos[0], self.cursor_pos[1])
        if len(hover_group) < 2:
            hover_group = set()

        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_index = self.grid[y, x]
                if color_index > 0:
                    color = self.BLOCK_COLORS[color_index - 1]
                    rect_x = self.GRID_OFFSET_X + x * self.CELL_SIZE + self.BLOCK_OFFSET
                    rect_y = self.GRID_OFFSET_Y + y * self.CELL_SIZE + self.BLOCK_OFFSET
                    
                    size = self.BLOCK_SIZE
                    if (y, x) in hover_group:
                        size_mod = (math.sin(wobble_phase) + 1) / 2 * 0.1 + 1.0 # Scale between 1.0 and 1.1
                        size = int(self.BLOCK_SIZE * size_mod)
                        rect_x -= (size - self.BLOCK_SIZE) // 2
                        rect_y -= (size - self.BLOCK_SIZE) // 2

                    block_rect = pygame.Rect(rect_x, rect_y, size, size)
                    pygame.draw.rect(self.screen, color, block_rect, border_radius=5)
                    # Add a subtle highlight
                    highlight_color = (min(255, color[0]+40), min(255, color[1]+40), min(255, color[2]+40))
                    pygame.draw.rect(self.screen, highlight_color, (rect_x+2, rect_y+2, size-10, size//4), border_radius=3)


    def _render_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(self.GRID_OFFSET_X + x * self.CELL_SIZE,
                           self.GRID_OFFSET_Y + y * self.CELL_SIZE,
                           self.CELL_SIZE, self.CELL_SIZE)
        
        # Pulsing glow effect
        alpha = 128 + math.sin(pygame.time.get_ticks() * 0.005) * 127
        glow_color = (*self.COLOR_CURSOR, alpha)
        
        # Use gfxdraw for anti-aliased shapes
        for i in range(4):
            s = self.screen.copy()
            s.fill((0,0,0,0))
            s.set_colorkey((0,0,0,0))
            pygame.draw.rect(s, glow_color, rect.inflate(i*2, i*2), 2, border_radius=8)
            s.set_alpha(100 - i*20)
            self.screen.blit(s, (0,0))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = int(6 * (p['life'] / p['max_life']))
            if size > 0:
                # Create a temporary surface for alpha blending
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (size, size), size)
                self.screen.blit(particle_surf, (int(p['pos'][0]-size), int(p['pos'][1]-size)))

    def _render_ui(self):
        # --- Score ---
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # --- Timer ---
        timer_color = self.COLOR_TEXT if self.timer > 10 else (255, 100, 100)
        timer_text = self.font_large.render(f"TIME: {self.timer:.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(timer_text, timer_rect)

    def _render_score_popups(self):
        for p in self.score_popups:
            alpha = int(255 * (p['life'] / 30.0))
            color = (*p['color'], alpha)
            text_surf = self.font_medium.render(p['text'], True, color)
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=(int(p['pos'][0]), int(p['pos'][1])))
            self.screen.blit(text_surf, text_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        is_win = np.sum(self.grid) == 0
        msg = "BOARD CLEARED!" if is_win else "TIME'S UP!"
        color = (100, 255, 100) if is_win else (255, 100, 100)
        
        text = self.font_large.render(msg, True, color)
        text_rect = text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 20))
        self.screen.blit(text, text_rect)

        final_score_text = self.font_medium.render(f"Final Score: {int(self.score)}", True, self.COLOR_TEXT)
        final_score_rect = final_score_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 30))
        self.screen.blit(final_score_text, final_score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "cursor_pos": self.cursor_pos,
        }

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

    def close(self):
        pygame.quit()