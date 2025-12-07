import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:00:19.509498
# Source Brief: brief_02509.md
# Brief Index: 2509
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Move blocks to match three in a row, merging them into higher-value blocks. "
        "Reach the target score before the timer runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the selected block. "
        "Use space and shift to cycle through selectable blocks."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    GRID_COLS = 10
    GRID_ROWS = 6
    CELL_SIZE = 50
    GRID_AREA_WIDTH = GRID_COLS * CELL_SIZE
    GRID_AREA_HEIGHT = GRID_ROWS * CELL_SIZE
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_AREA_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_AREA_HEIGHT) // 2 + 20

    # --- COLORS ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_TEXT = (220, 230, 240)
    COLOR_TIMER_BAR = (40, 160, 220)
    COLOR_TIMER_BG = (40, 60, 80)
    COLOR_CURSOR = (255, 255, 0)

    BLOCK_DEFINITIONS = {
        1: {'color': (227, 86, 74), 'next': 2, 'score': 2},    # Red
        2: {'color': (74, 189, 111), 'next': 3, 'score': 3},   # Green
        3: {'color': (74, 137, 227), 'next': 4, 'score': 4},   # Blue
        4: {'color': (227, 204, 74), 'next': 6, 'score': 6},   # Yellow
        6: {'color': (171, 74, 227), 'next': 9, 'score': 9},   # Purple
        9: {'color': (227, 143, 74), 'next': 12, 'score': 12}, # Orange
        12: {'color': (74, 227, 218), 'next': 15, 'score': 15},# Cyan
        15: {'color': (227, 74, 158), 'next': 20, 'score': 20},# Magenta
        20: {'color': (255, 255, 255), 'next': None, 'score': 30} # White
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 24)
        self.font_small = pygame.font.SysFont("Arial", 18)

        # Game parameters
        self.max_steps = 1000
        self.score_target = 200
        self.max_timer_seconds = 60
        self.max_timer_frames = self.max_timer_seconds * self.FPS
        self.inefficient_move_penalty_frames = 1 * self.FPS

        # State variables
        self.grid = None
        self.score = 0
        self.timer = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        self.selected_coords = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.animated_blocks = [] # For glow and pop-in effects
        
        # self.reset() is called by the environment wrapper, no need to call it here.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.timer = self.max_timer_frames
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.animated_blocks = []

        self._populate_grid()
        
        return self._get_observation(), self._get_info()

    def _populate_grid(self):
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        
        # Place a few initial blocks
        initial_block_count = 5
        for _ in range(initial_block_count):
            r, c = self.np_random.integers(0, self.GRID_ROWS), self.np_random.integers(0, self.GRID_COLS)
            while self.grid[r, c] != 0:
                r, c = self.np_random.integers(0, self.GRID_ROWS), self.np_random.integers(0, self.GRID_COLS)
            self.grid[r, c] = self.np_random.choice([1, 2])
        
        # Set initial selection
        block_coords = list(zip(*np.where(self.grid > 0)))
        if block_coords:
            self.selected_coords = block_coords[0]
        else: # Should not happen with initial blocks, but as a safeguard
            self._populate_grid()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        self._update_animations()

        # 1. Handle selection changes (on button press)
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        if space_pressed:
            self._cycle_selection(forward=True)
        elif shift_pressed:
            self._cycle_selection(forward=False)

        # 2. Handle movement
        if movement > 0 and self.selected_coords is not None:
            move_map = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)} # up, down, left, right
            dr, dc = move_map[movement]
            
            move_reward, score_gain = self._move_block(dr, dc)
            reward += move_reward
            self.score += score_gain

        # 3. Update timer and step counter
        self.timer -= 1
        self.steps += 1

        # 4. Check for termination
        terminated = False
        truncated = False
        if self.score >= self.score_target:
            terminated = True
            reward += 100
            self.game_over = True
            self.win_message = "YOU WIN!"
        elif self.timer <= 0:
            self.timer = 0
            terminated = True
            reward -= 50
            self.game_over = True
            self.win_message = "TIME'S UP!"
        elif self.steps >= self.max_steps:
            truncated = True
            terminated = True # In new Gymnasium API, truncated also implies terminated
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _cycle_selection(self, forward=True):
        block_coords = sorted(list(zip(*np.where(self.grid > 0))))
        if not block_coords:
            self.selected_coords = None
            return

        if self.selected_coords not in block_coords:
            self.selected_coords = block_coords[0]
            return
        
        current_index = block_coords.index(self.selected_coords)
        direction = 1 if forward else -1
        next_index = (current_index + direction) % len(block_coords)
        self.selected_coords = block_coords[next_index]

    def _move_block(self, dr, dc):
        r, c = self.selected_coords
        nr, nc = r + dr, c + dc

        # Check bounds
        if not (0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS):
            return 0, 0 # Invalid move, no penalty
        
        # Check if destination is empty
        if self.grid[nr, nc] != 0:
            return 0, 0 # Blocked move, no penalty

        # Execute move
        block_value = self.grid[r, c]
        self.grid[r, c] = 0
        self.grid[nr, nc] = block_value
        self.selected_coords = (nr, nc)

        # Check for merges
        reward, score_gain = self._check_and_process_merges(nr, nc)
        
        if reward == 0: # No merge occurred, inefficient move
            reward = -0.1
            self.timer = max(0, self.timer - self.inefficient_move_penalty_frames)
        
        return reward, score_gain

    def _check_and_process_merges(self, r, c):
        block_value = self.grid[r, c]
        if block_value == 0: return 0, 0

        # Check horizontal
        row = self.grid[r, :]
        for i in range(self.GRID_COLS - 2):
            if row[i] == row[i+1] == row[i+2] == block_value and block_value != 0:
                return self._process_merge([(r, i), (r, i+1), (r, i+2)])

        # Check vertical
        col = self.grid[:, c]
        for i in range(self.GRID_ROWS - 2):
            if col[i] == col[i+1] == col[i+2] == block_value and block_value != 0:
                return self._process_merge([(i, c), (i+1, c), (i+2, c)])
        
        return 0, 0

    def _process_merge(self, coords_to_merge):
        # Determine merge point (center block) and value
        merge_point = coords_to_merge[1]
        old_value = self.grid[merge_point]
        
        # Get new block info
        block_def = self.BLOCK_DEFINITIONS.get(old_value)
        if not block_def or block_def['next'] is None:
            return 0, 0 # Cannot merge max-level blocks

        new_value = block_def['next']
        score_gain = block_def['score']
        
        # Clear old blocks
        for r, c in coords_to_merge:
            self.grid[r, c] = 0
            
        # Place new block
        self.grid[merge_point] = new_value
        
        # Visual effects
        self._add_animated_block(merge_point, 'pop', 15) # 0.5s pop-in
        self._add_animated_block(merge_point, 'glow', 30) # 1s glow
        
        merge_pos_px = self._grid_to_pixel(merge_point[0], merge_point[1])
        self._create_particles(merge_pos_px, block_def['color'], 30)
        
        # Reward for creating a higher value block
        reward = 10
        
        # If selection was part of merge, deselect
        if self.selected_coords in coords_to_merge:
            self.selected_coords = merge_point

        # Spawn a new random low-level block
        self._spawn_new_block()
        
        return reward, score_gain

    def _spawn_new_block(self):
        empty_cells = list(zip(*np.where(self.grid == 0)))
        if not empty_cells:
            return # Grid is full
        
        idx = self.np_random.integers(0, len(empty_cells))
        r, c = empty_cells[idx]
        self.grid[r, c] = self.np_random.choice([1, 2], p=[0.7, 0.3])
        self._add_animated_block((r, c), 'pop', 15)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _grid_to_pixel(self, r, c):
        x = self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        return x, y

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_AREA_WIDTH, y), 2)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_AREA_HEIGHT), 2)

        # Draw blocks
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                val = self.grid[r, c]
                if val > 0:
                    self._draw_block(r, c, val)

        # Draw cursor
        if self.selected_coords and not self.game_over:
            r, c = self.selected_coords
            rect = pygame.Rect(self.GRID_OFFSET_X + c * self.CELL_SIZE, self.GRID_OFFSET_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 4, border_radius=8)

        # Draw particles
        self._update_and_draw_particles()

    def _draw_block(self, r, c, val):
        block_def = self.BLOCK_DEFINITIONS.get(val)
        if not block_def: return
        
        color = block_def['color']
        rect = pygame.Rect(self.GRID_OFFSET_X + c * self.CELL_SIZE, self.GRID_OFFSET_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        
        # Check for animations
        scale = 1.0
        glow_alpha = 0
        for anim in self.animated_blocks:
            if anim['coords'] == (r, c):
                if anim['type'] == 'pop':
                    progress = 1.0 - (anim['life'] / anim['max_life'])
                    scale = 0.1 + 0.9 * (1 - (1 - progress)**3) # Ease out cubic
                elif anim['type'] == 'glow':
                    progress = 1.0 - (anim['life'] / anim['max_life'])
                    glow_alpha = 128 * math.sin(progress * math.pi) # Sine pulse

        # Draw glow
        if glow_alpha > 0:
            glow_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            glow_radius = int(self.CELL_SIZE * 0.6)
            pygame.draw.circle(glow_surf, (*color, glow_alpha), (self.CELL_SIZE//2, self.CELL_SIZE//2), glow_radius)
            self.screen.blit(glow_surf, rect.topleft)

        # Draw block
        block_size = int(self.CELL_SIZE * 0.85 * scale)
        block_rect = pygame.Rect(0, 0, block_size, block_size)
        block_rect.center = rect.center
        pygame.draw.rect(self.screen, color, block_rect, 0, border_radius=int(8 * scale))
        
        # Draw value text
        if scale > 0.5:
            text_surf = self.font_medium.render(str(val), True, self.COLOR_BG)
            text_rect = text_surf.get_rect(center=block_rect.center)
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_medium.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 10))

        # Timer
        timer_rect_bg = pygame.Rect(self.SCREEN_WIDTH - 220, 15, 200, 20)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BG, timer_rect_bg, 0, 5)
        
        timer_ratio = self.timer / self.max_timer_frames
        timer_width = int(200 * timer_ratio)
        timer_rect_fg = pygame.Rect(self.SCREEN_WIDTH - 220, 15, timer_width, 20)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, timer_rect_fg, 0, 5)
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text_surf = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
            end_text_rect = end_text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text_surf, end_text_rect)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(15, 30), # 0.5 to 1 second
                'radius': self.np_random.uniform(2, 5),
                'color': color
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2 # Gravity
            p['life'] -= 1
            p['radius'] -= 0.1

            if p['life'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p['life'] / 30))))
                color_with_alpha = (*p['color'], alpha)
                temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color_with_alpha, (p['radius'], p['radius']), p['radius'])
                self.screen.blit(temp_surf, (p['pos'][0] - p['radius'], p['pos'][1] - p['radius']))
    
    def _add_animated_block(self, coords, anim_type, duration):
        # Remove existing animations of the same type for this block
        self.animated_blocks = [a for a in self.animated_blocks if not (a['coords'] == coords and a['type'] == anim_type)]
        self.animated_blocks.append({'coords': coords, 'type': anim_type, 'life': duration, 'max_life': duration})

    def _update_animations(self):
        for anim in self.animated_blocks[:]:
            anim['life'] -= 1
            if anim['life'] <= 0:
                self.animated_blocks.remove(anim)
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is NOT part of the Gymnasium environment
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a display for manual testing
    pygame.display.set_caption("Manual Game Test")
    display_surf = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    running = True
    terminated = False
    truncated = False
    
    # Game loop
    while running:
        # --- Action selection ---
        movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    terminated = False
                    truncated = False
                elif event.key == pygame.K_ESCAPE:
                    running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Step environment ---
        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render ---
        # The observation is the rendered frame, so we just need to display it
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_surf.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(GameEnv.FPS)

    env.close()