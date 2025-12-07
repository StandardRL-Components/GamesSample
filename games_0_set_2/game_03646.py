
# Generated: 2025-08-27T23:59:42.717368
# Source Brief: brief_03646.md
# Brief Index: 3646

        
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

    user_guide = (
        "Controls: Use arrow keys to highlight an adjacent tile for swapping. "
        "Press Space to perform the swap. Press Shift to deselect."
    )

    game_description = (
        "Swap adjacent colored tiles to create matches of 3 or more. "
        "Clear the entire board before the timer runs out to win!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_ROWS, GRID_COLS = 8, 10
    TILE_SIZE = 36
    GRID_WIDTH = GRID_COLS * TILE_SIZE
    GRID_HEIGHT = GRID_ROWS * TILE_SIZE
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + 20
    NUM_TILE_TYPES = 5
    MAX_STEPS = 1800  # 60 seconds * 30 FPS
    GAME_DURATION = 60.0  # seconds

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 60, 80)
    COLOR_TEXT = (220, 230, 240)
    COLOR_SCORE = (255, 215, 0)
    COLOR_TIMER_FG = (60, 220, 120)
    COLOR_TIMER_BG = (220, 60, 60)
    TILE_COLORS = [
        (230, 50, 50),   # Red
        (50, 200, 50),   # Green
        (50, 120, 230),  # Blue
        (230, 230, 50),  # Yellow
        (180, 50, 230),  # Purple
    ]
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 28, bold=True)
        
        self.render_mode = render_mode
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Game state
        self.board = None
        self.timer = self.GAME_DURATION
        self.selected_pos = None
        self.highlighted_pos = None
        self.animations = []
        self.particles = []
        
        # Input handling
        self.last_space_held = False
        self.last_shift_held = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.GAME_DURATION
        self.selected_pos = None
        self.highlighted_pos = None
        self.animations.clear()
        self.particles.clear()
        
        self._generate_board()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        dt = self.clock.tick(30) / 1000.0
        self.steps += 1
        reward = 0

        # --- Update Game State ---
        self.timer = max(0, self.timer - dt)
        
        self._handle_input(action)
        self._update_animations(dt)

        # Process game logic only when no animations are running
        if not self.animations:
            new_matches = self._find_matches()
            if new_matches:
                # sfx: match_found.wav
                reward += self._process_matches(new_matches)
                self.animations.append(FallAnimation(self.board, self, 0.3))
            
        # --- Check Termination ---
        terminated = False
        if self.timer <= 0:
            terminated = True
            reward = -100  # Penalty for timeout
            self.game_over = True
        elif self._is_board_clear():
            terminated = True
            reward += 100  # Bonus for clearing board
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        # Update button states for next frame
        self.last_space_held = action[1] == 1
        self.last_shift_held = action[2] == 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        if self.animations:  # Block input during animations
            return
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        if shift_pressed:
            # sfx: deselect.wav
            self.selected_pos = None
            self.highlighted_pos = None

        if movement > 0:
            if self.selected_pos is None:
                # sfx: select.wav
                self.selected_pos = (self.GRID_ROWS // 2, self.GRID_COLS // 2)
            
            r, c = self.selected_pos
            if movement == 1: # Up
                self.highlighted_pos = ((r - 1 + self.GRID_ROWS) % self.GRID_ROWS, c)
            elif movement == 2: # Down
                self.highlighted_pos = ((r + 1) % self.GRID_ROWS, c)
            elif movement == 3: # Left
                self.highlighted_pos = (r, (c - 1 + self.GRID_COLS) % self.GRID_COLS)
            elif movement == 4: # Right
                self.highlighted_pos = (r, (c + 1) % self.GRID_COLS)
            
        if space_pressed:
            if self.selected_pos and self.highlighted_pos:
                # sfx: swap.wav
                self.animations.append(SwapAnimation(self.selected_pos, self.highlighted_pos, self.board, self, 0.25))
                self.selected_pos = None
                self.highlighted_pos = None

    def _update_animations(self, dt):
        for anim in self.animations[:]:
            anim.update(dt)
            if anim.is_done:
                self.animations.remove(anim)
        
        for p in self.particles[:]:
            p.update(dt)
            if p.is_done:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_tiles()
        
        for anim in self.animations:
            anim.draw(self.screen)
        
        self._render_cursors()
        
        for p in self.particles:
            p.draw(self.screen)
            
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Board Logic ---
    def _generate_board(self):
        while True:
            self.board = self.np_random.integers(1, self.NUM_TILE_TYPES + 1, size=(self.GRID_ROWS, self.GRID_COLS))
            # Ensure no initial matches
            while self._find_matches():
                self.board = self.np_random.integers(1, self.NUM_TILE_TYPES + 1, size=(self.GRID_ROWS, self.GRID_COLS))
            # Ensure at least one move is possible
            if self._find_possible_moves():
                break

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.board[r, c] == 0: continue
                # Horizontal
                if c < self.GRID_COLS - 2 and self.board[r, c] == self.board[r, c+1] == self.board[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                # Vertical
                if r < self.GRID_ROWS - 2 and self.board[r, c] == self.board[r+1, c] == self.board[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _find_possible_moves(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Check swap right
                if c < self.GRID_COLS - 1:
                    self.board[r, c], self.board[r, c+1] = self.board[r, c+1], self.board[r, c]
                    if self._find_matches():
                        self.board[r, c], self.board[r, c+1] = self.board[r, c+1], self.board[r, c]
                        return True
                    self.board[r, c], self.board[r, c+1] = self.board[r, c+1], self.board[r, c]
                # Check swap down
                if r < self.GRID_ROWS - 1:
                    self.board[r, c], self.board[r+1, c] = self.board[r+1, c], self.board[r, c]
                    if self._find_matches():
                        self.board[r, c], self.board[r+1, c] = self.board[r+1, c], self.board[r, c]
                        return True
                    self.board[r, c], self.board[r+1, c] = self.board[r+1, c], self.board[r, c]
        return False

    def _process_matches(self, matches):
        reward = 0
        for r, c in matches:
            if self.board[r, c] != 0:
                reward += 1
                self.score += 10
                self._spawn_particles(r, c)
                self.board[r, c] = 0
        return reward

    def _spawn_particles(self, r, c):
        # This can be called on an already cleared tile during cascades
        # We need to find the original color, which is tricky.
        # For simplicity, we just won't spawn particles if tile is already 0.
        # A more complex implementation would store the color before clearing.
        if self.board[r,c] == 0: return
        
        tile_color_idx = self.board[r,c] - 1
        
        px, py = self._grid_to_pixel(r, c)
        px += self.TILE_SIZE // 2
        py += self.TILE_SIZE // 2
        
        color = self.TILE_COLORS[tile_color_idx]
        for _ in range(15):
            self.particles.append(Particle(px, py, color, self.np_random))

    def _is_board_clear(self):
        return np.sum(self.board) == 0

    # --- Rendering ---
    def _grid_to_pixel(self, r, c):
        return self.GRID_X + c * self.TILE_SIZE, self.GRID_Y + r * self.TILE_SIZE

    def _render_grid(self):
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y + r * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, y), (self.GRID_X + self.GRID_WIDTH, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X + c * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_HEIGHT))

    def _render_tiles(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile_val = self.board[r, c]
                if tile_val > 0:
                    # Check if this tile is being animated
                    is_animating = any(hasattr(anim, 'is_animating_tile') and anim.is_animating_tile(r, c) for anim in self.animations)
                    if not is_animating:
                        self._draw_tile(r, c, tile_val)

    def _draw_tile(self, r, c, tile_val, pos=None, size_mult=1.0):
        color = self.TILE_COLORS[tile_val - 1]
        px, py = pos if pos else self._grid_to_pixel(r, c)
        
        size = int(self.TILE_SIZE * 0.85 * size_mult)
        offset = (self.TILE_SIZE - size) // 2
        
        rect = pygame.Rect(px + offset, py + offset, size, size)
        
        # Draw a slightly darker background for depth
        darker_color = tuple(max(0, val - 40) for val in color)
        pygame.draw.rect(self.screen, darker_color, rect.move(2, 2), border_radius=6)
        
        pygame.draw.rect(self.screen, color, rect, border_radius=6)
        
        # Add a subtle highlight
        lighter_color = tuple(min(255, val + 50) for val in color)
        pygame.draw.line(self.screen, lighter_color, rect.topleft, rect.topright, 1)
        pygame.draw.line(self.screen, lighter_color, rect.topleft, rect.bottomleft, 1)


    def _render_cursors(self):
        if self.selected_pos:
            r, c = self.selected_pos
            px, py = self._grid_to_pixel(r, c)
            rect = pygame.Rect(px, py, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 3, border_radius=4)
        
        if self.highlighted_pos:
            r, c = self.highlighted_pos
            px, py = self._grid_to_pixel(r, c)
            rect = pygame.Rect(px, py, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, (255, 255, 0), rect, 3, border_radius=4)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (20, 10))
        
        # Timer bar
        timer_width = self.SCREEN_WIDTH - 40
        timer_ratio = self.timer / self.GAME_DURATION
        current_width = int(timer_width * timer_ratio)
        
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BG, (20, self.SCREEN_HEIGHT - 30, timer_width, 20), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_FG, (20, self.SCREEN_HEIGHT - 30, current_width, 20), border_radius=5)
        
        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self._is_board_clear():
                msg = "BOARD CLEARED!"
                color = (100, 255, 100)
            else:
                msg = "TIME UP!"
                color = (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

# --- Animation and Effect Classes ---

class SwapAnimation:
    def __init__(self, pos1, pos2, board, env, duration):
        self.pos1, self.pos2 = pos1, pos2
        self.board = board
        self.env = env
        self.duration = duration
        self.progress = 0
        self.is_done = False

        self.tile1_val = self.board[pos1]
        self.tile2_val = self.board[pos2]
        
        self.start_px1 = self.env._grid_to_pixel(*pos1)
        self.start_px2 = self.env._grid_to_pixel(*pos2)

    def update(self, dt):
        if self.is_done: return
        self.progress += dt / self.duration
        if self.progress >= 1:
            self.progress = 1
            self.is_done = True
            self.board[self.pos1], self.board[self.pos2] = self.board[self.pos2], self.board[self.pos1]
            
            # Check for matches after swap
            matches = self.env._find_matches()
            if not matches:
                # Invalid swap, animate back
                self.env.animations.append(SwapAnimation(self.pos1, self.pos2, self.board, self.env, self.duration))

    def draw(self, screen):
        p = self.progress
        
        px1, py1 = self.start_px1
        px2, py2 = self.start_px2
        
        curr_px1 = (px1 + (px2 - px1) * p, py1 + (py2 - py1) * p)
        curr_px2 = (px2 + (px1 - px2) * p, py2 + (py1 - py2) * p)
        
        if self.tile1_val > 0:
            self.env._draw_tile(0, 0, self.tile1_val, pos=curr_px1)
        if self.tile2_val > 0:
            self.env._draw_tile(0, 0, self.tile2_val, pos=curr_px2)
            
    def is_animating_tile(self, r, c):
        return (r, c) == self.pos1 or (r, c) == self.pos2

class FallAnimation:
    def __init__(self, board, env, duration):
        self.board = board
        self.env = env
        self.duration = duration
        self.progress = 0
        self.is_done = False
        self.falling_tiles = []
        
        new_board = np.zeros_like(self.board)
        for c in range(self.env.GRID_COLS):
            dest_r = self.env.GRID_ROWS - 1
            for r in range(self.env.GRID_ROWS - 1, -1, -1):
                if self.board[r, c] > 0:
                    val = self.board[r, c]
                    new_board[dest_r, c] = val
                    if dest_r != r:
                        start_pos = self.env._grid_to_pixel(r, c)
                        end_pos = self.env._grid_to_pixel(dest_r, c)
                        self.falling_tiles.append({'val': val, 'start': start_pos, 'end': end_pos})
                    dest_r -= 1
        
        self.board[:, :] = new_board
        
        # Fill empty top cells
        for r in range(self.env.GRID_ROWS):
            for c in range(self.env.GRID_COLS):
                if self.board[r, c] == 0:
                    val = self.env.np_random.integers(1, self.env.NUM_TILE_TYPES + 1)
                    self.board[r, c] = val
                    start_pos = self.env._grid_to_pixel(-1, c) # Start from above the grid
                    end_pos = self.env._grid_to_pixel(r, c)
                    self.falling_tiles.append({'val': val, 'start': start_pos, 'end': end_pos})

    def update(self, dt):
        if self.is_done: return
        self.progress += dt / self.duration
        if self.progress >= 1:
            self.progress = 1
            self.is_done = True
            
    def draw(self, screen):
        p = self.progress
        for tile in self.falling_tiles:
            start_x, start_y = tile['start']
            end_x, end_y = tile['end']
            
            curr_x = start_x + (end_x - start_x) * p
            curr_y = start_y + (end_y - start_y) * p
            
            self.env._draw_tile(0, 0, tile['val'], pos=(curr_x, curr_y))
            
    def is_animating_tile(self, r, c):
        return False # The board is already updated, we just draw over it

class Particle:
    def __init__(self, x, y, color, rng):
        self.x, self.y = x, y
        self.color = color
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(50, 120)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = rng.uniform(0.3, 0.7)
        self.age = 0
        self.is_done = False

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 150 * dt  # Gravity
        self.age += dt
        if self.age >= self.lifespan:
            self.is_done = True

    def draw(self, screen):
        life_ratio = self.age / self.lifespan
        size = int(max(0, (1 - life_ratio) * 6))
        alpha = int(max(0, (1 - life_ratio) * 255))
        
        temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*self.color, alpha), (size, size), size)
        screen.blit(temp_surf, (int(self.x) - size, int(self.y) - size), special_flags=pygame.BLEND_RGBA_ADD)

if __name__ == '__main__':
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    
    # Use a persistent action array
    current_action = env.action_space.sample()
    current_action.fill(0)
    
    while running:
        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # --- Map keyboard to MultiDiscrete action space ---
        keys = pygame.key.get_pressed()
        
        # Reset movement
        current_action[0] = 0
        
        if keys[pygame.K_UP]:
            current_action[0] = 1
        elif keys[pygame.K_DOWN]:
            current_action[0] = 2
        elif keys[pygame.K_LEFT]:
            current_action[0] = 3
        elif keys[pygame.K_RIGHT]:
            current_action[0] = 4
        
        current_action[1] = 1 if keys[pygame.K_SPACE] else 0
        current_action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(current_action)
        
        if reward != 0:
            print(f"Reward: {reward}")
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            current_action.fill(0)

        # --- Render to screen ---
        # The environment observation is already the rendered frame
        # We just need to display it
        # Note: Gymnasium's obs is (H, W, C), Pygame needs (W, H, C)
        # The env's _get_observation already handles the transpose for the agent
        # but for direct rendering, we use the internal screen surface.
        display_surface = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Match-3 Environment")
        display_surface.blit(env.screen, (0, 0))
        pygame.display.flip()

    pygame.quit()