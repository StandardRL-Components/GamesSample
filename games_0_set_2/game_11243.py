import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:52:21.175461
# Source Brief: brief_01243.md
# Brief Index: 1243
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class Block:
    """Represents a single code block in the grid."""
    def __init__(self, grid_x, grid_y, state, block_size, padding):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.state = state  # 'neutral', 'corrupted', 'repaired'
        self.pixel_pos = (
            padding + grid_x * (block_size + padding) + block_size // 2,
            padding + grid_y * (block_size + padding) + block_size // 2
        )
        self.glitch_timer = 0
        self.spawn_animation_timer = 10 # For a nice fade-in effect

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Repair a corrupted digital system by finding and rewinding glitched data blocks. "
        "Trigger chain reactions to clear entire corrupted clusters before you run out of rewind energy."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to move the cursor. "
        "Press space to use 'Rewind Energy' on the selected block to repair it."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 50)
    COLOR_SCANLINE = (20, 20, 40)
    COLOR_NEUTRAL = (0, 100, 200)
    COLOR_CORRUPTED = (255, 50, 50)
    COLOR_REPAIRED = (50, 255, 100)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_VALUE = (50, 255, 100)
    COLOR_UI_WARN = (255, 200, 0)

    # Screen and Grid
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 10, 6
    BLOCK_SIZE = 50
    PADDING = 8

    # Game Mechanics
    INITIAL_REWIND_TIME = 10
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- GYM SPACES ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- PYGAME SETUP ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 32)
        
        # --- STATE VARIABLES ---
        self.grid = []
        self.cursor_pos = [0, 0]
        self.rewind_time = 0
        self.total_corrupted = 0
        self.repaired_count = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_space_held = False
        self.particles = []
        self.level = 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "level" in options:
            self.level = options["level"]
        else:
            self.level = 1
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.particles = []
        
        self._setup_level()

        return self._get_observation(), self._get_info()

    def _setup_level(self):
        """Initializes the game state for the current level."""
        self.rewind_time = self.INITIAL_REWIND_TIME
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        
        num_corrupted = min(self.GRID_COLS * self.GRID_ROWS -1, 5 + (self.level - 1) * 2)
        self.total_corrupted = num_corrupted
        self.repaired_count = 0
        
        all_pos = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)]
        # Use self.np_random.choice on an array of indices, then map to coordinates
        corrupted_indices = self.np_random.choice(len(all_pos), num_corrupted, replace=False)
        corrupted_coords = [all_pos[i] for i in corrupted_indices]

        self.grid = []
        for y in range(self.GRID_ROWS):
            row = []
            for x in range(self.GRID_COLS):
                state = 'corrupted' if (x, y) in corrupted_coords else 'neutral'
                row.append(Block(x, y, state, self.BLOCK_SIZE, self.PADDING))
            self.grid.append(row)

    def step(self, action):
        self.steps += 1
        reward = 0.0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Actions ---
        self._handle_movement(movement)
        
        rewind_triggered = space_held and not self.last_space_held
        if rewind_triggered:
            # sfx: rewind_activate.wav
            reward = self._handle_rewind()
        self.last_space_held = space_held

        # --- Check Termination ---
        win = self.repaired_count >= self.total_corrupted and self.total_corrupted > 0
        lose = self.rewind_time <= 0 and not win
        
        terminated = win or lose or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            self.game_over = True
            if win:
                reward += 100
                # sfx: level_complete.wav
            if lose:
                reward -= 100
                # sfx: game_over.wav
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_movement(self, movement):
        """Updates cursor position based on movement action."""
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        # Wrap around
        self.cursor_pos[0] %= self.GRID_COLS
        self.cursor_pos[1] %= self.GRID_ROWS

    def _handle_rewind(self):
        """Processes the rewind action on the selected block."""
        if self.rewind_time <= 0:
            # sfx: action_fail.wav
            return 0

        self.rewind_time -= 1
        cx, cy = self.cursor_pos
        block = self.grid[cy][cx]

        if block.state == 'corrupted':
            repaired_in_chain = self._trigger_chain_reaction(cx, cy)
            self.repaired_count += repaired_in_chain
            self.score += repaired_in_chain
            
            # Check if this was the last set of blocks
            if self.repaired_count >= self.total_corrupted:
                self.score += 5
                return float(repaired_in_chain + 5)
            return float(repaired_in_chain)
        
        else: # Wasted rewind on neutral or repaired block
            # sfx: glitch_effect.wav
            block.glitch_timer = 15 # frames
            self._spawn_particles(block.pixel_pos, 20, self.COLOR_NEUTRAL, 0.5)
            return 0

    def _trigger_chain_reaction(self, start_x, start_y):
        """Performs a flood-fill to repair connected corrupted blocks."""
        if not (0 <= start_y < self.GRID_ROWS and 0 <= start_x < self.GRID_COLS):
            return 0

        start_block = self.grid[start_y][start_x]
        if start_block.state != 'corrupted':
            return 0

        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        repaired_count = 0
        
        while q:
            x, y = q.popleft()
            block = self.grid[y][x]
            
            if block.state == 'corrupted':
                block.state = 'repaired'
                repaired_count += 1
                # sfx: repair_tick.wav
                self._spawn_particles(block.pixel_pos, 50, self.COLOR_REPAIRED, 2.0)
                
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= ny < self.GRID_ROWS and 0 <= nx < self.GRID_COLS and 
                        (nx, ny) not in visited and self.grid[ny][nx].state == 'corrupted'):
                        visited.add((nx, ny))
                        q.append((nx, ny))
        
        return repaired_count

    def _get_observation(self):
        """Renders the game state to the screen and returns it as a numpy array."""
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._update_and_render_particles()
        self._render_grid()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame returns (width, height, channels), but Gym expects (height, width, channels)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        """Draws the background grid and scanlines."""
        for y in range(0, self.SCREEN_HEIGHT, 4):
            pygame.draw.line(self.screen, self.COLOR_SCANLINE, (0, y), (self.SCREEN_WIDTH, y))

        for x in range(self.GRID_COLS + 1):
            px = self.PADDING + x * (self.BLOCK_SIZE + self.PADDING)
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_ROWS + 1):
            py = self.PADDING + y * (self.BLOCK_SIZE + self.PADDING)
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py))

    def _render_grid(self):
        """Renders all blocks in the grid."""
        # Render cursor first (so it's behind the block)
        cursor_block = self.grid[self.cursor_pos[1]][self.cursor_pos[0]]
        cx, cy = cursor_block.pixel_pos
        size = self.BLOCK_SIZE + self.PADDING
        cursor_rect = pygame.Rect(cx - size // 2, cy - size // 2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=5)
        
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                block = self.grid[y][x]
                self._render_block(block)

    def _render_block(self, block):
        """Renders a single block with its state-specific effects."""
        pos = list(block.pixel_pos)
        size = self.BLOCK_SIZE

        if block.glitch_timer > 0:
            pos[0] += self.np_random.integers(-5, 6)
            pos[1] += self.np_random.integers(-5, 6)
            block.glitch_timer -= 1
        
        rect = pygame.Rect(pos[0] - size // 2, pos[1] - size // 2, size, size)
        
        color = self.COLOR_NEUTRAL
        if block.state == 'corrupted':
            # Pulsating effect
            pulse = (math.sin(self.steps * 0.2 + block.grid_x) + 1) / 2
            r = int(150 + 105 * pulse)
            color = (r, 50, 50)
        elif block.state == 'repaired':
            color = self.COLOR_REPAIRED
            # Glow effect
            glow_size = int(size * (1.2 + 0.2 * math.sin(self.steps * 0.1 + block.grid_y)))
            glow_alpha = int(50 + 40 * math.sin(self.steps * 0.1 + block.grid_y))
            glow_color = (*self.COLOR_REPAIRED, glow_alpha)
            
            glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, glow_color, (glow_size, glow_size), glow_size)
            self.screen.blit(glow_surf, (pos[0] - glow_size, pos[1] - glow_size), special_flags=pygame.BLEND_RGBA_ADD)

        if block.spawn_animation_timer > 0:
            # Fade-in effect
            alpha = int(255 * (1 - block.spawn_animation_timer / 10.0))
            block_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(block_surf, (*color, alpha), (0, 0, size, size), border_radius=4)
            self.screen.blit(block_surf, rect.topleft)
            block.spawn_animation_timer -=1
        else:
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            pygame.draw.rect(self.screen, tuple(int(c*0.7) for c in color), rect, 2, border_radius=4)


    def _spawn_particles(self, pos, count, color, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(20, 41),
                'color': color
            })

    def _update_and_render_particles(self):
        """Updates particle physics and draws them."""
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['vel'][0] *= 0.98 # friction
            p['vel'][1] *= 0.98

            if p['life'] > 0:
                alpha = int(255 * (p['life'] / 40.0))
                color = (*p['color'], alpha)
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                
                # Use gfxdraw for anti-aliased circles for better quality
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 2, color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, color)

        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_ui(self):
        """Renders the user interface text."""
        ui_y = self.SCREEN_HEIGHT - 35
        
        # Rewind Time
        rewind_text = self.font_title.render("REWIND ENERGY", True, self.COLOR_UI_TEXT)
        self.screen.blit(rewind_text, (20, ui_y))
        
        time_color = self.COLOR_UI_VALUE if self.rewind_time > 3 else self.COLOR_UI_WARN
        rewind_val = self.font_title.render(str(self.rewind_time), True, time_color)
        self.screen.blit(rewind_val, (210, ui_y))

        # Repaired Blocks
        repaired_text = self.font_title.render("SYSTEM INTEGRITY", True, self.COLOR_UI_TEXT)
        self.screen.blit(repaired_text, (self.SCREEN_WIDTH - 280, ui_y))
        
        repaired_val_str = f"{self.repaired_count} / {self.total_corrupted}"
        repaired_val = self.font_title.render(repaired_val_str, True, self.COLOR_UI_VALUE)
        self.screen.blit(repaired_val, (self.SCREEN_WIDTH - 80, ui_y))
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "rewind_time": self.rewind_time,
            "repaired_count": self.repaired_count,
            "total_corrupted": self.total_corrupted,
            "cursor_pos": list(self.cursor_pos)
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Example Usage ---
    # To play manually, you might need a different setup to capture keyboard events.
    # This block demonstrates the programmatic Gym interface.
    
    # Un-comment the line below to run with a display window
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human viewing
    pygame.display.set_caption("Glitch Puzzle Environment")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print("Running environment with random actions...")
    print("Press ESC or close the window to quit.")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        # Sample a random action
        action = env.action_space.sample()
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS

        if terminated or truncated:
            print(f"Episode finished in {info['steps']} steps. Final score: {info['score']}. Total reward: {total_reward}")
            # Reset for a new episode
            obs, info = env.reset()
            terminated = False
            total_reward = 0

    env.close()