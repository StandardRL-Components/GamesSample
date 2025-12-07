import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:04:52.494420
# Source Brief: brief_00798.md
# Brief Index: 798
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A falling block puzzle game. Match three or more blocks of the same color "
        "to clear them and score points before the stack reaches the top."
    )
    user_guide = (
        "Controls: ←→ to move the falling block, ↑ to change its color, and ↓ to drop it faster."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAY_AREA_WIDTH_BLOCKS = 10
    PLAY_AREA_HEIGHT_BLOCKS = 12
    BLOCK_SIZE = 32
    PLAY_AREA_WIDTH_PX = PLAY_AREA_WIDTH_BLOCKS * BLOCK_SIZE
    PLAY_AREA_HEIGHT_PX = PLAY_AREA_HEIGHT_BLOCKS * BLOCK_SIZE
    PLAY_AREA_X_OFFSET = (SCREEN_WIDTH - PLAY_AREA_WIDTH_PX) // 2
    PLAY_AREA_Y_OFFSET = (SCREEN_HEIGHT - PLAY_AREA_HEIGHT_PX) // 2

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_BORDER = (80, 80, 100)
    BLOCK_COLORS = {
        1: (255, 50, 50),   # Red (A)
        2: (50, 255, 50),   # Green (B)
        3: (50, 100, 255)   # Blue (C)
    }
    BLOCK_GLOW_COLORS = {
        1: (128, 25, 25),
        2: (25, 128, 25),
        3: (25, 50, 128)
    }
    UI_COLOR = (220, 220, 240)

    # Game parameters
    MAX_STEPS = 1000
    WIN_SCORE = 100
    INITIAL_FALL_SPEED = 0.5
    MAX_FALL_SPEED = 2.0
    FALL_SPEED_INCREMENT = 0.01
    DIFFICULTY_INTERVAL = 200
    PLAYER_MOVE_SPEED = 4.0

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
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.render_mode = render_mode
        self.active_block = None
        self.grid = None
        self.particles = None
        self.fall_speed = None
        self.steps = None
        self.score = None
        self.cleared_blocks_total = None
        self.game_over = None
        
        # Initialize state variables in reset
        # self.reset() # This is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.cleared_blocks_total = 0
        self.game_over = False
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.grid = np.zeros((self.PLAY_AREA_HEIGHT_BLOCKS, self.PLAY_AREA_WIDTH_BLOCKS), dtype=int)
        self.particles = []
        self._spawn_new_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0

        # --- Action Handling ---
        movement = action[0]
        # space_held = action[1] == 1 # Not used per brief
        # shift_held = action[2] == 1 # Not used per brief

        # Action 1: Transform block
        if movement == 1:
            # Sfx: transform_sound
            self.active_block['type'] = (self.active_block['type'] % 3) + 1
        # Action 3: Move left
        elif movement == 3:
            self.active_block['x'] = max(
                self.PLAY_AREA_X_OFFSET,
                self.active_block['x'] - self.PLAYER_MOVE_SPEED
            )
        # Action 4: Move right
        elif movement == 4:
            self.active_block['x'] = min(
                self.PLAY_AREA_X_OFFSET + self.PLAY_AREA_WIDTH_PX - self.BLOCK_SIZE,
                self.active_block['x'] + self.PLAYER_MOVE_SPEED
            )
        
        # --- Game Logic Update ---
        # Apply gravity unless action 2 (down) is pressed
        if movement == 2:
             self.active_block['y'] += self.PLAYER_MOVE_SPEED # Faster drop
        else:
            self.active_block['y'] += self.fall_speed

        # Update difficulty
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.fall_speed = min(self.MAX_FALL_SPEED, self.fall_speed + self.FALL_SPEED_INCREMENT)

        # Update particles
        self._update_particles()
        
        # --- Collision and Settling ---
        current_grid_x = int(round((self.active_block['x'] - self.PLAY_AREA_X_OFFSET) / self.BLOCK_SIZE))
        current_grid_y = int((self.active_block['y'] - self.PLAY_AREA_Y_OFFSET) / self.BLOCK_SIZE)
        
        collision = False
        if current_grid_y + 1 >= self.PLAY_AREA_HEIGHT_BLOCKS:
            collision = True
        elif current_grid_y >= -1 and 0 <= current_grid_x < self.PLAY_AREA_WIDTH_BLOCKS and self.grid[current_grid_y + 1, current_grid_x] != 0:
            collision = True
            
        if collision:
            # Sfx: block_land_sound
            settle_y = current_grid_y
            settle_x = current_grid_x
            
            if 0 <= settle_y < self.PLAY_AREA_HEIGHT_BLOCKS and 0 <= settle_x < self.PLAY_AREA_WIDTH_BLOCKS:
                self.grid[settle_y, settle_x] = self.active_block['type']
                
                # Check for matches and process chains
                cleared_this_turn = self._check_and_clear_matches()
                
                if cleared_this_turn > 0:
                    # Sfx: clear_blocks_sound
                    reward += cleared_this_turn  # +1 per block cleared
                    self.score += cleared_this_turn
                    self.cleared_blocks_total += cleared_this_turn
                
                # Check for loss condition
                if settle_y == 0:
                    self.game_over = True
                    reward -= 100 # Loss penalty
                else:
                    self._spawn_new_block()
                    # Check if new block spawn causes immediate game over
                    spawn_x = int(round((self.active_block['x'] - self.PLAY_AREA_X_OFFSET) / self.BLOCK_SIZE))
                    if self.grid[0, spawn_x] != 0:
                         self.game_over = True
                         reward -= 100
            else:
                # Block settled out of bounds (e.g. top edge)
                self.game_over = True
                reward -= 100
        else:
            # Continuous reward for good positioning
            if current_grid_y + 1 < self.PLAY_AREA_HEIGHT_BLOCKS and 0 <= current_grid_x < self.PLAY_AREA_WIDTH_BLOCKS and self.grid[current_grid_y + 1, current_grid_x] != 0:
                potential_x = current_grid_x
                potential_y = current_grid_y
                neighbor_count = 0
                for dx, dy in [(0, 1), (1, 0), (-1, 0)]:
                    nx, ny = potential_x + dx, potential_y + dy
                    if 0 <= nx < self.PLAY_AREA_WIDTH_BLOCKS and 0 <= ny < self.PLAY_AREA_HEIGHT_BLOCKS:
                        if self.grid[ny, nx] == self.active_block['type']:
                            neighbor_count += 1
                if neighbor_count >= 2:
                    reward += 0.1

        # --- Termination Check ---
        truncated = False
        if self.cleared_blocks_total >= self.WIN_SCORE:
            self.game_over = True
            reward += 100 # Win bonus
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            truncated = True

        terminated = self.game_over and not truncated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _spawn_new_block(self):
        grid_x = self.np_random.integers(0, self.PLAY_AREA_WIDTH_BLOCKS)
        self.active_block = {
            'x': self.PLAY_AREA_X_OFFSET + grid_x * self.BLOCK_SIZE,
            'y': self.PLAY_AREA_Y_OFFSET - self.BLOCK_SIZE, # Start just above the screen
            'type': self.np_random.integers(1, 4)
        }

    def _check_and_clear_matches(self):
        total_cleared = 0
        while True:
            blocks_to_clear = set()
            for y in range(self.PLAY_AREA_HEIGHT_BLOCKS):
                for x in range(self.PLAY_AREA_WIDTH_BLOCKS):
                    if self.grid[y, x] != 0 and (x, y) not in blocks_to_clear:
                        connected = self._find_connected_blocks(x, y)
                        if len(connected) >= 3:
                            for block_pos in connected:
                                blocks_to_clear.add(block_pos)
            
            if not blocks_to_clear:
                break

            for x, y in blocks_to_clear:
                block_type = self.grid[y, x]
                self._create_particles(x, y, self.BLOCK_COLORS[block_type])
                self.grid[y, x] = 0
            
            total_cleared += len(blocks_to_clear)
            self._apply_gravity_to_grid()
            # Sfx: blocks_fall_sound
        
        return total_cleared

    def _find_connected_blocks(self, start_x, start_y):
        block_type = self.grid[start_y, start_x]
        if block_type == 0:
            return []

        q = [(start_x, start_y)]
        visited = set(q)
        connected = []

        while q:
            x, y = q.pop(0)
            connected.append((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.PLAY_AREA_WIDTH_BLOCKS and
                    0 <= ny < self.PLAY_AREA_HEIGHT_BLOCKS and
                    (nx, ny) not in visited and
                    self.grid[ny, nx] == block_type):
                    
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return connected

    def _apply_gravity_to_grid(self):
        for x in range(self.PLAY_AREA_WIDTH_BLOCKS):
            empty_row = self.PLAY_AREA_HEIGHT_BLOCKS - 1
            for y in range(self.PLAY_AREA_HEIGHT_BLOCKS - 1, -1, -1):
                if self.grid[y, x] != 0:
                    if y != empty_row:
                        self.grid[empty_row, x] = self.grid[y, x]
                        self.grid[y, x] = 0
                    empty_row -= 1
    
    def _create_particles(self, grid_x, grid_y, color):
        px = self.PLAY_AREA_X_OFFSET + grid_x * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        py = self.PLAY_AREA_Y_OFFSET + grid_y * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        for _ in range(15): # Create 15 particles
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifespan = self.np_random.integers(20, 41)
            self.particles.append({'x': px, 'y': py, 'vx': vx, 'vy': vy, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifespan'] -= 1
            p['vy'] += 0.1 # particle gravity

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw play area border and grid
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (
            self.PLAY_AREA_X_OFFSET - 2, self.PLAY_AREA_Y_OFFSET - 2,
            self.PLAY_AREA_WIDTH_PX + 4, self.PLAY_AREA_HEIGHT_PX + 4
        ), 2)
        for x in range(1, self.PLAY_AREA_WIDTH_BLOCKS):
            px = self.PLAY_AREA_X_OFFSET + x * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.PLAY_AREA_Y_OFFSET), (px, self.PLAY_AREA_Y_OFFSET + self.PLAY_AREA_HEIGHT_PX))
        for y in range(1, self.PLAY_AREA_HEIGHT_BLOCKS):
            py = self.PLAY_AREA_Y_OFFSET + y * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.PLAY_AREA_X_OFFSET, py), (self.PLAY_AREA_X_OFFSET + self.PLAY_AREA_WIDTH_PX, py))

        # Draw settled blocks
        for y in range(self.PLAY_AREA_HEIGHT_BLOCKS):
            for x in range(self.PLAY_AREA_WIDTH_BLOCKS):
                block_type = self.grid[y, x]
                if block_type != 0:
                    self._draw_block(x, y, block_type)
        
        # Draw active block
        if self.active_block:
            grid_x = (self.active_block['x'] - self.PLAY_AREA_X_OFFSET) / self.BLOCK_SIZE
            grid_y = (self.active_block['y'] - self.PLAY_AREA_Y_OFFSET) / self.BLOCK_SIZE
            self._draw_block(grid_x, grid_y, self.active_block['type'], is_active=True)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 40))
            color = p['color']
            radius = max(0, int(2 * (p['lifespan'] / 40)))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), radius, (*color, alpha))

    def _draw_block(self, grid_x, grid_y, block_type, is_active=False):
        px = self.PLAY_AREA_X_OFFSET + grid_x * self.BLOCK_SIZE
        py = self.PLAY_AREA_Y_OFFSET + grid_y * self.BLOCK_SIZE
        
        color = self.BLOCK_COLORS[block_type]
        glow_color = self.BLOCK_GLOW_COLORS[block_type]
        
        center_x = int(px + self.BLOCK_SIZE / 2)
        center_y = int(py + self.BLOCK_SIZE / 2)
        radius = int(self.BLOCK_SIZE / 2 * 0.85)

        # Glow effect
        glow_radius = int(radius * 1.5)
        if is_active:
            glow_radius = int(radius * 2.0)
        
        # Draw multiple circles for a soft glow
        for i in range(glow_radius, radius, -2):
            alpha = int(50 * (1 - (i - radius) / (glow_radius - radius)))
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, i, (*glow_color, alpha))

        # Main block circle
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
        
        # Highlight
        highlight_color = (255, 255, 255, 80)
        pygame.gfxdraw.arc(self.screen, center_x, center_y, int(radius * 0.8), 120, 240, highlight_color)
        pygame.gfxdraw.arc(self.screen, center_x, center_y, int(radius * 0.8)-1, 120, 240, highlight_color)


    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.UI_COLOR)
        self.screen.blit(score_text, (10, 10))
        
        cleared_text = self.font.render(f"CLEARED: {self.cleared_blocks_total}/{self.WIN_SCORE}", True, self.UI_COLOR)
        self.screen.blit(cleared_text, (10, 40))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text_str = "YOU WIN!" if self.cleared_blocks_total >= self.WIN_SCORE else "GAME OVER"
            status_text = pygame.font.SysFont("Consolas", 60, bold=True).render(status_text_str, True, self.UI_COLOR)
            text_rect = status_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(status_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cleared_blocks": self.cleared_blocks_total,
            "fall_speed": self.fall_speed
        }
    
    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def render(self):
        # This method is not strictly required by the new API but is good practice
        # for environments that might be used with `render_mode="human"`.
        return self._get_observation()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a display, so it will not run in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
        env = GameEnv()
        obs, info = env.reset()
        
        # Use a separate screen for human rendering
        human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Block Fall")
        clock = pygame.time.Clock()
        
        action = np.array([0, 0, 0]) # [movement, space, shift]
        running = True
        done = False
        
        while running:
            # Reset if game is over
            if done:
                obs, info = env.reset()
                done = False

            # Event handling
            action[0] = 0 # Reset movement action by default
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Keydown events for single actions like transform
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action[0] = 1 # Transform
                    elif event.key == pygame.K_r: # Reset
                        done = True 
                    elif event.key == pygame.K_ESCAPE:
                        running = False

            # Continuous key presses for movement
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            elif keys[pygame.K_DOWN]:
                action[0] = 2 # Hold
            
            # Unused actions
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Done: {done}")

            # Render the observation to the human-visible screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Limit to 30 FPS for human playability

        env.close()
    except pygame.error as e:
        print("Could not run human-playable test. Pygame display unavailable.")
        print(f"Error: {e}")