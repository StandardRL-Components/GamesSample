
# Generated: 2025-08-27T14:03:12.485751
# Source Brief: brief_00570.md
# Brief Index: 570

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the block. ↓ to drop it instantly. Stack blocks to score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An arcade puzzle game. Stack falling blocks to score points within the time limit. "
        "Placing blue blocks near fewer neighbors grants a large bonus."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 10
    GRID_HEIGHT = 20
    CELL_SIZE = 18
    MAX_TIME = 60.0  # seconds
    FPS = 30
    WIN_SCORE = 100

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_PLAY_AREA = (10, 15, 25)
    COLOR_RED = (255, 70, 70)
    COLOR_BLUE = (70, 170, 255)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TIMER_WARN = (255, 200, 0)
    COLOR_TIMER_CRIT = (255, 50, 50)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        
        # Play area dimensions
        self.play_area_width = self.GRID_WIDTH * self.CELL_SIZE
        self.play_area_height = self.GRID_HEIGHT * self.CELL_SIZE
        self.play_area_x_offset = (self.SCREEN_WIDTH - self.play_area_width) // 2
        self.play_area_y_offset = (self.SCREEN_HEIGHT - self.play_area_height) // 2

        # Initialize state variables
        self.grid = None
        self.current_block = None
        self.score = 0
        self.game_timer = 0
        self.stack_height = 0
        self.placed_blocks = 0
        self.fall_speed = 0
        self.fall_timer = 0
        self.game_over = False
        self.particles = []
        self.drop_trail = []
        self.last_reward = 0
        
        self.reset()

        # Run validation check
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0.0
        self.game_timer = self.MAX_TIME
        self.stack_height = 0
        self.placed_blocks = 0
        self.fall_speed = 0.5  # seconds per grid cell
        self.fall_timer = 0.0
        self.game_over = False
        self.particles = []
        self.drop_trail = []
        self.last_reward = 0

        self._spawn_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        dt = 1.0 / self.FPS
        self.game_timer -= dt

        # --- Handle player input ---
        if self.current_block:
            # Action 3: Left
            if movement == 3:
                self._move_block(-1)
            # Action 4: Right
            elif movement == 4:
                self._move_block(1)
            # Action 2: Down (Instant Drop)
            elif movement == 2:
                reward += self._drop_block()
                # sound: instant_drop.wav
        
        # --- Update game logic ---
        if self.current_block:
            self.fall_timer += dt
            if self.fall_timer >= self.fall_speed:
                self.fall_timer = 0
                self.current_block['grid_y'] += 1
                
                # Check for landing
                if self._check_collision(self.current_block['grid_x'], self.current_block['grid_y']):
                    self.current_block['grid_y'] -= 1
                    reward += self._place_block()
                    # sound: block_land.wav

        # Update smooth y-position for rendering
        if self.current_block:
             self.current_block['y'] = self.current_block['grid_y'] * self.CELL_SIZE

        self._update_particles(dt)

        # --- Check for termination ---
        terminated = self.game_over or self.game_timer <= 0 or self.score >= self.WIN_SCORE
        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                reward += 100  # Win bonus
                # sound: win_game.wav
            elif self.game_timer <= 0:
                reward += 0 # No penalty, just end of game
                # sound: time_up.wav
            self.game_over = True
            self.current_block = None # No more falling blocks

        self.last_reward = reward
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_block(self):
        # 80% chance of red, 20% of blue
        block_type = 2 if self.np_random.random() < 0.2 else 1
        grid_x = self.GRID_WIDTH // 2
        grid_y = 0

        if self._check_collision(grid_x, grid_y):
            self.game_over = True
            self.last_reward = -50 # Game over penalty
            self.score += self.last_reward
            self.current_block = None
            # sound: game_over.wav
            return

        self.current_block = {
            'grid_x': grid_x,
            'grid_y': grid_y,
            'x': grid_x * self.CELL_SIZE,
            'y': grid_y * self.CELL_SIZE,
            'type': block_type
        }

    def _move_block(self, dx):
        if not self.current_block: return
        
        new_x = self.current_block['grid_x'] + dx
        if not self._check_collision(new_x, self.current_block['grid_y']):
            self.current_block['grid_x'] = new_x
            self.current_block['x'] = new_x * self.CELL_SIZE
            # sound: move_block.wav

    def _drop_block(self):
        if not self.current_block: return 0
        
        # Create a visual trail
        start_y = self.current_block['grid_y']
        
        final_y = self.current_block['grid_y']
        while not self._check_collision(self.current_block['grid_x'], final_y + 1):
            final_y += 1
        
        self.current_block['grid_y'] = final_y
        
        # Add trail particles
        for y in range(start_y, final_y):
             self.drop_trail.append({
                 'x': self.current_block['grid_x'] * self.CELL_SIZE + self.CELL_SIZE / 2,
                 'y': y * self.CELL_SIZE + self.CELL_SIZE / 2,
                 'alpha': 150,
                 'type': self.current_block['type']
             })

        return self._place_block()

    def _place_block(self):
        if not self.current_block: return 0
        
        x, y = self.current_block['grid_x'], self.current_block['grid_y']
        
        # Place block in grid
        self.grid[y, x] = self.current_block['type']
        
        # Calculate reward
        reward = 0.1  # Base reward for placing a block
        neighbors = self._count_neighbors(x, y)
        
        if neighbors >= 2:
            reward -= 0.2  # Penalty for safe placement
        
        # Bonus for risky blue block
        if self.current_block['type'] == 2 and neighbors < 2:
            reward += 0.5
            # sound: bonus_place.wav
        
        # Update game state
        self.placed_blocks += 1
        self.stack_height = self.GRID_HEIGHT - np.where(self.grid.any(axis=1))[0][0] if self.grid.any() else 0
        
        # Increase difficulty
        if self.placed_blocks > 0 and self.placed_blocks % 20 == 0:
            self.fall_speed = max(0.1, 0.5 - (self.placed_blocks // 20) * 0.05)

        # Create landing particles
        self._create_particles(x, y, self.current_block['type'])
        
        self.current_block = None
        self._spawn_block()
        
        return reward

    def _check_collision(self, grid_x, grid_y):
        if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
            return True
        if self.grid[grid_y, grid_x] != 0:
            return True
        return False

    def _count_neighbors(self, x, y):
        count = 0
        for dx, dy in [(0, 1), (1, 0), (-1, 0)]: # Check below, left, right
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                if self.grid[ny, nx] != 0:
                    count += 1
        return count

    def _create_particles(self, grid_x, grid_y, block_type):
        px = self.play_area_x_offset + grid_x * self.CELL_SIZE + self.CELL_SIZE / 2
        py = self.play_area_y_offset + grid_y * self.CELL_SIZE + self.CELL_SIZE / 2
        color = self.COLOR_RED if block_type == 1 else self.COLOR_BLUE
        
        for _ in range(10):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            self.particles.append({
                'x': px, 'y': py,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': 1.0, 'color': color
            })

    def _update_particles(self, dt):
        # Update landing particles
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= dt * 2
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        # Update drop trail particles
        for t in self.drop_trail:
            t['alpha'] -= 400 * dt
        self.drop_trail = [t for t in self.drop_trail if t['alpha'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw play area background
        play_area_rect = pygame.Rect(self.play_area_x_offset, self.play_area_y_offset, self.play_area_width, self.play_area_height)
        pygame.draw.rect(self.screen, self.COLOR_PLAY_AREA, play_area_rect)

        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.play_area_x_offset + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.play_area_y_offset), (x, self.play_area_y_offset + self.play_area_height))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.play_area_y_offset + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.play_area_x_offset, y), (self.play_area_x_offset + self.play_area_width, y))

        # Draw stacked blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] != 0:
                    color = self.COLOR_RED if self.grid[y, x] == 1 else self.COLOR_BLUE
                    rect = pygame.Rect(
                        self.play_area_x_offset + x * self.CELL_SIZE,
                        self.play_area_y_offset + y * self.CELL_SIZE,
                        self.CELL_SIZE, self.CELL_SIZE
                    )
                    pygame.draw.rect(self.screen, color, rect)
                    # Add a subtle inner highlight
                    pygame.draw.rect(self.screen, (255,255,255,20), rect.inflate(-4,-4))


        # Draw drop trail
        for t in self.drop_trail:
            color = self.COLOR_RED if t['type'] == 1 else self.COLOR_BLUE
            surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(surf, (*color, t['alpha']), (0, 0, self.CELL_SIZE, self.CELL_SIZE))
            self.screen.blit(surf, (self.play_area_x_offset + t['x'] - self.CELL_SIZE/2, self.play_area_y_offset + t['y'] - self.CELL_SIZE/2))

        # Draw falling block
        if self.current_block:
            block = self.current_block
            color = self.COLOR_RED if block['type'] == 1 else self.COLOR_BLUE
            glow_color = (*color, 50)
            
            px = self.play_area_x_offset + block['x']
            py = self.play_area_y_offset + block['y']
            
            # Glow effect
            glow_rect = pygame.Rect(px - 4, py - 4, self.CELL_SIZE + 8, self.CELL_SIZE + 8)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=5)
            self.screen.blit(glow_surf, glow_rect.topleft)
            
            # Main block
            rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            pygame.draw.rect(self.screen, (255,255,255,60), rect.inflate(-6,-6), border_radius=2)

        # Draw particles
        for p in self.particles:
            alpha = max(0, p['life'] * 255)
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), 2, (*p['color'], int(alpha)))
            
    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Timer
        timer_val = max(0, self.game_timer)
        timer_color = self.COLOR_TEXT
        if timer_val < 10: timer_color = self.COLOR_TIMER_CRIT
        elif timer_val < 20: timer_color = self.COLOR_TIMER_WARN
        timer_text = self.font_large.render(f"TIME: {math.ceil(timer_val):02}", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 20, 15))
        
        # Height
        height_text = self.font_medium.render(f"HEIGHT: {self.stack_height}/{self.GRID_HEIGHT}", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (self.SCREEN_WIDTH // 2 - height_text.get_width() // 2, self.SCREEN_HEIGHT - 35))

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "GOAL REACHED!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_large.render(msg, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.placed_blocks, # More informative than raw steps
            "timer": self.game_timer,
            "height": self.stack_height,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Block Stacker")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    terminated = False
    
    while running:
        # --- Human Controls ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        elif keys[pygame.K_DOWN]:
            movement = 2
            
        action = [movement, 0, 0] # Space and shift are not used

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    terminated = False

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)

    env.close()