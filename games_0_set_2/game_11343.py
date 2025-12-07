import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:02:36.652114
# Source Brief: brief_01343.md
# Brief Index: 1343
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a block-stacking puzzle game.
    The goal is to stack falling blocks of the same color in vertical columns.
    When a block lands on another of the same color, all blocks of that color
    in the column are cleared, awarding points. The game ends if the stack
    reaches the top or the player scores 1000 points.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A block-stacking puzzle game. Stack falling blocks in columns and clear them "
        "by matching colors to score points."
    )
    user_guide = (
        "Controls: ←→ to move the block, ↓ for a soft drop, and space for a hard drop."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAYFIELD_COLS = 10
    PLAYFIELD_ROWS = 16
    BLOCK_SIZE = 24
    PLAYFIELD_WIDTH = PLAYFIELD_COLS * BLOCK_SIZE
    PLAYFIELD_HEIGHT = PLAYFIELD_ROWS * BLOCK_SIZE
    PLAYFIELD_X_OFFSET = (SCREEN_WIDTH - PLAYFIELD_WIDTH) // 2
    PLAYFIELD_Y_OFFSET = SCREEN_HEIGHT - PLAYFIELD_HEIGHT

    COLOR_BG = (15, 15, 25)
    COLOR_BORDER = (80, 80, 100)
    COLOR_GRID = (30, 30, 45)
    COLOR_TEXT = (220, 220, 255)
    BLOCK_COLORS = [
        (255, 50, 50),   # Red
        (50, 255, 50),   # Green
        (50, 150, 255),  # Blue
        (255, 255, 50),  # Yellow
        (200, 50, 255),  # Purple
    ]

    MAX_STEPS = 2000
    WIN_SCORE = 1000

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # State variables are initialized in reset()
        self.playfield = None
        self.falling_block = None
        self.next_block_color_idx = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.gravity = None
        self.prev_space_held = None
        self.particles = None
        self.clearing_effects = None
        self.last_score_milestone = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.playfield = np.zeros((self.PLAYFIELD_COLS, self.PLAYFIELD_ROWS), dtype=int)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.gravity = 1.0
        self.prev_space_held = False
        self.particles = []
        self.clearing_effects = []
        self.last_score_milestone = 0

        self.next_block_color_idx = self.np_random.integers(1, len(self.BLOCK_COLORS) + 1)
        self._spawn_new_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        step_reward = 0

        self._handle_input(movement)
        self._update_particles_and_effects()

        # Hard drop (instant placement on space press)
        if space_held and not self.prev_space_held:
            # Sound effect: Hard drop
            while not self._check_collision(self.falling_block['grid_x'], self.falling_block['y'] + 1):
                self.falling_block['y'] += 1
            step_reward += self._place_block()
        else:
            # Normal gravity / soft drop
            soft_drop_multiplier = 5 if movement == 2 else 1
            self.falling_block['y'] += self.gravity * soft_drop_multiplier / self.BLOCK_SIZE

            if self._check_collision(self.falling_block['grid_x'], self.falling_block['y']):
                step_reward += self._place_block()
        
        self.prev_space_held = space_held
        
        # Update score-based difficulty
        current_milestone = self.score // 200
        if current_milestone > self.last_score_milestone:
            self.gravity *= 1.05
            self.last_score_milestone = current_milestone
            # Sound effect: Level up

        self.steps += 1
        
        if self.score >= self.WIN_SCORE and not self.game_over:
            self.game_over = True
            step_reward += 100
            # Sound effect: Win
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = False
        if terminated and not self.game_over: # Lost due to timeout or topping out
            step_reward -= 100

        return self._get_observation(), step_reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement):
        if not self.falling_block: return

        new_grid_x = self.falling_block['grid_x']
        if movement == 3: # Left
            new_grid_x -= 1
        elif movement == 4: # Right
            new_grid_x += 1
        
        if 0 <= new_grid_x < self.PLAYFIELD_COLS:
            if not self._check_collision(new_grid_x, self.falling_block['y']):
                if self.falling_block['grid_x'] != new_grid_x:
                    # Sound effect: Move
                    self.falling_block['grid_x'] = new_grid_x

    def _place_block(self):
        if not self.falling_block: return 0
        
        land_y_grid = math.floor(self.falling_block['y']) - 1
        land_x_grid = self.falling_block['grid_x']
        
        if land_y_grid < 0:
            self.game_over = True
            # Sound effect: Lose
            self.falling_block = None
            return -100
        
        self.playfield[land_x_grid, land_y_grid] = self.falling_block['color_idx']
        
        cleared_count = self._check_and_clear_columns(land_x_grid, land_y_grid)
        
        self._spawn_new_block()
        
        reward = 0.1 # Reward for placing a block
        if cleared_count > 0:
            reward += cleared_count * 10
            self.score += cleared_count
            # Sound effect: Clear
        
        return reward

    def _check_and_clear_columns(self, x, y):
        color_to_match = self.playfield[x, y]
        
        if y + 1 >= self.PLAYFIELD_ROWS or self.playfield[x, y + 1] != color_to_match:
            return 0

        cleared_blocks = []
        for row in range(self.PLAYFIELD_ROWS):
            if self.playfield[x, row] == color_to_match:
                cleared_blocks.append((x, row))
        
        if not cleared_blocks: return 0

        for cx, cy in cleared_blocks:
            self.playfield[cx, cy] = 0
            self.clearing_effects.append({'x': cx, 'y': cy, 'color': color_to_match, 'timer': 10})
            self._create_particles(cx, cy, color_to_match)
        
        new_col = np.zeros(self.PLAYFIELD_ROWS, dtype=int)
        write_idx = self.PLAYFIELD_ROWS - 1
        for read_idx in range(self.PLAYFIELD_ROWS - 1, -1, -1):
            if self.playfield[x, read_idx] != 0:
                new_col[write_idx] = self.playfield[x, read_idx]
                write_idx -= 1
        self.playfield[x, :] = new_col

        return len(cleared_blocks)

    def _spawn_new_block(self):
        spawn_x = self.PLAYFIELD_COLS // 2
        
        if self.playfield[spawn_x, 0] != 0:
            self.game_over = True
            # Sound effect: Lose
            self.falling_block = None
            return

        self.falling_block = {
            'grid_x': spawn_x,
            'y': 0.0,
            'x_visual': float(spawn_x * self.BLOCK_SIZE),
            'color_idx': self.next_block_color_idx
        }
        self.next_block_color_idx = self.np_random.integers(1, len(self.BLOCK_COLORS) + 1)
        # Sound effect: Spawn
        
    def _check_collision(self, grid_x, y_pos):
        if y_pos >= self.PLAYFIELD_ROWS:
            return True
        
        grid_y = math.floor(y_pos)
        if grid_y < 0: return False
        if grid_y < self.PLAYFIELD_ROWS and self.playfield[grid_x, grid_y] != 0:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "gravity": round(self.gravity, 2)}

    def _render_block(self, surface, x_pixel, y_pixel, color_idx, glow=False):
        if color_idx == 0: return
        base_color = self.BLOCK_COLORS[color_idx - 1]
        shadow_color = (max(0, base_color[0]-50), max(0, base_color[1]-50), max(0, base_color[2]-50))
        highlight_color = (min(255, base_color[0]+50), min(255, base_color[1]+50), min(255, base_color[2]+50))
        
        if glow:
            glow_size = int(self.BLOCK_SIZE * 2)
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*base_color, 60), (glow_size//2, glow_size//2), glow_size//2)
            surface.blit(glow_surf, (x_pixel - self.BLOCK_SIZE//2, y_pixel - self.BLOCK_SIZE//2), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(surface, shadow_color, (x_pixel, y_pixel, self.BLOCK_SIZE, self.BLOCK_SIZE))
        pygame.draw.rect(surface, base_color, (x_pixel + 2, y_pixel + 2, self.BLOCK_SIZE - 4, self.BLOCK_SIZE - 4))
        pygame.draw.line(surface, highlight_color, (x_pixel+2, y_pixel+2), (x_pixel + self.BLOCK_SIZE-3, y_pixel+2), 2)
        pygame.draw.line(surface, highlight_color, (x_pixel+2, y_pixel+2), (x_pixel+2, y_pixel + self.BLOCK_SIZE-3), 2)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (self.PLAYFIELD_X_OFFSET - 2, self.PLAYFIELD_Y_OFFSET - 2, self.PLAYFIELD_WIDTH + 4, self.PLAYFIELD_HEIGHT + 4), 2)
        for i in range(1, self.PLAYFIELD_COLS):
            x = self.PLAYFIELD_X_OFFSET + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.PLAYFIELD_Y_OFFSET), (x, self.SCREEN_HEIGHT))
        for i in range(1, self.PLAYFIELD_ROWS):
            y = self.PLAYFIELD_Y_OFFSET + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.PLAYFIELD_X_OFFSET, y), (self.PLAYFIELD_X_OFFSET + self.PLAYFIELD_WIDTH, y))

        for x in range(self.PLAYFIELD_COLS):
            for y in range(self.PLAYFIELD_ROWS):
                if self.playfield[x, y] != 0:
                    px = self.PLAYFIELD_X_OFFSET + x * self.BLOCK_SIZE
                    py = self.PLAYFIELD_Y_OFFSET + y * self.BLOCK_SIZE
                    self._render_block(self.screen, px, py, self.playfield[x, y])
        
        for effect in self.clearing_effects:
            px = self.PLAYFIELD_X_OFFSET + effect['x'] * self.BLOCK_SIZE
            py = self.PLAYFIELD_Y_OFFSET + effect['y'] * self.BLOCK_SIZE
            alpha = int(255 * (effect['timer'] / 10))
            s = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
            s.fill((255, 255, 255, alpha))
            self.screen.blit(s, (px, py), special_flags=pygame.BLEND_RGBA_ADD)

        if self.falling_block:
            target_x = self.PLAYFIELD_X_OFFSET + self.falling_block['grid_x'] * self.BLOCK_SIZE
            self.falling_block['x_visual'] = 0.6 * self.falling_block['x_visual'] + 0.4 * target_x
            px = int(self.falling_block['x_visual'])
            py = int(self.PLAYFIELD_Y_OFFSET + self.falling_block['y'] * self.BLOCK_SIZE)
            self._render_block(self.screen, px, py, self.falling_block['color_idx'], glow=True)

            ghost_y = self.falling_block['y']
            while not self._check_collision(self.falling_block['grid_x'], ghost_y + 1):
                ghost_y += 1
            ghost_py = int(self.PLAYFIELD_Y_OFFSET + ghost_y * self.BLOCK_SIZE)
            rect = (self.PLAYFIELD_X_OFFSET + self.falling_block['grid_x'] * self.BLOCK_SIZE, ghost_py, self.BLOCK_SIZE, self.BLOCK_SIZE)
            color = self.BLOCK_COLORS[self.falling_block['color_idx']-1]
            pygame.gfxdraw.rectangle(self.screen, rect, (*color, 80))

        for p in self.particles:
            color = self.BLOCK_COLORS[p['color']-1]
            pygame.draw.circle(self.screen, color, (int(p['x']), int(p['y'])), int(p['life'] / 5))

    def _render_ui(self):
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        next_text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.SCREEN_WIDTH - 120, 20))
        preview_box_x = self.SCREEN_WIDTH - 125
        preview_box_y = 45
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (preview_box_x, preview_box_y, 70, 70), 2)
        if self.next_block_color_idx:
            self._render_block(self.screen, preview_box_x + (70 - self.BLOCK_SIZE)//2, preview_box_y + (70 - self.BLOCK_SIZE)//2, self.next_block_color_idx)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        status_text = "YOU WON!" if self.score >= self.WIN_SCORE else "GAME OVER"
        text_surf = self.font_large.render(status_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    def _create_particles(self, grid_x, grid_y, color_idx):
        center_x = self.PLAYFIELD_X_OFFSET + grid_x * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        center_y = self.PLAYFIELD_Y_OFFSET + grid_y * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'x': center_x, 'y': center_y,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': random.randint(15, 30),
                'color': color_idx
            })

    def _update_particles_and_effects(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
        
        for effect in self.clearing_effects[:]:
            effect['timer'] -= 1
            if effect['timer'] <= 0:
                self.clearing_effects.remove(effect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block is for human play and visualization.
    # It is not part of the required environment implementation.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a visible driver
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Block Stacker")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    action = [0, 0, 0]
    
    while not done:
        # --- Human Controls ---
        # This mapping is for playability, an agent would learn the raw actions.
        # Action is reset each frame to handle key presses correctly.
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action[1] = 1 # Press space
                if event.key == pygame.K_ESCAPE:
                    done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        elif keys[pygame.K_DOWN]:
            action[0] = 2 # Soft drop
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Gravity: {info['gravity']}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Control game speed

    env.close()
    print("Game Over!")
    print(f"Final Score: {info['score']}")