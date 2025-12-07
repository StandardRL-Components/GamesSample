import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to move the falling piece. Press Space to drop it instantly."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Stack falling blocks to build a tower. Clear full rows to score points and prevent the tower from reaching the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.CELL_SIZE = 16
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.CELL_SIZE # 40
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.CELL_SIZE # 25
        
        self.MAX_STEPS = 5000
        self.WIN_SCORE = 500 # A challenging but achievable score goal

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (35, 40, 60)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_BOX = (50, 60, 80)
        self.TILE_COLORS = [
            (0, 0, 0), # 0 is empty
            (0, 220, 220), # Cyan
            (220, 220, 0), # Yellow
            (220, 0, 220), # Magenta
            (0, 220, 0),   # Green
            (220, 120, 0), # Orange
        ]

        # --- Game Objects ---
        self.tile_shapes = [
            [(0, 0)], # 1x1
            [(-1, 0), (0, 0)], # 2x1
            [(-1, 0), (0, 0), (1, 0)], # 3x1
            [(-2, 0), (-1, 0), (0, 0), (1, 0)], # 4x1
            [(0, 0), (1, 0), (0, 1), (1, 1)] # 2x2 square
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- State Variables (initialized in reset) ---
        self.grid = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.np_random = None
        
        self.falling_tile = None
        self.next_tile = None
        self.fall_speed = None
        self.fall_progress = None
        
        self.particles = []
        
        self.move_cooldown = 0
        self.MOVE_COOLDOWN_FRAMES = 3 

        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.initial_fall_speed = 1.0 # cells per second
        self.fall_speed = self.initial_fall_speed
        
        self.next_tile = None
        self._spawn_new_tile()
        self._spawn_new_tile() # Once for next_tile, once for falling_tile

        self.particles = []
        self.move_cooldown = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        terminated = False
        self.steps += 1
        
        # --- Handle Action ---
        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        
        self.move_cooldown = max(0, self.move_cooldown - 1)

        if self.move_cooldown == 0:
            if movement == 3: # Left
                if not self._check_collision(-1, 0):
                    self.falling_tile['x'] -= 1
                    self.move_cooldown = self.MOVE_COOLDOWN_FRAMES
            elif movement == 4: # Right
                if not self._check_collision(1, 0):
                    self.falling_tile['x'] += 1
                    self.move_cooldown = self.MOVE_COOLDOWN_FRAMES

        if space_pressed:
            # Fast drop
            while not self._check_collision(0, 1):
                self.falling_tile['y'] += 1
            placement_reward, game_over_flag = self._place_tile()
            reward += placement_reward
            if game_over_flag:
                terminated = True
        else:
            # --- Normal Gravity ---
            self.fall_progress += self.fall_speed / 30.0 
            
            if self.fall_progress >= 1.0:
                moves = int(self.fall_progress)
                self.fall_progress -= moves
                for _ in range(moves):
                    if self.falling_tile and not self._check_collision(0, 1):
                        self.falling_tile['y'] += 1
                    elif self.falling_tile:
                        placement_reward, game_over_flag = self._place_tile()
                        reward += placement_reward
                        if game_over_flag:
                            terminated = True
                        break 
                
        self._update_particles()
        
        if self.steps > 0 and self.steps % 500 == 0:
            self.fall_speed += 0.2

        if not terminated:
            rows_cleared = self._check_and_clear_rows()
            if rows_cleared > 0:
                # Sound: Row clear!
                clear_score = (10 * rows_cleared) * rows_cleared
                reward += clear_score
                self.score += clear_score

        if self.steps >= self.MAX_STEPS:
            terminated = True

        if self.score >= self.WIN_SCORE and not terminated:
            reward += 100
            terminated = True
        
        if terminated and not self.game_over:
            self.game_over = True
        elif terminated and self.game_over:
            # Penalize for taking actions after game is over
            reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _spawn_new_tile(self):
        self.falling_tile = self.next_tile
        
        shape_idx = self.np_random.integers(0, len(self.tile_shapes))
        color_idx = self.np_random.integers(1, len(self.TILE_COLORS))
        
        self.next_tile = {
            'shape': self.tile_shapes[shape_idx],
            'color_idx': color_idx,
            'x': self.GRID_WIDTH // 2,
            'y': 0
        }

        if self.falling_tile is not None:
            self.fall_progress = 0.0
            if self._check_collision(0, 0):
                self.game_over = True

    def _check_collision(self, dx, dy):
        if self.falling_tile is None:
            return False
        
        for block_x, block_y in self.falling_tile['shape']:
            x = self.falling_tile['x'] + block_x + dx
            y = self.falling_tile['y'] + block_y + dy
            
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return True
            if y >= 0 and self.grid[x, y] != 0:
                return True
        return False

    def _place_tile(self):
        if self.falling_tile is None:
            return 0, False
            
        is_supported = False
        for block_x, block_y in self.falling_tile['shape']:
            x = self.falling_tile['x'] + block_x
            y = self.falling_tile['y'] + block_y
            
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                self.game_over = True
                return -10, True

            is_at_bottom = (y == self.GRID_HEIGHT - 1)
            is_on_tile = (y + 1 < self.GRID_HEIGHT and self.grid[x, y + 1] != 0)
            if is_at_bottom or is_on_tile:
                is_supported = True

            self.grid[x, y] = self.falling_tile['color_idx']
        
        if not is_supported:
            self.game_over = True
            return -10, True
        
        self._spawn_new_tile()
        
        if self.game_over:
            return -10, True
        
        self.score += 1
        return 0.1, False

    def _check_and_clear_rows(self):
        rows_cleared = 0
        y = self.GRID_HEIGHT - 1
        while y >= 0:
            if np.all(self.grid[:, y] != 0):
                rows_cleared += 1
                for x in range(self.GRID_WIDTH):
                    self._create_particles(x, y, self.grid[x,y])
                
                self.grid[:, 1:y+1] = self.grid[:, 0:y]
                self.grid[:, 0] = 0
            else:
                y -= 1
        return rows_cleared

    def _create_particles(self, grid_x, grid_y, color_idx):
        px, py = (grid_x + 0.5) * self.CELL_SIZE, (grid_y + 0.5) * self.CELL_SIZE
        color = self.TILE_COLORS[color_idx]
        for _ in range(5):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _render_block(self, surface, x, y, color_idx, is_ghost=False):
        color = self.TILE_COLORS[color_idx]
        px, py = x * self.CELL_SIZE, y * self.CELL_SIZE
        
        if is_ghost:
            alpha_color = (*color, 100)
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(s, alpha_color, (1, 1, self.CELL_SIZE-2, self.CELL_SIZE-2))
            surface.blit(s, (px, py))
            return

        # FIX: A generator is exhausted after one use. Convert to a list/tuple first.
        dark_color = [max(0, c - 50) for c in color]
        light_color = [min(255, c + 50) for c in color]

        pygame.draw.rect(surface, color, (px + 1, py + 1, self.CELL_SIZE - 2, self.CELL_SIZE - 2))
        pygame.draw.line(surface, dark_color, (px + self.CELL_SIZE - 1, py), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1), 2)
        pygame.draw.line(surface, dark_color, (px, py + self.CELL_SIZE - 1), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1), 2)
        pygame.draw.line(surface, light_color, (px, py), (px + self.CELL_SIZE - 1, py), 2)
        pygame.draw.line(surface, light_color, (px, py), (px, py + self.CELL_SIZE - 1), 2)

    def _render_game(self):
        for x in range(self.GRID_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x * self.CELL_SIZE, 0), (x * self.CELL_SIZE, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y * self.CELL_SIZE), (self.SCREEN_WIDTH, y * self.CELL_SIZE))

        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] != 0:
                    self._render_block(self.screen, x, y, int(self.grid[x, y]))
        
        if self.falling_tile and not self.game_over:
            original_y = self.falling_tile['y']
            
            # Find ghost piece position
            ghost_y = original_y
            temp_y = original_y
            while True:
                self.falling_tile['y'] += 1
                if self._check_collision(0, 0):
                    self.falling_tile['y'] -= 1
                    ghost_y_final = self.falling_tile['y']
                    break
            self.falling_tile['y'] = original_y
            
            # Render ghost piece
            for block_x, block_y in self.falling_tile['shape']:
                self._render_block(self.screen, self.falling_tile['x'] + block_x, ghost_y_final + block_y, self.falling_tile['color_idx'], is_ghost=True)

            # Render falling piece
            for block_x, block_y in self.falling_tile['shape']:
                self._render_block(self.screen, self.falling_tile['x'] + block_x, self.falling_tile['y'] + block_y, self.falling_tile['color_idx'])

        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            size = int(self.CELL_SIZE/2 * (p['life'] / p['max_life']))
            
            if size > 0:
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)))


    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        preview_box_rect = pygame.Rect(self.SCREEN_WIDTH - 130, 10, 120, 80)
        pygame.draw.rect(self.screen, self.COLOR_UI_BOX, preview_box_rect, 0, 8)
        pygame.draw.rect(self.screen, self.COLOR_GRID, preview_box_rect, 2, 8)
        
        next_text = self.font_small.render("NEXT", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (preview_box_rect.centerx - next_text.get_width() // 2, preview_box_rect.top + 5))

        if self.next_tile:
            shape = self.next_tile['shape']
            min_x = min(p[0] for p in shape); max_x = max(p[0] for p in shape)
            min_y = min(p[1] for p in shape); max_y = max(p[1] for p in shape)
            shape_width = (max_x - min_x + 1) * self.CELL_SIZE
            shape_height = (max_y - min_y + 1) * self.CELL_SIZE
            
            offset_x = preview_box_rect.centerx - shape_width // 2
            offset_y = preview_box_rect.centery - shape_height // 2 + 10

            for block_x, block_y in self.next_tile['shape']:
                px = offset_x - min_x * self.CELL_SIZE + block_x * self.CELL_SIZE
                py = offset_y - min_y * self.CELL_SIZE + block_y * self.CELL_SIZE
                
                block_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                self._render_block(block_surf, 0, 0, self.next_tile['color_idx'])
                self.screen.blit(block_surf, (px, py))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "GAME OVER"
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"

            game_over_text = self.font_main.render(msg, True, (255, 255, 255))
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    import sys

    # Unset the dummy video driver if we want to visualize
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # To test the environment without visualization
    # env.validate_implementation()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tile Stacker")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    obs, info = env.reset()

    movement = 0
    space_pressed = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_SPACE: space_pressed = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: space_pressed = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        else: movement = 0
            
        action = [movement, space_pressed, 0]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))

        pygame.display.flip()

        if terminated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        if space_pressed == 1: space_pressed = 0
        clock.tick(30)

    env.close()
    sys.exit()