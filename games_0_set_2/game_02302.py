
# Generated: 2025-08-27T19:57:28.156277
# Source Brief: brief_02302.md
# Brief Index: 2302

        
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
        "Controls: ←→ to move the falling block. ↓ to soft drop. "
        "Space to hard drop. Shift to swap with the next block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match falling colored blocks to clear the board in this fast-paced, top-down puzzle game. "
        "Get 3 or more of the same color in a row or column to clear them."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 10
        self.BLOCK_SIZE = 30
        self.BOARD_X = (self.WIDTH - self.GRID_WIDTH * self.BLOCK_SIZE) // 2
        self.BOARD_Y = (self.HEIGHT - self.GRID_HEIGHT * self.BLOCK_SIZE) + 20 # Move board down a bit

        # --- Colors ---
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_PANEL = (15, 25, 35)
        self.BLOCK_COLORS = [
            (0, 0, 0),          # 0: Empty
            (255, 87, 87),      # 1: Red
            (87, 255, 87),      # 2: Green
            (87, 150, 255),     # 3: Blue
            (255, 255, 87),     # 4: Yellow
            (170, 87, 255),     # 5: Purple
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        # --- Game State ---
        self.np_random = None
        self.grid = None
        self.falling_block = None
        self.next_block_color = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.fall_speed = 0.0
        self.fall_progress = 0.0
        
        self.match_animation_queue = deque()
        self.particles = []
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.MAX_STEPS = 2000

        self.reset()
        
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        for r in range(self.GRID_HEIGHT - 4, self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.np_random.random() < 0.4:
                     self.grid[r][c] = self.np_random.integers(1, len(self.BLOCK_COLORS))

        self.falling_block = None
        self.next_block_color = self._get_new_block_color()
        self._spawn_block()

        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.fall_speed = 1.5
        self.fall_progress = 0.0
        
        self.match_animation_queue.clear()
        self.particles.clear()
        
        self.prev_space_held = False
        self.prev_shift_held = False

        while self._find_and_clear_matches(initial_clear=True):
            pass

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        step_reward = 0

        if self.match_animation_queue:
            done_animating = self._update_match_animations()
            if done_animating:
                reward, _ = self._resolve_cleared_blocks()
                step_reward += reward
                self._apply_gravity_to_grid()
        elif self.falling_block:
            self._handle_player_input(movement, space_held, shift_held)
            if self.falling_block: # Can be set to None by hard drop
                self._update_falling_block()
        else:
            if not self._find_and_clear_matches():
                if np.all(self.grid == 0):
                    self.win = True
                    self.game_over = True
                else:
                    self._spawn_block()

        if self.steps > 0 and self.steps % 200 == 0:
            self.fall_speed = min(5.0, self.fall_speed + 0.1)

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated:
            if self.win:
                step_reward += 100
            elif self.game_over:
                step_reward -= 100

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return self._get_observation(), step_reward, terminated, False, self._get_info()

    def _handle_player_input(self, movement, space_held, shift_held):
        if not self.falling_block: return

        if space_held and not self.prev_space_held:
            self._hard_drop()
            # // Sound: Hard Drop
            return 
        
        if shift_held and not self.prev_shift_held:
            self._swap_block()
            # // Sound: Swap

        if movement == 3: # Left
            self.falling_block['x'] = max(0, self.falling_block['x'] - 1)
        elif movement == 4: # Right
            self.falling_block['x'] = min(self.GRID_WIDTH - 1, self.falling_block['x'] + 1)
        elif movement == 2: # Down (soft drop)
            self.fall_progress += self.fall_speed * 1.5

    def _update_falling_block(self):
        self.fall_progress += self.fall_speed
        
        target_y = self.falling_block['y'] + 1
        landed = (target_y >= self.GRID_HEIGHT or 
                  (0 <= self.falling_block['x'] < self.GRID_WIDTH and self.grid[target_y][self.falling_block['x']] != 0))

        if self.fall_progress >= self.BLOCK_SIZE:
            self.fall_progress = 0
            if not landed:
                self.falling_block['y'] += 1
            else:
                self._land_block()
        
        self.falling_block['pixel_y'] = self.falling_block['y'] * self.BLOCK_SIZE + self.fall_progress

    def _land_block(self):
        r, c = self.falling_block['y'], self.falling_block['x']
        if 0 <= r < self.GRID_HEIGHT and 0 <= c < self.GRID_WIDTH:
            self.grid[r][c] = self.falling_block['color']
            # // Sound: Block Land
        
        self.falling_block = None
        self._find_and_clear_matches()

    def _hard_drop(self):
        c = self.falling_block['x']
        
        lowest_r = self.GRID_HEIGHT - 1
        while lowest_r >= 0 and self.grid[lowest_r][c] != 0:
            lowest_r -= 1
        
        if lowest_r >= self.falling_block['y']:
            self.falling_block['y'] = lowest_r
            self._land_block()
        else: # Column is full or something is wrong
            self._land_block()

    def _swap_block(self):
        current_color = self.falling_block['color']
        self.falling_block['color'] = self.next_block_color
        self.next_block_color = current_color

    def _find_and_clear_matches(self, initial_clear=False):
        to_clear = set()
        
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                color = self.grid[r][c]
                if color != 0 and color == self.grid[r][c+1] == self.grid[r][c+2]:
                    match_len = 2
                    while c + match_len < self.GRID_WIDTH and self.grid[r][c+match_len] == color:
                        match_len += 1
                    for i in range(match_len): to_clear.add((r, c+i))

        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                color = self.grid[r][c]
                if color != 0 and color == self.grid[r+1][c] == self.grid[r+2][c]:
                    match_len = 2
                    while r + match_len < self.GRID_HEIGHT and self.grid[r+match_len][c] == color:
                        match_len += 1
                    for i in range(match_len): to_clear.add((r+i, c))

        if not to_clear: return False

        if initial_clear:
            for r, c in to_clear: self.grid[r, c] = 0
            self._apply_gravity_to_grid()
        else:
            # // Sound: Match Found
            for r, c in to_clear: self.match_animation_queue.append({'pos': (r, c), 'timer': 15})
        
        return True

    def _update_match_animations(self):
        done = True
        for item in self.match_animation_queue:
            item['timer'] -= 1
            if item['timer'] > 0: done = False
        return done
    
    def _resolve_cleared_blocks(self):
        cleared_count = len(self.match_animation_queue)
        for item in self.match_animation_queue:
            r, c = item['pos']
            if self.grid[r][c] != 0:
                self._create_particles(c, r, self.grid[r][c])
                self.grid[r][c] = 0
                # // Sound: Block Clear
        
        self.match_animation_queue.clear()
        
        reward = cleared_count * 0.1
        if cleared_count == 3: reward += 1
        elif cleared_count == 4: reward += 2
        elif cleared_count >= 5: reward += 5
        
        self.score += cleared_count
        return reward, cleared_count

    def _apply_gravity_to_grid(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r][c] != 0:
                    if r != empty_row:
                        self.grid[empty_row][c] = self.grid[r][c]
                        self.grid[r][c] = 0
                    empty_row -= 1
    
    def _spawn_block(self):
        spawn_x = self.GRID_WIDTH // 2
        if self.grid[0][spawn_x] != 0:
            self.game_over = True
            return

        self.falling_block = {
            'x': spawn_x, 'y': 0, 'pixel_y': -self.BLOCK_SIZE, 'color': self.next_block_color
        }
        self.next_block_color = self._get_new_block_color()
        self.fall_progress = 0

    def _get_new_block_color(self):
        return self.np_random.integers(1, len(self.BLOCK_COLORS))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for r in range(self.GRID_HEIGHT + 1):
            y = self.BOARD_Y + r * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.BOARD_X, y), (self.BOARD_X + self.GRID_WIDTH * self.BLOCK_SIZE, y))
        for c in range(self.GRID_WIDTH + 1):
            x = self.BOARD_X + c * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.BOARD_Y), (x, self.BOARD_Y + self.GRID_HEIGHT * self.BLOCK_SIZE))

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] != 0: self._draw_block(c, r, self.grid[r][c])

        if self.falling_block:
            rect = pygame.Rect(
                self.BOARD_X + self.falling_block['x'] * self.BLOCK_SIZE,
                self.BOARD_Y + self.falling_block['pixel_y'],
                self.BLOCK_SIZE, self.BLOCK_SIZE
            )
            self._draw_block_rect(rect, self.falling_block['color'], glow=True)

        for item in self.match_animation_queue:
            r, c = item['pos']
            alpha = 255 * (math.sin(item['timer'] * math.pi / 5) * 0.5 + 0.5)
            flash_surface = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surface, (self.BOARD_X + c * self.BLOCK_SIZE, self.BOARD_Y + r * self.BLOCK_SIZE))

        self._update_and_draw_particles()

    def _draw_block_rect(self, rect, color_id, glow=False):
        main_color = self.BLOCK_COLORS[color_id]
        light_color = tuple(min(255, val + 50) for val in main_color)
        dark_color = tuple(max(0, val - 50) for val in main_color)
        
        pygame.draw.rect(self.screen, main_color, rect)
        
        pygame.draw.line(self.screen, light_color, rect.topleft, rect.topright, 2)
        pygame.draw.line(self.screen, light_color, rect.topleft, rect.bottomleft, 2)
        pygame.draw.line(self.screen, dark_color, rect.bottomleft, rect.bottomright, 2)
        pygame.draw.line(self.screen, dark_color, rect.topright, rect.bottomright, 2)

        if glow:
            glow_surface = pygame.Surface((self.BLOCK_SIZE + 8, self.BLOCK_SIZE + 8), pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (*light_color, 60), glow_surface.get_rect(), border_radius=5)
            self.screen.blit(glow_surface, (rect.x - 4, rect.y - 4), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_block(self, c, r, color_id):
        rect = pygame.Rect(self.BOARD_X + c * self.BLOCK_SIZE, self.BOARD_Y + r * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
        self._draw_block_rect(rect, color_id)

    def _render_ui(self):
        panel_rect = pygame.Rect(0, 0, self.WIDTH, self.BOARD_Y - 20)
        pygame.draw.rect(self.screen, self.COLOR_PANEL, panel_rect)
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, panel_rect.bottom), (self.WIDTH, panel_rect.bottom))

        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        next_text = self.font_main.render("Next:", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.WIDTH - 150, 20))
        next_block_rect = pygame.Rect(self.WIDTH - 70, 15, self.BLOCK_SIZE, self.BLOCK_SIZE)
        self._draw_block_rect(next_block_rect, self.next_block_color)

        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (150, 255, 150) if self.win else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            
            bg_surf = pygame.Surface((text_rect.width + 20, text_rect.height + 20), pygame.SRCALPHA)
            pygame.draw.rect(bg_surf, (0, 0, 0, 180), bg_surf.get_rect(), border_radius=10)
            self.screen.blit(bg_surf, (text_rect.x-10, text_rect.y-10))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, c, r, color_id):
        px = self.BOARD_X + c * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        py = self.BOARD_Y + r * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        color = self.BLOCK_COLORS[color_id]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': [px, py], 'vel': vel, 'life': self.np_random.integers(20, 40),
                'size': self.np_random.uniform(2, 6), 'color': color
            })
    
    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]; p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1; p['life'] -= 1; p['size'] -= 0.1
            if p['life'] > 0 and p['size'] > 0:
                pygame.draw.circle(self.screen, p['color'], p['pos'], p['size'])
                active_particles.append(p)
        self.particles = active_particles

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Fall")
    clock = pygame.time.Clock()

    running = True
    while running:
        mov, space_pressed, shift_pressed = 0, False, False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: space_pressed = True
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_pressed = True
                if event.key == pygame.K_r: obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_DOWN]: mov = 2
        if keys[pygame.K_LEFT]: mov = 3
        if keys[pygame.K_RIGHT]: mov = 4

        current_action = np.array([mov, 1 if space_pressed else 0, 1 if shift_pressed else 0])
        
        obs, reward, terminated, truncated, info = env.step(current_action)

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']} in {info['steps']} steps.")
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(30)

    env.close()