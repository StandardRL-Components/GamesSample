import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:44:29.349446
# Source Brief: brief_00629.md
# Brief Index: 629
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player stacks blocks against the clock.

    The goal is to achieve a score of 100 by stacking blocks of different
    sizes and properties in a 3-column grid. The stack can become unstable
    and collapse, ending the game. The game also ends if the 15-second
    timer runs out. Visual polish and satisfying game feel are prioritized.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Stack blocks of different sizes and properties against the clock to reach a target score. Unstable structures will collapse, ending the game."
    user_guide = "Use ← and → arrow keys to move the block. Press space to place the block and shift to rotate it."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS = 3
        self.GRID_ROWS = 6 # Visual height limit for placement
        self.CELL_SIZE = 80
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_COLS * self.CELL_SIZE) // 2
        self.GRID_Y_OFFSET = self.HEIGHT - 40 # Floor Y position
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 15
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS
        self.WIN_SCORE = 100

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TIMER_WARN = (255, 100, 100)

        # --- Block Properties ---
        self.BLOCK_DEFINITIONS = {
            'small': {'dims': (1, 1), 'color': (220, 50, 50), 'property': 'sticky', 'score': 10},
            'medium': {'dims': (2, 1), 'color': (50, 220, 50), 'property': 'slippery', 'score': 20},
            'large': {'dims': (3, 1), 'color': (50, 100, 220), 'property': 'normal', 'score': 30}
        }

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
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.end_timer = 0
        self.cursor_col = 0
        self.placed_blocks = []
        self.grid_heights = []
        self.next_block = {}
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.screen_shake = 0

        # self.reset() is called by the environment wrapper
        # self.validate_implementation() # This can be removed for production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.end_timer = self.FPS * 2 # seconds to show end screen
        self.cursor_col = 1
        self.placed_blocks = []
        self.grid_heights = [0] * self.GRID_COLS
        self._generate_next_block()
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.screen_shake = 0
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            self.end_timer -= 1
            # The episode is already terminated, but we show the end screen for a bit.
            # The agent should see terminated=True and reset.
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Handle Input ---
        reward += self._handle_input(action)

        # --- 2. Update Game State ---
        self._update_physics()
        self._update_particles()
        if self.screen_shake > 0:
            self.screen_shake -= 1
        self.steps += 1

        # --- 3. Check for Termination ---
        collapse = self._check_collapse()
        time_up = self.steps >= self.MAX_STEPS
        self.win = self.score >= self.WIN_SCORE

        if collapse or time_up or self.win:
            terminated = True
            self.game_over = True
            if self.win:
                reward += 100
                # sfx: win_jingle
            elif collapse:
                reward -= 50
                self.screen_shake = 15
                # sfx: stack_collapse
            # sfx: game_over
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Movement ---
        if movement == 3: # Left
            self.cursor_col = max(0, self.cursor_col - 1)
        elif movement == 4: # Right
            self.cursor_col = min(self.GRID_COLS - 1, self.cursor_col + 1)
        # Up/Down (1, 2) are ignored for this game's control scheme

        # --- Rotation (on press) ---
        if shift_held and not self.last_shift_held:
            if self.next_block['type'] != 'small':
                self.next_block['rotation'] = 1 - self.next_block['rotation']
                # sfx: rotate_block

        # --- Placement (on press) ---
        if space_held and not self.last_space_held:
            placed, place_reward = self._place_block()
            if placed:
                reward += place_reward
                # sfx: place_block_success
            else:
                self.screen_shake = 5
                # sfx: place_block_fail

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        return reward

    def _place_block(self):
        w, h = self.next_block['dims']
        if self.next_block['rotation'] == 1:
            w, h = h, w

        place_col = min(self.cursor_col, self.GRID_COLS - w)

        max_support_height = 0
        for i in range(w):
            max_support_height = max(max_support_height, self.grid_heights[place_col + i])
        
        for i in range(w):
            if self.grid_heights[place_col + i] != max_support_height:
                return False, 0 # Cannot place on uneven surface

        if max_support_height + h > self.GRID_ROWS:
            return False, 0 # Stack too high

        new_block = {
            'id': self.steps, 'type': self.next_block['type'],
            'cx': place_col, 'cy': max_support_height,
            'w': w, 'h': h,
            'color': self.next_block['color'], 'property': self.next_block['property'],
            'angle': 0.0, 'wobble': 0.0,
        }
        self.placed_blocks.append(new_block)

        for i in range(w):
            self.grid_heights[place_col + i] += h
        
        wobble_amount = 0.5 * w * h
        if new_block['property'] == 'slippery': wobble_amount *= 2.0
        if new_block['property'] == 'sticky': wobble_amount *= 0.5
        
        self._propagate_wobble(place_col, w, max_support_height, wobble_amount)
        new_block['wobble'] += wobble_amount

        self._create_particles(place_col, w, max_support_height + h)
        
        self.score += self.next_block['score']
        reward = self.next_block['score'] + 1 # +1 for any stable placement

        self._generate_next_block()
        return True, reward

    def _propagate_wobble(self, start_x, width, start_y, amount):
        for block in self.placed_blocks:
            is_in_affected_column = (block['cx'] < start_x + width and block['cx'] + block['w'] > start_x)
            is_below = (block['cy'] < start_y)
            if is_in_affected_column and is_below:
                block['wobble'] += amount * 0.7

    def _update_physics(self):
        for block in self.placed_blocks:
            block['wobble'] = max(0, block['wobble'] * 0.96 - 0.01)
            time_factor = (self.steps + block['id']) * 0.25
            block['angle'] = math.sin(time_factor) * block['wobble'] * 0.05

    def _check_collapse(self):
        for block in self.placed_blocks:
            if abs(block['angle']) > math.radians(20):
                return True
        return False

    def _generate_next_block(self):
        block_type = self.np_random.choice(list(self.BLOCK_DEFINITIONS.keys()))
        props = self.BLOCK_DEFINITIONS[block_type]
        self.next_block = {
            'type': block_type,
            'dims': props['dims'],
            'color': props['color'],
            'property': props['property'],
            'score': props['score'],
            'rotation': 0,
        }

    def _create_particles(self, cx, w, top_y):
        px = self.GRID_X_OFFSET + (cx + w / 2) * self.CELL_SIZE
        py = self.GRID_Y_OFFSET - top_y * self.CELL_SIZE
        color = self.next_block['color']
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5)
            self.particles.append({
                'x': px, 'y': py,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(15, 30), 'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.2 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        render_offset_x = self.np_random.integers(-self.screen_shake, self.screen_shake + 1) if self.screen_shake > 0 else 0
        render_offset_y = self.np_random.integers(-self.screen_shake, self.screen_shake + 1) if self.screen_shake > 0 else 0
        
        self.screen.fill(self.COLOR_BG)
        self._render_grid(render_offset_x, render_offset_y)

        for block in sorted(self.placed_blocks, key=lambda b: b['cy']):
            self._render_block_in_grid(self.screen, block, render_offset_x, render_offset_y)

        if not self.game_over:
            self._render_ghost(render_offset_x, render_offset_y)
        
        for p in self.particles:
            size = p['life'] / 7.0
            color_val = tuple(c * (p['life'] / 30.0) for c in p['color'])
            pygame.draw.rect(self.screen, color_val, (p['x'] + render_offset_x, p['y'] + render_offset_y, size, size))

        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self, ox, oy):
        pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET + ox, self.GRID_Y_OFFSET + oy), (self.GRID_X_OFFSET + self.GRID_COLS * self.CELL_SIZE + ox, self.GRID_Y_OFFSET + oy), 3)
        for i in range(self.GRID_COLS + 1):
            x = self.GRID_X_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x + ox, self.GRID_Y_OFFSET + oy), (x + ox, self.GRID_Y_OFFSET - self.GRID_ROWS * self.CELL_SIZE + oy))

    def _render_block_in_grid(self, surface, block, ox, oy):
        w, h = block['w'], block['h']
        center_x = self.GRID_X_OFFSET + (block['cx'] + w / 2) * self.CELL_SIZE + ox
        center_y = self.GRID_Y_OFFSET - (block['cy'] + h / 2) * self.CELL_SIZE + oy
        self._draw_rotated_rect(surface, center_x, center_y, w * self.CELL_SIZE, h * self.CELL_SIZE, block['color'], block['angle'])

    def _render_ghost(self, ox, oy):
        w, h = self.next_block['dims']
        if self.next_block['rotation'] == 1: w, h = h, w
        
        place_col = min(self.cursor_col, self.GRID_COLS - w)

        max_support_height, valid_placement = 0, True
        for i in range(w):
            max_support_height = max(max_support_height, self.grid_heights[place_col + i])
        for i in range(w):
            if self.grid_heights[place_col + i] != max_support_height: valid_placement = False; break
        if max_support_height + h > self.GRID_ROWS: valid_placement = False

        color = self.next_block['color'] if valid_placement else (128, 0, 0)
        ghost_color = (*color, 60)
        
        center_x = self.GRID_X_OFFSET + (place_col + w / 2) * self.CELL_SIZE + ox
        center_y = self.GRID_Y_OFFSET - (max_support_height + h / 2) * self.CELL_SIZE + oy
        
        self._draw_rotated_rect(self.screen, center_x, center_y, w * self.CELL_SIZE, h * self.CELL_SIZE, ghost_color, 0, is_surface=True)
        
        cursor_y = self.GRID_Y_OFFSET - max_support_height * self.CELL_SIZE
        cursor_rect = pygame.Rect(self.GRID_X_OFFSET + place_col * self.CELL_SIZE + ox, cursor_y + oy, w * self.CELL_SIZE, 5)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect)

    def _draw_rotated_rect(self, surface, center_x, center_y, w, h, color, angle, is_surface=False):
        if is_surface:
            temp_surf = pygame.Surface((w, h), pygame.SRCALPHA)
            temp_surf.fill((0,0,0,0))
            pygame.gfxdraw.aapolygon(temp_surf, [(0,0), (w,0), (w,h), (0,h)], color)
            pygame.gfxdraw.filled_polygon(temp_surf, [(0,0), (w,0), (w,h), (0,h)], color)
            rotated_surf = pygame.transform.rotate(temp_surf, -math.degrees(angle))
            rect = rotated_surf.get_rect(center=(center_x, center_y))
            surface.blit(rotated_surf, rect)
        else:
            w_pad, h_pad = w - 4, h - 4
            points = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
            points_inner = [(-w_pad/2, -h_pad/2), (w_pad/2, -h_pad/2), (w_pad/2, h_pad/2), (-w_pad/2, h_pad/2)]

            rotated_points = [(p[0]*math.cos(angle) - p[1]*math.sin(angle) + center_x, p[0]*math.sin(angle) + p[1]*math.cos(angle) + center_y) for p in points]
            rotated_points_inner = [(p[0]*math.cos(angle) - p[1]*math.sin(angle) + center_x, p[0]*math.sin(angle) + p[1]*math.cos(angle) + center_y) for p in points_inner]

            darker_color = tuple(max(0, c - 40) for c in color)
            pygame.gfxdraw.aapolygon(surface, rotated_points, darker_color)
            pygame.gfxdraw.filled_polygon(surface, rotated_points, darker_color)
            pygame.gfxdraw.aapolygon(surface, rotated_points_inner, color)
            pygame.gfxdraw.filled_polygon(surface, rotated_points_inner, color)

    def _render_ui(self):
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_color = self.COLOR_TEXT if time_left > 5 else self.COLOR_TIMER_WARN
        timer_text = self.font_medium.render(f"TIME: {time_left:.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 20, 15))
        self.screen.blit(timer_text, timer_rect)

        preview_text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        preview_rect = preview_text.get_rect(centerx=self.WIDTH - 80, top=70)
        self.screen.blit(preview_text, preview_rect)
        w, h = self.next_block['dims']
        if self.next_block['rotation'] == 1: w, h = h, w
        self._draw_rotated_rect(self.screen, self.WIDTH - 80, 130, w * 30, h * 30, self.next_block['color'], 0)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((*self.COLOR_BG, 200))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    pygame.display.set_caption("Block Stacker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    action = [0, 0, 0]
    
    while running:
        movement, space, shift = 0, 0, 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("\n--- ENV RESET ---\n")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Total: {total_reward:.2f}, Score: {info['score']}")

        if terminated or truncated:
            if 'last_msg' not in locals() or last_msg != info['score']:
                print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                last_msg = info['score']

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()