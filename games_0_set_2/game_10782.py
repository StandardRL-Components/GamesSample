import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:03:55.856393
# Source Brief: brief_00782.md
# Brief Index: 782
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a minimalist puzzle game.
    The player stacks falling shapes, and identical adjacent shapes merge into a larger, higher-level shape.
    The goal is to score as many points as possible by creating merges before the stack reaches the top.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Space button (unused)
    - actions[2]: Shift button (unused)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Stack falling shapes and merge identical adjacent ones to create higher-level shapes. "
        "Score points by creating merges before the stack reaches the top."
    )
    user_guide = "Controls: Use the ← and → arrow keys to move the falling shape."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 10
    CELL_SIZE = 40
    MAX_STEPS = 1000
    MAX_LEVEL = 7

    COLOR_BG = (20, 25, 35)
    COLOR_GRID = (35, 40, 55)
    COLOR_DANGER = (180, 50, 50)
    
    SHAPE_PROPS = [
        {'color': (50, 150, 255), 'radius_mult': 0.6, 'score': 10},   # Level 0: Blue
        {'color': (80, 220, 120), 'radius_mult': 0.65, 'score': 20},  # Level 1: Green
        {'color': (255, 220, 80), 'radius_mult': 0.7, 'score': 40},   # Level 2: Yellow
        {'color': (255, 150, 50), 'radius_mult': 0.75, 'score': 80},  # Level 3: Orange
        {'color': (255, 80, 80), 'radius_mult': 0.8, 'score': 160},  # Level 4: Red
        {'color': (220, 100, 255), 'radius_mult': 0.85, 'score': 320},# Level 5: Purple
        {'color': (255, 255, 255), 'radius_mult': 0.9, 'score': 640},# Level 6: White
        {'color': (255, 215, 0), 'radius_mult': 1.0, 'score': 1280}, # Level 7: Gold (Max)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        self.game_over_font = pygame.font.SysFont("monospace", 48, bold=True)

        # Game state variables
        self.grid = None
        self.falling_shape = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.merge_animations = []
        self.last_action_time = 0
        self.action_cooldown = 100 # ms between moves

        # Derived constants
        self.play_area_width = self.GRID_WIDTH * self.CELL_SIZE
        self.start_x = (self.WIDTH - self.play_area_width) // 2
        self.fall_speed = 2.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.full((self.GRID_HEIGHT, self.GRID_WIDTH), -1, dtype=int)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.merge_animations = []
        self._spawn_new_shape()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, _, _ = action
        reward = 0
        terminated = False
        
        current_time = pygame.time.get_ticks()
        if current_time - self.last_action_time > self.action_cooldown:
            if movement == 3:  # Left
                self.falling_shape['grid_x'] = max(0, self.falling_shape['grid_x'] - 1)
            elif movement == 4: # Right
                self.falling_shape['grid_x'] = min(self.GRID_WIDTH - 1, self.falling_shape['grid_x'] + 1)
            if movement in [3,4]:
                 self.last_action_time = current_time

        # --- Update game logic ---
        self._update_falling_shape()

        if self.falling_shape is None: # A shape has landed
            reward += 0.1 # Placement reward
            reward += self._settle_board()
            
            if self._check_game_over():
                self.game_over = True
                reward = -100.0
            else:
                self._spawn_new_shape()

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
             reward += 10.0 # Survival bonus

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _update_falling_shape(self):
        if self.falling_shape is None:
            return

        shape = self.falling_shape
        shape['pixel_y'] += self.fall_speed
        
        target_grid_y = int(shape['pixel_y'] / self.CELL_SIZE)
        
        # Check for collision with floor or other shapes
        collided = False
        if target_grid_y >= self.GRID_HEIGHT:
            collided = True
            target_grid_y = self.GRID_HEIGHT - 1
        elif self.grid[target_grid_y][shape['grid_x']] != -1:
            collided = True
            target_grid_y -= 1

        if collided:
            if target_grid_y >= 0:
                self.grid[target_grid_y][shape['grid_x']] = shape['level']
            else: # Placed above the screen
                self.game_over = True
            self.falling_shape = None
            # Sound placeholder: # sfx_land.play()
    
    def _settle_board(self):
        total_merge_reward = 0
        while True:
            merges_found, merge_reward = self._process_merges()
            total_merge_reward += merge_reward
            gravity_applied = self._apply_gravity()
            if not merges_found and not gravity_applied:
                break
        return total_merge_reward

    def _process_merges(self):
        merges_found = False
        reward = 0
        
        # Horizontal merges
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 1):
                level1 = self.grid[y][x]
                level2 = self.grid[y][x+1]
                if level1 != -1 and level1 == level2 and level1 < self.MAX_LEVEL:
                    self.grid[y][x] = -1
                    self.grid[y][x+1] = level1 + 1
                    self._create_merge_fx(x + 0.5, y, level1 + 1)
                    reward += 10
                    self.score += self.SHAPE_PROPS[level1]['score']
                    merges_found = True
                    # Sound placeholder: # sfx_merge.play()
        
        # Vertical merges
        for y in range(self.GRID_HEIGHT - 1):
            for x in range(self.GRID_WIDTH):
                level1 = self.grid[y][x]
                level2 = self.grid[y+1][x]
                if level1 != -1 and level1 == level2 and level1 < self.MAX_LEVEL:
                    self.grid[y][x] = -1
                    self.grid[y+1][x] = level1 + 1
                    self._create_merge_fx(x, y + 0.5, level1 + 1)
                    reward += 10
                    self.score += self.SHAPE_PROPS[level1]['score']
                    merges_found = True
                    # Sound placeholder: # sfx_merge.play()
        
        return merges_found, reward

    def _apply_gravity(self):
        moved = False
        for x in range(self.GRID_WIDTH):
            empty_y = -1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y][x] == -1 and empty_y == -1:
                    empty_y = y
                elif self.grid[y][x] != -1 and empty_y != -1:
                    self.grid[empty_y][x] = self.grid[y][x]
                    self.grid[y][x] = -1
                    moved = True
                    empty_y -= 1
        return moved
        
    def _check_game_over(self):
        return np.any(self.grid[0] != -1)

    def _spawn_new_shape(self):
        level = self.np_random.integers(0, 3) # Spawn more of the basic shapes
        self.falling_shape = {
            'level': level,
            'grid_x': self.GRID_WIDTH // 2,
            'pixel_y': -self.CELL_SIZE, # Start off-screen
        }

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid_and_danger_zone()
        self._update_and_draw_effects()
        self._draw_stacked_shapes()
        self._draw_falling_shape()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
    
    def _draw_grid_and_danger_zone(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = self.start_x + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.HEIGHT))
        
        # Danger Zone
        danger_y = self.CELL_SIZE
        danger_rect = pygame.Rect(self.start_x, 0, self.play_area_width, danger_y)
        s = pygame.Surface((self.play_area_width, danger_y), pygame.SRCALPHA)
        s.fill((self.COLOR_DANGER[0], self.COLOR_DANGER[1], self.COLOR_DANGER[2], 50))
        self.screen.blit(s, (self.start_x, 0))
        pygame.draw.line(self.screen, self.COLOR_DANGER, (self.start_x, danger_y), (self.start_x + self.play_area_width, danger_y), 2)

    def _draw_stacked_shapes(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                level = self.grid[y][x]
                if level != -1:
                    self._draw_shape(level, x, y)

    def _draw_falling_shape(self):
        if self.falling_shape:
            shape = self.falling_shape
            px = self.start_x + shape['grid_x'] * self.CELL_SIZE + self.CELL_SIZE / 2
            py = shape['pixel_y'] + self.CELL_SIZE / 2
            props = self.SHAPE_PROPS[shape['level']]
            self._draw_circle(px, py, props['radius_mult'], props['color'])

    def _draw_shape(self, level, grid_x, grid_y):
        props = self.SHAPE_PROPS[level]
        px = self.start_x + grid_x * self.CELL_SIZE + self.CELL_SIZE / 2
        py = grid_y * self.CELL_SIZE + self.CELL_SIZE / 2
        self._draw_circle(px, py, props['radius_mult'], props['color'])

    def _draw_circle(self, x, y, radius_mult, color):
        radius = int(self.CELL_SIZE / 2 * radius_mult)
        x, y = int(x), int(y)
        
        # Glow effect
        glow_radius = int(radius * 1.5)
        glow_color = (*color, 100)
        temp_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surface, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surface, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Main circle
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)

    def _create_merge_fx(self, grid_x, grid_y, level):
        px = self.start_x + grid_x * self.CELL_SIZE + self.CELL_SIZE / 2
        py = grid_y * self.CELL_SIZE + self.CELL_SIZE / 2
        props = self.SHAPE_PROPS[level]
        
        # Add merge ring animation
        self.merge_animations.append({'x': px, 'y': py, 'radius': 0, 'max_radius': self.CELL_SIZE, 'color': props['color'], 'life': 1.0})
        
        # Add particles
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': px, 'y': py,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': self.np_random.uniform(0.5, 1.0), 'max_life': 1.0,
                'color': props['color']
            })

    def _update_and_draw_effects(self):
        # Update and draw particles
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 0.03
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / p['max_life']))
                pygame.draw.circle(self.screen, (*p['color'], alpha), (int(p['x']), int(p['y'])), 2)
        
        # Update and draw merge animations
        for anim in self.merge_animations[:]:
            anim['life'] -= 0.05
            if anim['life'] <= 0:
                self.merge_animations.remove(anim)
            else:
                current_radius = int(anim['max_radius'] * (1.0 - anim['life']))
                alpha = int(255 * anim['life'])
                color = (*anim['color'], alpha)
                if current_radius > 0:
                    temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, color, (int(anim['x']), int(anim['y'])), current_radius, width=max(1, int(8 * anim['life'])))
                    self.screen.blit(temp_surf, (0,0))

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, (240, 240, 255))
        self.screen.blit(score_text, (10, 10))

    def _render_game_over(self):
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))
        
        text = self.game_over_font.render("GAME OVER", True, self.COLOR_DANGER)
        text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(text, text_rect)
    
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This validation code is removed in the final output as per instructions,
    # but it's useful for local testing. The original `validate_implementation`
    # method was removed as it's not part of the standard Gym API.
    
    # --- Manual Play Example ---
    # This part is for human interaction and debugging, not for the agent environment.
    # It requires a display.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Re-initialize pygame for display
    pygame.display.set_caption("Shape Stacker")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            pygame.time.wait(2000) # Pause on game over
            obs, info = env.reset()
            terminated = False

        clock.tick(60) # Run at 60 FPS for smooth visuals

    env.close()