
# Generated: 2025-08-28T00:39:57.616793
# Source Brief: brief_03858.md
# Brief Index: 3858

        
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
        "Controls: Arrows to move cursor. Hold Space to start a connection on a colored block. "
        "Drag over adjacent, same-colored blocks to extend the line. Release Space to clear."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Connect lines of 3 or more same-colored blocks to clear them from the grid. "
        "Clear all blocks before the 60-second timer runs out to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Sizing
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 8
    BLOCK_SIZE = 40
    GRID_PIXEL_WIDTH = GRID_WIDTH * BLOCK_SIZE
    GRID_PIXEL_HEIGHT = GRID_HEIGHT * BLOCK_SIZE
    X_OFFSET = (SCREEN_WIDTH - GRID_PIXEL_WIDTH) // 2
    Y_OFFSET = (SCREEN_HEIGHT - GRID_PIXEL_HEIGHT) // 2

    # Colors (0=empty, -1=grey, 1-5=colors)
    COLOR_BG = (25, 35, 45)
    COLOR_GRID_BG = (40, 50, 60)
    COLOR_CURSOR = (255, 255, 255)
    COLORS = {
        -1: (100, 110, 120),  # Grey
        1: (227, 80, 80),    # Red
        2: (80, 200, 120),   # Green
        3: (80, 120, 227),   # Blue
        4: (240, 230, 100),  # Yellow
        5: (180, 100, 220),  # Purple
    }
    NUM_COLORS = len(COLORS) -1

    # Game parameters
    FPS = 30
    MAX_TIME = 60.0
    MAX_STEPS = int(MAX_TIME * FPS)
    MIN_CLEAR_LENGTH = 3

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
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 16)
        
        self.grid = None
        self.cursor_pos = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.timer = None
        self.is_connecting = None
        self.connection_path = None
        self.connection_color = None
        self.previous_space_held = None
        self.particles = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_TIME
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.is_connecting = False
        self.connection_path = []
        self.connection_color = 0
        self.previous_space_held = False
        self.particles = []

        self._generate_grid()

        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.int8)

        num_grey_blocks = int(self.GRID_WIDTH * self.GRID_HEIGHT * 0.1)
        for _ in range(num_grey_blocks):
            gx, gy = self.np_random.integers(0, [self.GRID_WIDTH, self.GRID_HEIGHT])
            self.grid[gy][gx] = -1

        attempts = 0
        while attempts < 100:
            start_x = self.np_random.integers(0, self.GRID_WIDTH - (self.MIN_CLEAR_LENGTH - 1))
            start_y = self.np_random.integers(0, self.GRID_HEIGHT)
            path_color = self.np_random.integers(1, self.NUM_COLORS + 1)
            
            can_place = all(self.grid[start_y][start_x + i] != -1 for i in range(self.MIN_CLEAR_LENGTH))
            
            if can_place:
                for i in range(self.MIN_CLEAR_LENGTH):
                    self.grid[start_y][start_x + i] = path_color
                break
            attempts += 1

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held = action[0], action[1] == 1
        reward = self._handle_input(movement, space_held)
        
        self._update_game_state()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            self.game_over = True
            if self._all_blocks_cleared():
                reward += 100
            else:
                reward -= 100
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        reward = 0
        
        if movement > 0:
            reward -= 0.01
            if movement == 1: self.cursor_pos[1] -= 1
            elif movement == 2: self.cursor_pos[1] += 1
            elif movement == 3: self.cursor_pos[0] -= 1
            elif movement == 4: self.cursor_pos[0] += 1
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        space_pressed = space_held and not self.previous_space_held
        space_released = not space_held and self.previous_space_held
        
        cx, cy = self.cursor_pos
        
        if space_pressed:
            block_type = self.grid[cy][cx]
            if block_type > 0 and not self.is_connecting:
                self.is_connecting = True
                self.connection_path = [(cx, cy)]
                self.connection_color = block_type
        
        elif space_held and self.is_connecting:
            last_pos = self.connection_path[-1]
            is_adjacent = abs(cx - last_pos[0]) + abs(cy - last_pos[1]) == 1
            is_correct_color = self.grid[cy][cx] == self.connection_color
            is_new = (cx, cy) not in self.connection_path

            if is_adjacent and is_correct_color and is_new:
                self.connection_path.append((cx, cy))
                reward += 0.1
            elif len(self.connection_path) > 1 and (cx, cy) == self.connection_path[-2]:
                self.connection_path.pop()

        elif space_released and self.is_connecting:
            if len(self.connection_path) >= self.MIN_CLEAR_LENGTH:
                reward += len(self.connection_path) - (self.MIN_CLEAR_LENGTH - 1)
                for bx, by in self.connection_path:
                    self.grid[by][bx] = 0
                    self._create_particles(bx, by, self.connection_color)
                # play_sound('clear_blocks.wav')
                self._apply_gravity()
            
            self.is_connecting = False
            self.connection_path = []
            self.connection_color = 0

        self.previous_space_held = space_held
        return reward

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y][x] != 0:
                    if y != empty_row:
                        self.grid[empty_row][x] = self.grid[y][x]
                        self.grid[y][x] = 0
                    empty_row -= 1

    def _create_particles(self, block_x, block_y, color_id):
        center_x = self.X_OFFSET + block_x * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        center_y = self.Y_OFFSET + block_y * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        color = self.COLORS[color_id]
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 25)
            radius = self.np_random.uniform(2, 5)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': life, 'max_life': life, 'color': color, 'radius': radius})
        # play_sound('particle_burst.wav')

    def _update_game_state(self):
        self.timer -= 1 / self.FPS
        
        new_particles = []
        for p in self.particles:
            p['life'] -= 1
            if p['life'] > 0:
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                p['vel'][0] *= 0.98
                p['vel'][1] *= 0.98
                new_particles.append(p)
        self.particles = new_particles

    def _all_blocks_cleared(self):
        return not np.any((self.grid > 0))

    def _check_termination(self):
        return self.timer <= 0 or self._all_blocks_cleared() or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_rect = pygame.Rect(self.X_OFFSET, self.Y_OFFSET, self.GRID_PIXEL_WIDTH, self.GRID_PIXEL_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)

        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                block_type = self.grid[y][x]
                if block_type != 0:
                    color = self.COLORS[block_type]
                    rect = pygame.Rect(self.X_OFFSET + x * self.BLOCK_SIZE, self.Y_OFFSET + y * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
                    pygame.draw.rect(self.screen, color, rect.inflate(-4, -4), border_radius=6)
                    pygame.draw.rect(self.screen, tuple(c*0.8 for c in color), rect.inflate(-4, -4), width=2, border_radius=6)

        if self.is_connecting and self.connection_path:
            highlight_surface = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
            highlight_surface.fill((255, 255, 255, 60))
            for x, y in self.connection_path:
                self.screen.blit(highlight_surface, (self.X_OFFSET + x * self.BLOCK_SIZE, self.Y_OFFSET + y * self.BLOCK_SIZE))
            
            if len(self.connection_path) > 1:
                points = [(self.X_OFFSET + x * self.BLOCK_SIZE + self.BLOCK_SIZE // 2, 
                           self.Y_OFFSET + y * self.BLOCK_SIZE + self.BLOCK_SIZE // 2) 
                          for x, y in self.connection_path]
                color = self.COLORS[self.connection_color]
                pygame.draw.lines(self.screen, color, False, points, width=8)

        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            life_ratio = p['life'] / p['max_life']
            current_radius = int(p['radius'] * life_ratio)
            if current_radius > 0:
                alpha = int(255 * life_ratio)
                color = (*p['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], current_radius, color)
        
        cursor_rect = pygame.Rect(self.X_OFFSET + self.cursor_pos[0] * self.BLOCK_SIZE,
                                  self.Y_OFFSET + self.cursor_pos[1] * self.BLOCK_SIZE,
                                  self.BLOCK_SIZE, self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=3, border_radius=8)

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 10))

        time_left = max(0, self.timer)
        timer_color = (255, 255, 255) if time_left > 10 else (255, 100, 100)
        timer_text = self.font_large.render(f"TIME: {time_left:.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(timer_text, timer_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self._all_blocks_cleared() else "TIME'S UP!"
            end_text = self.font_large.render(msg, True, (255, 255, 255))
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "cleared_all": self._all_blocks_cleared()
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    try:
        window = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Clear The Lines")
        
        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            window.blit(surf, (0, 0))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            env.clock.tick(GameEnv.FPS)

    finally:
        env.close()