
# Generated: 2025-08-27T16:49:06.811642
# Source Brief: brief_01340.md
# Brief Index: 1340

        
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
        "Controls: Use arrow keys to move the cursor. Press space to clear a matching color group of 3 or more."
    )

    game_description = (
        "A fast-paced puzzle game. Match cascading colors in a grid to clear the board and achieve the highest score within the time limit. Clear 3 stages to win!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 12, 10
    TILE_SIZE = 32
    MIN_MATCH_SIZE = 3
    NUM_COLORS = 3
    FPS = 30
    STAGE_TIME = 60.0

    # --- Colors ---
    COLOR_BG_TOP = (20, 20, 40)
    COLOR_BG_BOTTOM = (40, 40, 60)
    COLOR_GRID = (60, 60, 80)
    TILE_COLORS = {
        1: (255, 80, 80),   # Red
        2: (80, 255, 80),   # Green
        3: (80, 120, 255)   # Blue
    }
    TILE_SHADOWS = {
        1: (180, 50, 50),
        2: (50, 180, 50),
        3: (50, 80, 180)
    }
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        self.grid_rect = pygame.Rect(
            (self.SCREEN_WIDTH - self.GRID_WIDTH * self.TILE_SIZE) // 2,
            (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.TILE_SIZE) // 2 + 20,
            self.GRID_WIDTH * self.TILE_SIZE,
            self.GRID_HEIGHT * self.TILE_SIZE
        )

        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.cursor_pos = [0, 0]
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.won = False
        self.stage = 1
        self.time_remaining = self.STAGE_TIME
        self.space_pressed_last_frame = False
        self.rng = None

        self.reset()
        # self.validate_implementation() # Optional: Uncomment to run validation on init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.won = False
        self.stage = 1
        self._start_stage()
        
        return self._get_observation(), self._get_info()

    def _start_stage(self):
        self.time_remaining = self.STAGE_TIME
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.grid = self.rng.integers(1, self.NUM_COLORS + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
        self.particles.clear()
        self.space_pressed_last_frame = True # Prevent action on first frame of stage

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # --- Handle Actions ---
        if shift_held:
            # sfx: restart_stage_sound
            self._start_stage()
            reward -= 10.0 # Penalty for restarting
        
        self._move_cursor(movement)

        # Process space press as a discrete event
        if space_held and not self.space_pressed_last_frame:
            # sfx: select_sound
            reward += self._attempt_clear_group()
        self.space_pressed_last_frame = space_held

        # --- Update Game State ---
        self._update_particles()
        self.time_remaining = max(0, self.time_remaining - 1 / self.FPS)
        self.steps += 1

        # --- Check Win/Loss Conditions ---
        if np.all(self.grid == 0): # Stage cleared
            # sfx: stage_clear_sound
            self.score += 10 * self.stage
            reward += 10.0
            self.stage += 1
            if self.stage > 3:
                # sfx: game_win_sound
                self.won = True
                self.game_over = True
                reward += 50.0
            else:
                self._start_stage()
        
        if self.time_remaining <= 0 and not self.game_over:
            # sfx: game_over_sound
            self.game_over = True
            reward -= 50.0

        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_cursor(self, movement):
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

    def _attempt_clear_group(self):
        cx, cy = self.cursor_pos
        if self.grid[cx, cy] == 0:
            return 0

        group = self._find_contiguous_group(cx, cy)
        
        if len(group) >= self.MIN_MATCH_SIZE:
            # sfx: match_clear_sound
            for x, y in group:
                self.grid[x, y] = 0
                self._create_particles(x, y)
            
            self._apply_gravity_and_refill()
            
            # Score based on number of tiles cleared
            tiles_cleared = len(group)
            self.score += tiles_cleared
            return float(tiles_cleared) # Reward
        else:
            # sfx: invalid_move_sound
            return 0.0

    def _find_contiguous_group(self, start_x, start_y):
        target_color = self.grid[start_x, start_y]
        if target_color == 0:
            return []

        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        group = []

        while q:
            x, y = q.popleft()
            group.append((x, y))
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in visited and self.grid[nx, ny] == target_color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return group

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_WIDTH):
            col = self.grid[c, :]
            non_empty_tiles = col[col != 0]
            num_empty = self.GRID_HEIGHT - len(non_empty_tiles)
            
            if num_empty > 0:
                new_tiles = self.rng.integers(1, self.NUM_COLORS + 1, size=num_empty)
                self.grid[c, :] = np.concatenate((new_tiles, non_empty_tiles))

    def _create_particles(self, grid_x, grid_y):
        center_x = self.grid_rect.left + grid_x * self.TILE_SIZE + self.TILE_SIZE // 2
        center_y = self.grid_rect.top + grid_y * self.TILE_SIZE + self.TILE_SIZE // 2
        
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(15, 30) # frames
            size = random.uniform(2, 5)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': life, 'size': size})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.1)
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._update_and_render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stage": self.stage}

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            r = self.COLOR_BG_TOP[0] + (self.COLOR_BG_BOTTOM[0] - self.COLOR_BG_TOP[0]) * y / self.SCREEN_HEIGHT
            g = self.COLOR_BG_TOP[1] + (self.COLOR_BG_BOTTOM[1] - self.COLOR_BG_TOP[1]) * y / self.SCREEN_HEIGHT
            b = self.COLOR_BG_TOP[2] + (self.COLOR_BG_BOTTOM[2] - self.COLOR_BG_TOP[2]) * y / self.SCREEN_HEIGHT
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, (0, 0, 0, 50), self.grid_rect)
        
        # Draw tiles
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                color_id = self.grid[x, y]
                if color_id > 0:
                    tile_rect = pygame.Rect(
                        self.grid_rect.left + x * self.TILE_SIZE,
                        self.grid_rect.top + y * self.TILE_SIZE,
                        self.TILE_SIZE, self.TILE_SIZE
                    )
                    shadow_rect = tile_rect.move(2, 2)
                    pygame.draw.rect(self.screen, self.TILE_SHADOWS[color_id], shadow_rect, border_radius=5)
                    pygame.draw.rect(self.screen, self.TILE_COLORS[color_id], tile_rect, border_radius=5)
        
        # Draw cursor
        cursor_pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
        alpha = 100 + 100 * cursor_pulse
        cursor_rect = pygame.Rect(
            self.grid_rect.left + self.cursor_pos[0] * self.TILE_SIZE,
            self.grid_rect.top + self.cursor_pos[1] * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        cursor_surface = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, (*self.COLOR_CURSOR, alpha), cursor_surface.get_rect(), width=3, border_radius=5)
        self.screen.blit(cursor_surface, cursor_rect.topleft)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, self.COLOR_CURSOR, p['pos'], p['size'])

    def _update_and_render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos, center=False):
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_surf = font.render(text, True, color)
            
            shadow_rect = shadow_surf.get_rect()
            text_rect = text_surf.get_rect()
            
            if center:
                shadow_rect.center = (pos[0] + 2, pos[1] + 2)
                text_rect.center = pos
            else:
                shadow_rect.topleft = (pos[0] + 2, pos[1] + 2)
                text_rect.topleft = pos

            self.screen.blit(shadow_surf, shadow_rect)
            self.screen.blit(text_surf, text_rect)

        # Score
        draw_text(f"SCORE: {self.score}", self.font_medium, self.COLOR_TEXT, (20, 15))
        
        # Timer
        timer_text = f"TIME: {int(self.time_remaining):02d}"
        text_w = self.font_medium.render(timer_text, True, self.COLOR_TEXT).get_width()
        draw_text(timer_text, self.font_medium, self.COLOR_TEXT, (self.SCREEN_WIDTH - text_w - 20, 15))

        # Stage
        draw_text(f"STAGE {self.stage} / 3", self.font_medium, self.COLOR_TEXT, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 25), center=True)

        # Game Over / Win Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.won:
                draw_text("YOU WIN!", self.font_large, (100, 255, 100), (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 40), center=True)
                draw_text(f"Final Score: {self.score}", self.font_medium, self.COLOR_TEXT, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 20), center=True)
            else:
                draw_text("GAME OVER", self.font_large, (255, 100, 100), (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 40), center=True)
                draw_text(f"You reached stage {self.stage}", self.font_medium, self.COLOR_TEXT, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 20), center=True)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    
    # Optional: Run validation
    try:
        env.validate_implementation()
    except AssertionError as e:
        print(f"Validation failed: {e}")
        env.close()
        exit()

    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Color Cascade")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    # --- Main game loop ---
    while not done:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Info: {info}")
    
    # Keep window open for a few seconds to see the final screen
    pygame.time.wait(3000)
    
    env.close()