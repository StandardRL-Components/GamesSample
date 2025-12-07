import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:20:23.829452
# Source Brief: brief_00401.md
# Brief Index: 401
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A helper class for creating visual effects like sparks and crumbling blocks."""
    def __init__(self, x, y, color, type='spark'):
        self.x = x
        self.y = y
        self.color = color
        self.type = type
        if self.type == 'spark':
            self.vx = random.uniform(-2, 2)
            self.vy = random.uniform(-2, 2)
            self.lifespan = random.randint(15, 30)
            self.size = random.uniform(2, 5)
        elif self.type == 'crumble':
            self.vx = random.uniform(-1, 1)
            self.vy = random.uniform(-1, 3) # Start with a slight upward pop
            self.lifespan = random.randint(40, 80)
            self.size = random.uniform(5, 15)
            self.gravity = 0.1
            self.rotation = random.uniform(0, 360)
            self.rot_speed = random.uniform(-5, 5)

    def update(self):
        if self.type == 'crumble':
            self.vy += self.gravity
            self.rotation += self.rot_speed
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.size = max(0, self.size * 0.98)
        return self.lifespan <= 0

    def draw(self, surface):
        if self.lifespan <= 0:
            return
        alpha = int(255 * (self.lifespan / 30)) if self.type == 'spark' else int(255 * (self.lifespan / 80))
        alpha = max(0, min(255, alpha))
        
        r, g, b = self.color
        draw_color = (r, g, b, alpha)

        if self.type == 'spark':
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.size), draw_color)
        elif self.type == 'crumble':
            rect = pygame.Rect(0, 0, int(self.size), int(self.size))
            rect.center = (int(self.x), int(self.y))
            
            poly_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(poly_surf, draw_color, (0, 0, rect.width, rect.height))
            rotated_surf = pygame.transform.rotate(poly_surf, self.rotation)
            
            new_rect = rotated_surf.get_rect(center=rect.center)
            surface.blit(rotated_surf, new_rect.topleft)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Stack falling blocks to build a stable structure. Transform blocks between cubes and pyramids on the beat to complete the grid."
    )
    user_guide = (
        "Controls: ←→ to move, ↑ to rotate, ↓ to drop faster. Press Shift on the beat to transform the block's shape."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 5
    BLOCK_PIXEL_SIZE = 50
    GRID_WIDTH = GRID_SIZE * BLOCK_PIXEL_SIZE
    GRID_HEIGHT = GRID_SIZE * BLOCK_PIXEL_SIZE
    GRID_ORIGIN_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_ORIGIN_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + 20 # Lower grid slightly

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (50, 60, 80)
    COLOR_BEAT_GOOD = (0, 255, 150)
    COLOR_BEAT_BAD = (80, 100, 120)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CUBE = (60, 150, 255)
    COLOR_PYRAMID = (255, 80, 120)

    # Game Mechanics
    MAX_STEPS = 3600 # 60 seconds at 60 FPS
    BASE_FALL_SPEED = 0.8
    BEAT_PERIOD = 40 # steps for one beat cycle
    
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
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        
        self.render_mode = render_mode
        
        # State variables initialized in reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = None
        self.current_block = None
        self.particles = []
        self.last_shift_state = 0
        self.beat_phase = 0.0
        
        self.validate_implementation_is_optional = True
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.particles = []
        self.last_shift_state = 0
        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # Update beat
        self.beat_phase = (self.steps % self.BEAT_PERIOD) / self.BEAT_PERIOD
        
        # Handle input and get immediate rewards
        reward += self._handle_input(action)
        
        # Update physics
        landed = self._update_physics()
        
        # Handle landing/collapse
        if landed:
            landing_reward, is_win = self._handle_landing()
            reward += landing_reward
            if is_win:
                self.game_over = True
                reward += 100
            else:
                self._spawn_new_block()
        
        # Update visual effects
        self._update_particles()
        
        # Check termination conditions
        terminated = self.game_over
        if not terminated and self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 100 # Timeout penalty
            self.game_over = True

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_new_block(self):
        shape_id = self.np_random.integers(1, 3) # 1 for cube, 2 for pyramid
        self.current_block = {
            "shape": shape_id,
            "x": self.np_random.integers(0, self.GRID_SIZE),
            "y": -self.BLOCK_PIXEL_SIZE * 2, # Start above screen
            "rotation": 0, # 0, 1, 2, 3 for 90-degree increments
            "fall_speed": self.BASE_FALL_SPEED,
            "transform_progress": 0.0,
            "target_shape": shape_id,
        }

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Movement
        if movement == 1: # Up -> Rotate
            self.current_block["rotation"] = (self.current_block["rotation"] + 1) % 4
        elif movement == 2: # Down -> Accelerate
            self.current_block["y"] += self.BLOCK_PIXEL_SIZE / 4
        elif movement == 3: # Left
            self.current_block["x"] = max(0, self.current_block["x"] - 1)
        elif movement == 4: # Right
            self.current_block["x"] = min(self.GRID_SIZE - 1, self.current_block["x"] + 1)

        # Transformation (on key press)
        if shift_held and not self.last_shift_state:
            # sound_placeholder: "transform_sfx"
            self.current_block["target_shape"] = 3 - self.current_block["shape"] # Toggle 1<->2
            self.current_block["transform_progress"] = 1.0
            
            # Reward for on-beat transform
            if self.beat_phase < 0.1 or self.beat_phase > 0.9:
                reward += 0.1
                cx, cy = self._grid_to_pixel(self.current_block["x"], self.current_block["y"] / self.BLOCK_PIXEL_SIZE)
                for _ in range(10):
                    self.particles.append(Particle(cx, cy, self.COLOR_BEAT_GOOD, 'spark'))

        self.last_shift_state = shift_held
        return reward

    def _update_physics(self):
        # Update transformation animation
        if self.current_block["transform_progress"] > 0:
            self.current_block["transform_progress"] -= 0.1 # Animation speed
            if self.current_block["transform_progress"] <= 0:
                self.current_block["shape"] = self.current_block["target_shape"]

        # Apply gravity
        self.current_block["y"] += self.current_block["fall_speed"]

        # Check for landing
        grid_x = self.current_block["x"]
        grid_y = math.floor(self.current_block["y"] / self.BLOCK_PIXEL_SIZE)
        
        # Land on floor
        if grid_y >= self.GRID_SIZE - 1:
            return True
        # Land on another block
        if grid_y >= -1 and self.grid[grid_y + 1, grid_x] > 0:
            if self.current_block["y"] > grid_y * self.BLOCK_PIXEL_SIZE:
                 return True
        return False

    def _handle_landing(self):
        # sound_placeholder: "land_sfx"
        grid_x = self.current_block["x"]
        
        # Find the highest empty spot in the column
        landing_y = self.GRID_SIZE - 1
        while landing_y >= 0 and self.grid[landing_y, grid_x] > 0:
            landing_y -= 1
        
        if landing_y < 0: # Column is full, treat as collapse
            return -5, False # No win
        
        # Place block
        self.grid[landing_y, grid_x] = self.current_block["shape"]
        
        px, py = self._grid_to_pixel(grid_x, landing_y)
        for _ in range(20):
            self.particles.append(Particle(px, py, self.COLOR_CUBE if self.current_block["shape"] == 1 else self.COLOR_PYRAMID, 'spark'))

        # Check for collapse and win
        num_collapsed = self._check_collapse()
        if num_collapsed > 0:
            # sound_placeholder: "collapse_sfx"
            return -5 * num_collapsed, False

        is_win = np.all(self.grid > 0)
        return 1, is_win # +1 for successful placement

    def _check_collapse(self):
        supported_mask = np.zeros_like(self.grid, dtype=bool)
        
        # Blocks on the floor are supported
        supported_mask[self.GRID_SIZE - 1, :] = (self.grid[self.GRID_SIZE - 1, :] > 0)
        
        # Propagate support upwards
        for r in range(self.GRID_SIZE - 2, -1, -1):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] > 0 and supported_mask[r + 1, c]:
                    supported_mask[r, c] = True

        # Find and remove unsupported blocks
        num_collapsed = 0
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] > 0 and not supported_mask[r, c]:
                    num_collapsed += 1
                    block_type = self.grid[r, c]
                    self.grid[r, c] = 0
                    px, py = self._grid_to_pixel(c, r)
                    color = self.COLOR_CUBE if block_type == 1 else self.COLOR_PYRAMID
                    for _ in range(15):
                        self.particles.append(Particle(px, py, color, 'crumble'))
        return num_collapsed

    def _update_particles(self):
        self.particles = [p for p in self.particles if not p.update()]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _grid_to_pixel(self, grid_x, grid_y):
        px = self.GRID_ORIGIN_X + grid_x * self.BLOCK_PIXEL_SIZE + self.BLOCK_PIXEL_SIZE // 2
        py = self.GRID_ORIGIN_Y + grid_y * self.BLOCK_PIXEL_SIZE + self.BLOCK_PIXEL_SIZE // 2
        return px, py

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            start_pos_v = (self.GRID_ORIGIN_X + i * self.BLOCK_PIXEL_SIZE, self.GRID_ORIGIN_Y)
            end_pos_v = (self.GRID_ORIGIN_X + i * self.BLOCK_PIXEL_SIZE, self.GRID_ORIGIN_Y + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos_v, end_pos_v, 1)
            start_pos_h = (self.GRID_ORIGIN_X, self.GRID_ORIGIN_Y + i * self.BLOCK_PIXEL_SIZE)
            end_pos_h = (self.GRID_ORIGIN_X + self.GRID_WIDTH, self.GRID_ORIGIN_Y + i * self.BLOCK_PIXEL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos_h, end_pos_h, 1)

        # Draw stacked blocks
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] > 0:
                    px, py = self._grid_to_pixel(c, r)
                    self._draw_block(self.screen, px, py, self.grid[r,c], 0, 0.0, False)

        # Draw falling block
        if self.current_block:
            px = self.GRID_ORIGIN_X + self.current_block["x"] * self.BLOCK_PIXEL_SIZE + self.BLOCK_PIXEL_SIZE // 2
            py = self.current_block["y"] + self.GRID_ORIGIN_Y + self.BLOCK_PIXEL_SIZE / 2
            self._draw_block(self.screen, px, py, self.current_block["shape"], 
                             self.current_block["rotation"], self.current_block["transform_progress"], True)
        
        # Draw particles
        particle_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for p in self.particles:
            p.draw(particle_surface)
        self.screen.blit(particle_surface, (0,0), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_block(self, surface, px, py, shape, rotation, transform_progress, is_falling):
        size = self.BLOCK_PIXEL_SIZE * 0.85
        half = size / 2

        color = self.COLOR_CUBE if shape == 1 else self.COLOR_PYRAMID
        if not is_falling:
            color = tuple(int(c * 0.8) for c in color)
        
        if is_falling:
            glow_color = (*color, 60)
            pygame.gfxdraw.filled_circle(surface, int(px), int(py), int(size * 0.7), glow_color)

        p = transform_progress
        target_shape = self.current_block["target_shape"]
        
        p1 = (-half, -half); p2 = ( half, -half); p3 = ( half,  half); p4 = (-half,  half)
        t1 = (0, -half); t2 = (half, half); t3 = (-half, half)
        
        if p > 0:
            if shape == 1 and target_shape == 2: # Cube to Pyramid
                p1 = (p1[0] * (1-p) + t1[0] * p, p1[1])
                p2 = (p2[0] * (1-p) + t1[0] * p, p2[1])
                points = [p1, p2, p3, p4]
            elif shape == 2 and target_shape == 1: # Pyramid to Cube
                tp1 = (t1[0] * (1-p) + p1[0] * p, t1[1])
                tp2 = (t1[0] * (1-p) + p2[0] * p, t1[1])
                points = [tp1, tp2, p3, p4]
            else: p = 0
        
        if p == 0:
            points = [p1, p2, p3, p4] if shape == 1 else [t1, t2, t3]

        if shape == 2 or (p > 0 and target_shape == 2):
            angle = math.radians(rotation * 90)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            points = [(pt[0] * cos_a - pt[1] * sin_a, pt[0] * sin_a + pt[1] * cos_a) for pt in points]
        
        final_points = [(int(pt[0] + px), int(pt[1] + py)) for pt in points]
        pygame.gfxdraw.aapolygon(surface, final_points, color)
        pygame.gfxdraw.filled_polygon(surface, final_points, color)

    def _render_ui(self):
        score_text = self.font_large.render(f"{int(self.score):05d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        remaining_seconds = max(0, (self.MAX_STEPS - self.steps) / 60)
        timer_text = self.font_large.render(f"{remaining_seconds:.1f}s", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))
        
        is_on_beat = self.beat_phase < 0.1 or self.beat_phase > 0.9
        beat_color = self.COLOR_BEAT_GOOD if is_on_beat else self.COLOR_BEAT_BAD
        
        pulse = (math.sin(self.beat_phase * 2 * math.pi) + 1) / 2
        pulse_radius = 15 + pulse * 10
        
        center_x, bottom_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 30
        
        pygame.gfxdraw.filled_circle(self.screen, center_x, bottom_y, int(pulse_radius), (*beat_color, 50))
        pygame.gfxdraw.aacircle(self.screen, center_x, bottom_y, int(pulse_radius), beat_color)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            
            is_win = np.all(self.grid > 0)
            end_text_str = "STRUCTURE COMPLETE" if is_win else "GAME OVER"
            color = self.COLOR_BEAT_GOOD if is_win else self.COLOR_PYRAMID
            end_text = self.font_large.render(end_text_str, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def _validate_implementation(self):
        # This method is optional and is for the developer's convenience.
        # It is not called by the evaluation system.
        try:
            # Test action space
            assert self.action_space.shape == (3,)
            assert self.action_space.nvec.tolist() == [5, 2, 2]
            
            # Test observation space
            obs, _ = self.reset()
            assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert obs.dtype == np.uint8
            
            # Test step
            test_action = self.action_space.sample()
            obs, reward, term, trunc, info = self.step(test_action)
            assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert isinstance(reward, (int, float))
            assert isinstance(term, bool)
            assert trunc == False
            assert isinstance(info, dict)
            
            print("✓ Implementation validated successfully")
        except Exception as e:
            print(f"✗ Implementation validation failed: {e}")


if __name__ == '__main__':
    # This block allows you to run the environment locally for testing.
    # It will not be executed by the evaluation system.
    # It requires pygame to be installed with display support.
    
    # Re-enable display for local testing
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("StackaBlock")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
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
        total_reward += reward

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}. Score: {info['score']:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(60)
        
    env.close()