
# Generated: 2025-08-27T21:26:25.522389
# Source Brief: brief_02784.md
# Brief Index: 2784

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Particle:
    """A simple particle for effects."""
    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.life = random.randint(10, 20)  # Lifespan in frames

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = max(0, min(255, int(255 * (self.life / 20))))
            # Create a temporary surface for transparency
            particle_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            particle_surf.fill((*self.color, alpha))
            surface.blit(particle_surf, (int(self.x), int(self.y)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Press Space to paint. Press Shift to cycle colors."
    )

    game_description = (
        "Recreate the target image on the pixel grid before time runs out. Select colors and paint pixels to match the reference image."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Configuration ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 20, 15
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        # --- Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 22)
        
        # --- Colors ---
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_FRAME = (60, 68, 87)
        self.COLOR_TEXT = (210, 220, 230)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_BLANK = (40, 45, 58)
        self.PALETTE = [
            (255, 87, 87), (255, 168, 87), (245, 255, 87), (138, 255, 87),
            (87, 255, 204), (87, 173, 255), (133, 87, 255), (255, 87, 245),
            (230, 230, 230), (100, 100, 100)
        ]
        
        # --- Layout ---
        self.CANVAS_PIXEL_SIZE = 16
        self.CANVAS_W = self.GRID_W * self.CANVAS_PIXEL_SIZE
        self.CANVAS_H = self.GRID_H * self.CANVAS_PIXEL_SIZE
        self.CANVAS_X = (self.WIDTH - self.CANVAS_W) // 2
        self.CANVAS_Y = (self.HEIGHT - self.CANVAS_H) // 2 + 10

        self.TARGET_PIXEL_SIZE = 5
        self.TARGET_W = self.GRID_W * self.TARGET_PIXEL_SIZE
        self.TARGET_H = self.GRID_H * self.TARGET_PIXEL_SIZE
        self.TARGET_X = 20
        self.TARGET_Y = 35

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.target_image = None
        self.player_canvas = None
        self.cursor_pos = [0, 0]
        self.selected_color_idx = 0
        self.accuracy = 0.0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_color_idx = 0
        self.accuracy = 0.0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []

        # Generate new target image
        self.target_image = self.np_random.integers(0, len(self.PALETTE), size=(self.GRID_H, self.GRID_W))
        
        # Initialize blank player canvas
        self.player_canvas = np.full((self.GRID_H, self.GRID_W), -1, dtype=int) # -1 for blank

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        reward = 0
        self.steps += 1

        # --- Unpack Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        # --- Update Game Logic ---
        # 1. Move Cursor
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] %= self.GRID_W
        self.cursor_pos[1] %= self.GRID_H

        # 2. Cycle Color
        if shift_pressed:
            self.selected_color_idx = (self.selected_color_idx + 1) % len(self.PALETTE)
            # sfx: color_cycle.wav

        # 3. Paint Pixel
        if space_pressed:
            cx, cy = self.cursor_pos
            
            old_color_idx = self.player_canvas[cy][cx]
            target_color_idx = self.target_image[cy][cx]
            new_color_idx = self.selected_color_idx

            is_correct_before = (old_color_idx == target_color_idx)
            is_correct_after = (new_color_idx == target_color_idx)

            if not is_correct_before and is_correct_after:
                reward += 0.1  # Corrected a pixel
                # sfx: paint_correct.wav
            elif is_correct_before and not is_correct_after:
                reward -= 0.1  # Made a mistake on a correct pixel
                # sfx: paint_wrong.wav
            
            if old_color_idx != new_color_idx:
                self.player_canvas[cy][cx] = new_color_idx
                self._create_particles(cx, cy, self.PALETTE[new_color_idx])
                # sfx: paint.wav
        
        self.score += reward

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

        # --- Check Termination ---
        self.accuracy = self._calculate_accuracy()
        terminated = False
        if self.accuracy >= 1.0:
            reward += 10.0
            self.score += 10.0
            terminated = True
            self.game_over = True
            # sfx: win.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            # sfx: lose.wav

        # --- Update previous action states ---
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _calculate_accuracy(self):
        matches = np.sum(self.player_canvas == self.target_image)
        total_pixels = self.GRID_W * self.GRID_H
        return matches / total_pixels if total_pixels > 0 else 0.0

    def _create_particles(self, grid_x, grid_y, color):
        world_x = self.CANVAS_X + grid_x * self.CANVAS_PIXEL_SIZE + self.CANVAS_PIXEL_SIZE / 2
        world_y = self.CANVAS_Y + grid_y * self.CANVAS_PIXEL_SIZE + self.CANVAS_PIXEL_SIZE / 2
        for _ in range(15):
            self.particles.append(Particle(world_x, world_y, color))
    
    def _render_pixel_grid(self, surface, grid_data, pixel_size, offset, is_player_canvas=False):
        ox, oy = offset
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                color_idx = grid_data[y, x]
                color = self.PALETTE[color_idx] if color_idx != -1 else self.COLOR_BLANK
                rect = (ox + x * pixel_size, oy + y * pixel_size, pixel_size, pixel_size)
                pygame.draw.rect(surface, color, rect)
                if is_player_canvas: # Add a subtle grid
                     pygame.draw.rect(surface, self.COLOR_FRAME, rect, 1)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # --- Render UI Panels ---
        pygame.draw.rect(self.screen, self.COLOR_FRAME, (self.CANVAS_X-2, self.CANVAS_Y-2, self.CANVAS_W+4, self.CANVAS_H+4), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_FRAME, (self.TARGET_X-2, self.TARGET_Y-2, self.TARGET_W+4, self.TARGET_H+4), border_radius=5)
        
        # --- Render Grids ---
        self._render_text("TARGET", (self.TARGET_X, self.TARGET_Y - 18), self.font_title, self.COLOR_TEXT)
        self._render_pixel_grid(self.screen, self.target_image, self.TARGET_PIXEL_SIZE, (self.TARGET_X, self.TARGET_Y))
        
        self._render_text("CANVAS", (self.CANVAS_X, self.CANVAS_Y - 22), self.font_main, self.COLOR_TEXT)
        self._render_pixel_grid(self.screen, self.player_canvas, self.CANVAS_PIXEL_SIZE, (self.CANVAS_X, self.CANVAS_Y), is_player_canvas=True)

        # --- Render Cursor ---
        cursor_x = self.CANVAS_X + self.cursor_pos[0] * self.CANVAS_PIXEL_SIZE
        cursor_y = self.CANVAS_Y + self.cursor_pos[1] * self.CANVAS_PIXEL_SIZE
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y, self.CANVAS_PIXEL_SIZE, self.CANVAS_PIXEL_SIZE), 3, border_radius=2)

        # --- Render Particles ---
        for p in self.particles:
            p.draw(self.screen)
        
        # --- Render Color Palette ---
        palette_w = len(self.PALETTE) * 28
        palette_x = (self.WIDTH - palette_w) / 2
        for i, color in enumerate(self.PALETTE):
            rect = pygame.Rect(palette_x + i * 28, self.HEIGHT - 40, 24, 24)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            if i == self.selected_color_idx:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2, border_radius=4)

        # --- Render UI Text ---
        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_seconds = time_left / self.FPS
        timer_text = f"TIME: {time_seconds:.1f}"
        self._render_text(timer_text, (self.WIDTH - 130, 35), self.font_main, self.COLOR_TEXT)

        # Accuracy
        acc_text = f"ACC: {self.accuracy:.1%}"
        self._render_text(acc_text, (self.WIDTH - 130, 65), self.font_main, self.COLOR_TEXT)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = "COMPLETE!" if self.accuracy >= 1.0 else "TIME'S UP!"
            self._render_text(msg, (self.WIDTH / 2, self.HEIGHT / 2), pygame.font.Font(None, 60), self.COLOR_CURSOR, center=True)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_text(self, text, pos, font, color, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "accuracy": self.accuracy,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Mapping from Pygame keys to action components
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # We need a window to capture key presses
    pygame.display.set_caption("Pixel Art Painter")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        # Construct the action based on keyboard state
        movement_action = 0
        space_action = 0
        shift_action = 0

        keys = pygame.key.get_pressed()
        for key, move in key_map.items():
            if keys[key]:
                movement_action = move
                break # Prioritize one move direction
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Accuracy: {info['accuracy']:.1%}")
            pygame.time.wait(2000) # Pause for 2 seconds before reset
            obs, info = env.reset()
        
        env.clock.tick(env.FPS)

    env.close()