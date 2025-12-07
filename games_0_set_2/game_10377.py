import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()
pygame.font.init()

# Helper class for the rain splash visual effect
class RainSplash:
    def __init__(self, x, y, max_radius=30, duration=20):
        self.x = x
        self.y = y
        self.max_radius = max_radius
        self.duration = duration
        self.life = duration
        self.color = (100, 150, 255)

    def update(self):
        self.life -= 1
        return self.life > 0

    def draw(self, surface):
        progress = (self.duration - self.life) / self.duration
        current_radius = int(self.max_radius * progress)
        alpha = int(255 * (1 - progress))
        
        # Use gfxdraw for anti-aliased circles
        if current_radius > 0:
            pygame.gfxdraw.aacircle(surface, self.x, self.y, current_radius, (*self.color, alpha))

# Helper class for individual raindrops
class RainParticle:
    def __init__(self, x, y, tile_center_y):
        self.x = x
        self.y = y
        self.start_y = y
        self.end_y = tile_center_y + random.uniform(-5, 5)
        self.vel_y = random.uniform(4, 8)
        self.life = (self.end_y - self.start_y) / self.vel_y if self.vel_y > 0 else 1
        self.max_life = self.life if self.life > 0 else 1
        self.color = (150, 180, 255)
        self.length = random.uniform(5, 10)

    def update(self):
        self.y += self.vel_y
        self.life -= 1
        return self.life > 0

    def draw(self, surface):
        alpha = int(255 * (self.life / self.max_life))
        # Create a temporary surface for transparency
        line_surf = pygame.Surface((2, self.length + 1), pygame.SRCALPHA)
        pygame.draw.line(line_surf, (*self.color, alpha), (1, 0), (1, self.length), 2)
        surface.blit(line_surf, (self.x - 1, self.y))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Restore a barren grid by summoning rain to increase the fertility of the land. "
        "Turn the entire desert green before time runs out."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor. Press space to summon a rain shower on the selected tile."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30 # For simulation speed
        self.GRID_ROWS = 10
        self.GRID_COLS = 10
        self.TILE_SIZE = 36
        self.GRID_WIDTH = self.GRID_COLS * self.TILE_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.TILE_SIZE
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2
        
        self.TIME_LIMIT_SECONDS = 180
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_DESERT_LOW = (139, 115, 85) # Sandy brown
        self.COLOR_FERTILE_HIGH = (60, 179, 113) # Medium sea green
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_GRID_LINE = (40, 50, 60)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("sans-serif", 24, bold=True)
        self.font_tile = pygame.font.SysFont("sans-serif", 14, bold=True)
        
        # Initialize state variables to be defined in reset()
        self.grid_fertility = None
        self.cursor_pos = None
        self.visual_cursor_pos = None
        self.time_remaining = None
        self.steps = None
        self.previous_space_state = None
        self.visual_effects = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid_fertility = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.float32)
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        
        # Set visual cursor to initial logical position
        target_x = self.GRID_OFFSET_X + self.cursor_pos[1] * self.TILE_SIZE
        target_y = self.GRID_OFFSET_Y + self.cursor_pos[0] * self.TILE_SIZE
        self.visual_cursor_pos = [float(target_x), float(target_y)]

        self.time_remaining = float(self.TIME_LIMIT_SECONDS)
        self.steps = 0
        self.previous_space_state = 0
        self.visual_effects = [] # For rain splashes and particles
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- Handle Input ---
        self._handle_movement(movement)
        
        space_pressed = space_held and not self.previous_space_state
        if space_pressed:
            reward += self._trigger_rain()

        self.previous_space_state = space_held

        # --- Update Game State ---
        self.steps += 1
        self.time_remaining -= 1.0 / self.FPS
        self._update_visuals()

        # --- Check Termination ---
        terminated = False
        win_condition = np.all(self.grid_fertility >= 100)
        time_out_condition = self.time_remaining <= 0
        step_limit_reached = self.steps >= self.MAX_STEPS

        if win_condition:
            reward += 100.0
            terminated = True
        elif time_out_condition:
            reward -= 100.0
            terminated = True
        
        truncated = step_limit_reached

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_movement(self, movement):
        row, col = self.cursor_pos
        if movement == 1: # Up
            self.cursor_pos[0] = (row - 1) % self.GRID_ROWS
        elif movement == 2: # Down
            self.cursor_pos[0] = (row + 1) % self.GRID_ROWS
        elif movement == 3: # Left
            self.cursor_pos[1] = (col - 1) % self.GRID_COLS
        elif movement == 4: # Right
            self.cursor_pos[1] = (col + 1) % self.GRID_COLS

    def _trigger_rain(self):
        row, col = self.cursor_pos
        
        fertility_before = np.sum(self.grid_fertility)
        tiles_at_100_before = np.sum(self.grid_fertility >= 100)

        # Apply fertility to target tile
        self.grid_fertility[row, col] = min(100.0, self.grid_fertility[row, col] + 20)
        
        # Apply fertility to adjacent tiles
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS:
                self.grid_fertility[nr, nc] = min(100.0, self.grid_fertility[nr, nc] + 10)

        # --- Calculate Reward ---
        fertility_after = np.sum(self.grid_fertility)
        tiles_at_100_after = np.sum(self.grid_fertility >= 100)
        
        fertility_increase_reward = (fertility_after - fertility_before) * 0.1
        new_fertile_tiles_reward = float(tiles_at_100_after - tiles_at_100_before)
        
        # --- Create Visual Effects ---
        tile_center_x = self.GRID_OFFSET_X + col * self.TILE_SIZE + self.TILE_SIZE // 2
        tile_center_y = self.GRID_OFFSET_Y + row * self.TILE_SIZE + self.TILE_SIZE // 2
        self.visual_effects.append(RainSplash(tile_center_x, tile_center_y, max_radius=self.TILE_SIZE, duration=self.FPS // 2))
        
        for _ in range(15): # Spawn 15 raindrops
            particle_x = tile_center_x + random.uniform(-self.TILE_SIZE//2, self.TILE_SIZE//2)
            particle_y = tile_center_y - self.TILE_SIZE # Start above the tile
            self.visual_effects.append(RainParticle(int(particle_x), int(particle_y), tile_center_y))

        return fertility_increase_reward + new_fertile_tiles_reward

    def _update_visuals(self):
        # Interpolate visual cursor position
        target_x = self.GRID_OFFSET_X + self.cursor_pos[1] * self.TILE_SIZE
        target_y = self.GRID_OFFSET_Y + self.cursor_pos[0] * self.TILE_SIZE
        self.visual_cursor_pos[0] += (target_x - self.visual_cursor_pos[0]) * 0.3
        self.visual_cursor_pos[1] += (target_y - self.visual_cursor_pos[1]) * 0.3

        # Update and filter visual effects
        self.visual_effects = [effect for effect in self.visual_effects if effect.update()]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                fertility = self.grid_fertility[r, c] / 100.0
                color = self._lerp_color(self.COLOR_DESERT_LOW, self.COLOR_FERTILE_HIGH, fertility)
                
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + c * self.TILE_SIZE,
                    self.GRID_OFFSET_Y + r * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, rect, 1)

                # Render fertility text
                text_surf = self.font_tile.render(f"{int(self.grid_fertility[r, c])}", True, self.COLOR_UI_TEXT)
                text_rect = text_surf.get_rect(center=rect.center)
                self.screen.blit(text_surf, text_rect)

        # Render cursor
        self._render_cursor()

        # Render visual effects
        for effect in self.visual_effects:
            effect.draw(self.screen)

    def _render_cursor(self):
        # Pulsating glow effect
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
        
        cursor_rect = pygame.Rect(
            int(self.visual_cursor_pos[0]),
            int(self.visual_cursor_pos[1]),
            self.TILE_SIZE, self.TILE_SIZE
        )
        
        # Draw a thicker, glowing border
        border_thickness = 2 + int(pulse * 2)
        glow_alpha = 150 + int(pulse * 105)
        
        # Create a temporary surface for the glowing rectangle
        glow_surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_CURSOR, glow_alpha), glow_surf.get_rect(), border_radius=4)
        
        # Blit the glow surf, then draw the sharp inner border
        self.screen.blit(glow_surf, cursor_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, border_thickness, border_radius=4)


    def _render_ui(self):
        # Timer display
        mins, secs = divmod(max(0, self.time_remaining), 60)
        timer_text = f"TIME: {int(mins):02d}:{int(secs):02d}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 20, 10))

        # Total fertility display
        total_fertility = np.sum(self.grid_fertility)
        max_fertility = self.GRID_ROWS * self.GRID_COLS * 100
        fertility_percent = (total_fertility / max_fertility) * 100 if max_fertility > 0 else 0
        score_text = f"FERTILITY: {fertility_percent:.1f}%"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (20, 10))

    def _get_info(self):
        total_fertility = np.sum(self.grid_fertility)
        max_fertility = self.GRID_ROWS * self.GRID_COLS * 100
        return {
            "score": total_fertility,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "total_fertility_percent": (total_fertility / max_fertility) * 100 if max_fertility > 0 else 0,
        }

    def _lerp_color(self, color1, color2, t):
        t = max(0, min(1, t))
        return tuple(int(a + (b - a) * t) for a, b in zip(color1, color2))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Terraform")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    
    # Game loop for manual play
    while not terminated and not truncated:
        # Action defaults
        movement = 0 # none
        space_held = 0 # released
        shift_held = 0 # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # Get key presses for manual control
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.0f}, Terminated: {terminated}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    print("Game Over!")
    print(f"Final Info: {info}")
    env.close()