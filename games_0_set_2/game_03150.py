
# Generated: 2025-08-27T22:30:32.362889
# Source Brief: brief_03150.md
# Brief Index: 3150

        
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
        "Controls: Use arrow keys (↑↓←→) to move. Collect all gems before time expires."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a grid to collect all the gems against the clock. Every move counts!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 10
    GRID_SIZE = 400
    CELL_SIZE = GRID_SIZE // GRID_WIDTH
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_SIZE) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_SIZE) // 2
    
    TIME_LIMIT = 60
    NUM_GEMS = 10

    # --- Colors ---
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (30, 45, 75)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (255, 255, 255, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_WIN_TEXT = (150, 255, 150)
    COLOR_LOSE_TEXT = (255, 150, 150)
    
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
        (255, 160, 80),  # Orange
        (80, 255, 255),  # Cyan
        (255, 80, 255),  # Magenta
        (255, 180, 200), # Pink
        (180, 255, 80),  # Lime
    ]

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        
        self.player_pos = (0, 0)
        self.gems = []
        self.particles = []
        self.time_remaining = 0
        self.gems_collected = 0
        self.steps = 0
        self.game_over = False
        self.win_status = ""
        
        self.reset()
        
        # self.validate_implementation() # Optional: Call to check compliance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.time_remaining = self.TIME_LIMIT
        self.gems_collected = 0
        self.game_over = False
        self.win_status = ""
        self.particles = []

        self.player_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)

        all_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        all_coords.remove(self.player_pos)
        
        chosen_indices = self.np_random.choice(len(all_coords), self.NUM_GEMS, replace=False)
        gem_positions = [all_coords[i] for i in chosen_indices]
        
        self.gems = []
        for i, pos in enumerate(gem_positions):
            self.gems.append({
                "pos": pos,
                "color": self.GEM_COLORS[i % len(self.GEM_COLORS)]
            })
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        self.time_remaining -= 1
        
        reward = -0.1 # Per-step penalty

        # --- Handle Movement ---
        px, py = self.player_pos
        if movement == 1 and py > 0: # Up
            self.player_pos = (px, py - 1)
        elif movement == 2 and py < self.GRID_HEIGHT - 1: # Down
            self.player_pos = (px, py + 1)
        elif movement == 3 and px > 0: # Left
            self.player_pos = (px - 1, py)
        elif movement == 4 and px < self.GRID_WIDTH - 1: # Right
            self.player_pos = (px + 1, py)

        # --- Handle Gem Collection ---
        gem_to_remove = None
        for gem in self.gems:
            if self.player_pos == gem["pos"]:
                gem_to_remove = gem
                break
        
        if gem_to_remove:
            # sfx: gem_collect.wav
            self.gems.remove(gem_to_remove)
            self.gems_collected += 1
            reward += 1.0
            self._create_sparkles(gem_to_remove["pos"], gem_to_remove["color"])

        # --- Check Termination ---
        terminated = False
        if self.gems_collected == self.NUM_GEMS:
            # sfx: win_sound.wav
            terminated = True
            self.game_over = True
            self.win_status = "YOU WIN!"
            reward += 100.0
        elif self.time_remaining <= 0:
            # sfx: lose_sound.wav
            terminated = True
            self.game_over = True
            self.win_status = "TIME'S UP!"
            reward += -10.0
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._update_and_draw_particles()
        self._draw_grid()
        self._draw_gems()
        self._draw_player()
        self._draw_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.gems_collected,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
        }
        
    def _grid_to_screen(self, gx, gy):
        sx = self.GRID_X_OFFSET + gx * self.CELL_SIZE + self.CELL_SIZE // 2
        sy = self.GRID_Y_OFFSET + gy * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(sx), int(sy)

    def _draw_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            start_pos = (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET)
            end_pos = (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + y * self.CELL_SIZE)
            end_pos = (self.GRID_X_OFFSET + self.GRID_SIZE, self.GRID_Y_OFFSET + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _draw_gems(self):
        gem_size = int(self.CELL_SIZE * 0.6)
        offset = (self.CELL_SIZE - gem_size) // 2
        for gem in self.gems:
            sx, sy = self._grid_to_screen(gem["pos"][0], gem["pos"][1])
            rect = pygame.Rect(sx - gem_size // 2, sy - gem_size // 2, gem_size, gem_size)
            pygame.draw.rect(self.screen, gem["color"], rect, border_radius=3)
            # Add a slight inner highlight
            highlight_color = tuple(min(255, c + 60) for c in gem["color"])
            pygame.draw.rect(self.screen, highlight_color, (rect.x + 2, rect.y + 2, 4, 4), border_radius=1)


    def _draw_player(self):
        sx, sy = self._grid_to_screen(self.player_pos[0], self.player_pos[1])
        radius = int(self.CELL_SIZE * 0.35)
        
        # Glow effect
        glow_radius = int(radius * 1.8)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (sx - glow_radius, sy - glow_radius))

        # Player circle with anti-aliasing
        pygame.gfxdraw.aacircle(self.screen, sx, sy, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius, self.COLOR_PLAYER)
        
    def _create_sparkles(self, pos, color):
        sx, sy = self._grid_to_screen(pos[0], pos[1])
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': sx, 'y': sy,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'lifespan': self.np_random.integers(15, 30),
                'max_lifespan': 30,
                'color': color
            })

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                active_particles.append(p)
                
                # Fade out effect
                alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
                current_color = p['color'] + (alpha,)
                
                # Draw particle as a small line
                end_x = p['x'] + p['vx'] * 2
                end_y = p['y'] + p['vy'] * 2
                
                temp_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                pygame.draw.line(temp_surf, current_color, (p['x'], p['y']), (end_x, end_y), 2)
                self.screen.blit(temp_surf, (0, 0))

        self.particles = active_particles

    def _draw_ui(self):
        # --- Timer Display ---
        time_text = f"Time: {self.time_remaining}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 20, 10))

        # --- Gems Display ---
        gems_text = f"Gems: {self.gems_collected} / {self.NUM_GEMS}"
        gems_surf = self.font_ui.render(gems_text, True, self.COLOR_TEXT)
        self.screen.blit(gems_surf, (self.SCREEN_WIDTH - gems_surf.get_width() - 20, 35))
        
        # --- Game Over Message ---
        if self.game_over:
            text_color = self.COLOR_WIN_TEXT if self.win_status == "YOU WIN!" else self.COLOR_LOSE_TEXT
            
            # Semi-transparent background
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            # Game over text
            go_surf = self.font_game_over.render(self.win_status, True, text_color)
            go_rect = go_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(go_surf, go_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up a window to display the game
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0.0

    print("\n" + "="*30)
    print("      MANUAL PLAY TEST")
    print("="*30)
    print(env.game_description)
    print(env.user_guide)
    print("Press ESC or close window to quit.")
    print("="*30 + "\n")

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                # We only process one key press to generate an action for the step
                # This is because auto_advance is False
                if not done:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_UP]:
                        action[0] = 1
                    elif keys[pygame.K_DOWN]:
                        action[0] = 2
                    elif keys[pygame.K_LEFT]:
                        action[0] = 3
                    elif keys[pygame.K_RIGHT]:
                        action[0] = 4
                    else: # any other key press also advances frame
                        action[0] = 0

                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += reward

                    print(f"Step: {info['steps']}, Time: {info['time_remaining']}, Gems: {info['score']}, Reward: {reward:.2f}, Total: {total_reward:.2f}")

                    if done:
                        print("\n--- GAME OVER ---")
                        print(f"Final Score (Gems): {info['score']}")
                        print(f"Total Reward: {total_reward:.2f}")
                        print("Press R to reset.")

                elif event.key == pygame.K_r:
                    print("\n--- RESETTING GAME ---")
                    obs, info = env.reset()
                    done = False
                    total_reward = 0.0
                    print(f"Game reset. Initial state: {info}")


        # Render the observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit the loop to 30 FPS

    env.close()