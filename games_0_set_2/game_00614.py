
# Generated: 2025-08-27T14:13:06.907109
# Source Brief: brief_00614.md
# Brief Index: 614

        
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

    user_guide = (
        "Controls: Use arrow keys to move on the grid. Collect 50 gems before the time runs out."
    )

    game_description = (
        "Navigate an isometric grid to collect vibrant gems. Plan your path efficiently to beat the clock in this fast-paced puzzle challenge."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 15, 15
        self.TARGET_GEMS = 50
        self.INITIAL_GEM_COUNT = 75 # More than target to ensure it's possible
        self.MAX_STEPS = 1000
        
        # --- Visuals ---
        self.TILE_WIDTH = 32
        self.TILE_HEIGHT = 16
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

        # --- Colors ---
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID_LINE = (50, 55, 65)
        self.COLOR_TILE = (40, 44, 52)
        self.COLOR_PLAYER = (255, 215, 0) # Gold
        self.COLOR_PLAYER_GLOW = (255, 215, 0, 50)
        self.GEM_COLORS = [(255, 0, 80), (0, 255, 150), (0, 150, 255)] # Magenta, Green, Blue
        self.COLOR_UI_BG = (15, 18, 26, 200)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_TIMER_GREEN = (0, 200, 83)
        self.COLOR_TIMER_RED = (213, 0, 0)

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
        try:
            self.font_main = pygame.font.Font(None, 24)
            self.font_title = pygame.font.Font(None, 32)
        except IOError:
            self.font_main = pygame.font.SysFont("Arial", 22)
            self.font_title = pygame.font.SysFont("Arial", 30)
        
        # --- State Variables ---
        self.player_pos = [0, 0]
        self.gems = set()
        self.particles = []
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.time_remaining = self.MAX_STEPS
        self.game_over = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.time_remaining = self.MAX_STEPS
        self.game_over = False
        self.particles = []
        
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self._spawn_gems()
        
        return self._get_observation(), self._get_info()
    
    def _spawn_gems(self):
        self.gems.clear()
        possible_locations = []
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x, y) != tuple(self.player_pos):
                    possible_locations.append((x, y))
        
        num_gems = min(self.INITIAL_GEM_COUNT, len(possible_locations))
        gem_indices = self.np_random.choice(len(possible_locations), num_gems, replace=False)
        
        for i in gem_indices:
            self.gems.add(possible_locations[i])

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        reward = -0.01 # Cost per step
        
        # --- Update Game Logic ---
        self._update_player_position(movement)
        self._update_particles()
        
        # Check for gem collection
        if tuple(self.player_pos) in self.gems:
            # SFX: Gem collect sound
            self.gems.remove(tuple(self.player_pos))
            self.gems_collected += 1
            reward += 1
            
            if self.gems_collected == self.TARGET_GEMS:
                reward += 5 # Bonus for last gem
                
            self._spawn_particles(self.player_pos)

        # Update time and step count
        self.time_remaining -= 1
        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        if self.gems_collected >= self.TARGET_GEMS:
            # SFX: Win sound
            terminated = True
            self.game_over = True
            reward += 50 # Win bonus
        elif self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            # SFX: Lose sound
            terminated = True
            self.game_over = True
            reward -= 50 # Loss penalty
            
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _update_player_position(self, movement):
        dx, dy = 0, 0
        if movement == 1: dy = -1 # Up
        elif movement == 2: dy = 1 # Down
        elif movement == 3: dx = -1 # Left
        elif movement == 4: dx = 1 # Right

        if dx != 0 or dy != 0:
            # SFX: Player move sound
            new_x = self.player_pos[0] + dx
            new_y = self.player_pos[1] + dy
            
            if 0 <= new_x < self.GRID_WIDTH and 0 <= new_y < self.GRID_HEIGHT:
                self.player_pos = [new_x, new_y]

    def _spawn_particles(self, grid_pos):
        screen_pos = self._grid_to_screen(grid_pos[0], grid_pos[1])
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 40)
            radius = self.np_random.uniform(2, 5)
            color = random.choice(self.GEM_COLORS)
            self.particles.append({
                "pos": list(screen_pos),
                "vel": vel,
                "lifespan": lifespan,
                "max_life": lifespan,
                "radius": radius,
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid tiles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                screen_x, screen_y = self._grid_to_screen(x, y)
                points = [
                    (screen_x, screen_y),
                    (screen_x + self.TILE_WIDTH / 2, screen_y + self.TILE_HEIGHT / 2),
                    (screen_x, screen_y + self.TILE_HEIGHT),
                    (screen_x - self.TILE_WIDTH / 2, screen_y + self.TILE_HEIGHT / 2)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_TILE)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID_LINE)
        
        # Render gems
        for gx, gy in self.gems:
            self._draw_iso_gem(gx, gy)
            
        # Render player
        self._draw_player()
        
        # Render particles
        for p in self.particles:
            life_ratio = p["lifespan"] / p["max_life"]
            current_radius = int(p["radius"] * life_ratio)
            if current_radius > 0:
                pos = (int(p["pos"][0]), int(p["pos"][1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], current_radius, p["color"])

    def _draw_iso_gem(self, x, y):
        screen_x, screen_y = self._grid_to_screen(x, y)
        center_y = screen_y + self.TILE_HEIGHT / 2
        
        # Pulsing effect
        pulse = math.sin(pygame.time.get_ticks() * 0.005 + x + y) * 2
        size = 6 + pulse
        
        color_index = (x + y) % len(self.GEM_COLORS)
        color = self.GEM_COLORS[color_index]
        
        points = [
            (screen_x, center_y - size / 2),
            (screen_x + size / 2, center_y),
            (screen_x, center_y + size / 2),
            (screen_x - size / 2, center_y)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, (255, 255, 255, 150))
        
    def _draw_player(self):
        px, py = self.player_pos
        screen_x, screen_y = self._grid_to_screen(px, py)
        center_y = screen_y + self.TILE_HEIGHT / 2
        
        # Glow effect
        glow_radius = 15 + math.sin(pygame.time.get_ticks() * 0.008) * 3
        pygame.gfxdraw.filled_circle(self.screen, int(screen_x), int(center_y), int(glow_radius), self.COLOR_PLAYER_GLOW)
        
        # Main body
        size = 8
        points = [
            (screen_x, center_y - size),
            (screen_x + size, center_y),
            (screen_x, center_y + size),
            (screen_x - size, center_y)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, (255, 255, 255))

    def _render_ui(self):
        # UI background panel
        ui_panel = pygame.Surface((self.WIDTH, 60), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, self.HEIGHT - 60))

        # Gem count
        gem_text = self.font_title.render(f"GEMS: {self.gems_collected} / {self.TARGET_GEMS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(gem_text, (20, self.HEIGHT - 45))

        # Timer
        time_ratio = max(0, self.time_remaining / self.MAX_STEPS)
        
        # Timer text
        time_text = self.font_title.render("TIME", True, self.COLOR_UI_TEXT)
        text_rect = time_text.get_rect(right=self.WIDTH - 20, centery=self.HEIGHT - 45)
        self.screen.blit(time_text, text_rect)

        # Timer bar
        bar_max_width = 200
        bar_width = int(bar_max_width * time_ratio)
        bar_height = 15
        bar_x = text_rect.left - bar_max_width - 10
        bar_y = self.HEIGHT - 45 - bar_height / 2

        # Interpolate color from green to red
        r = int(self.COLOR_TIMER_RED[0] * (1 - time_ratio) + self.COLOR_TIMER_GREEN[0] * time_ratio)
        g = int(self.COLOR_TIMER_RED[1] * (1 - time_ratio) + self.COLOR_TIMER_GREEN[1] * time_ratio)
        bar_color = (r, g, 0)
        
        pygame.draw.rect(self.screen, (50, 55, 65), (bar_x, bar_y, bar_max_width, bar_height))
        if bar_width > 0:
            pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, bar_width, bar_height))

    def _grid_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH / 2
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
            "time_remaining": self.time_remaining,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Override screen for display
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Gem Collector")

    action = env.action_space.sample()
    action[0] = 0 # Start with no-op

    print(env.user_guide)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    action[0] = 0
            # Since auto_advance is False, we only step on keydown
            if event.type == pygame.KEYDOWN:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Gems: {info['gems_collected']}, Terminated: {terminated}")
                # Reset action to no-op after each step
                action[0] = 0
        
        # Render the environment to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        env.screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for human play

    print("Game Over!")
    env.close()