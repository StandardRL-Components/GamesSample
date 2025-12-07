
# Generated: 2025-08-28T06:50:51.346330
# Source Brief: brief_03059.md
# Brief Index: 3059

        
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
        "Controls: Use arrow keys to select a plot. Press Shift to water, and Space to plant a seed."
    )

    game_description = (
        "Cultivate a thriving grid garden. Plant seeds and keep them watered to help all 10 plants "
        "reach full maturity. Don't let any plant wither away!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 64)

        # Game Constants
        self.max_steps = 1000
        self.grid_cols, self.grid_rows = 5, 2
        self.num_plots = self.grid_cols * self.grid_rows

        # Plant Constants
        self.plant_max_health = 100
        self.plant_growth_thresholds = [30, 60, 90]
        self.plant_max_growth_stage = 3
        self.thirsty_threshold = 50

        # Resource Constants
        self.initial_seeds = 10
        self.initial_water = 10
        self.max_water = 20
        self.water_replenish_amount = 3

        # Color Palette
        self.color_bg = (20, 30, 40)
        self.color_grid_bg = (40, 50, 60)
        self.color_grid_line = (60, 70, 80)
        self.color_cursor = (255, 215, 0)
        self.color_plant_healthy = (40, 200, 80)
        self.color_plant_thirsty = (200, 180, 50)
        self.color_plant_dead = (100, 80, 60)
        self.color_flower = (255, 50, 150)
        self.color_water = (50, 150, 255)
        self.color_seed = (140, 100, 80)
        self.color_ui_bg = (30, 40, 50, 180)
        self.color_ui_text = (230, 240, 255)

        # Grid layout
        self.plot_size = 80
        self.plot_padding = 15
        grid_width = self.grid_cols * (self.plot_size + self.plot_padding) - self.plot_padding
        grid_height = self.grid_rows * (self.plot_size + self.plot_padding) - self.plot_padding
        self.grid_offset_x = (self.screen_width - grid_width) // 2
        self.grid_offset_y = (self.screen_height - grid_height) // 2 + 30

        # State variables (initialized in reset)
        self.plots = []
        self.cursor_pos = 0
        self.water_level = 0
        self.seeds_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.wither_rate = 0.0
        self.particles = []
        self.rng = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.wither_rate = 1.0
        self.cursor_pos = 0
        self.water_level = self.initial_water
        self.seeds_left = self.initial_seeds
        self.plots = [{'state': 'empty', 'health': 0, 'growth_stage': 0} for _ in range(self.num_plots)]
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = self.game_over

        if not terminated:
            self._handle_input(action)
            action_reward = self._process_actions(action)
            reward += action_reward

            self.water_level = min(self.max_water, self.water_level + self.water_replenish_amount)

            growth_reward, plant_death = self._update_plants()
            reward += growth_reward

            if plant_death:
                terminated = True
                reward -= 100
                self.win_state = False
            elif self._check_win_condition():
                terminated = True
                reward += 100
                self.win_state = True

            self.steps += 1
            if self.steps >= self.max_steps:
                terminated = True

            if self.steps > 0 and self.steps % 200 == 0:
                self.wither_rate += 0.05
        
        self.score += reward
        self.game_over = terminated
        self._update_particles()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement = action[0]
        if movement == 1:  # Up
            self.cursor_pos = (self.cursor_pos - self.grid_cols + self.num_plots) % self.num_plots
        elif movement == 2:  # Down
            self.cursor_pos = (self.cursor_pos + self.grid_cols) % self.num_plots
        elif movement == 3:  # Left
            self.cursor_pos = (self.cursor_pos - 1 + self.num_plots) % self.num_plots
        elif movement == 4:  # Right
            self.cursor_pos = (self.cursor_pos + 1) % self.num_plots

    def _process_actions(self, action):
        space_held = action[1] == 1
        shift_held = action[2] == 1
        reward = 0
        plot = self.plots[self.cursor_pos]

        if space_held and plot['state'] == 'empty' and self.seeds_left > 0:
            # Plant action
            plot['state'] = 'planted'
            plot['health'] = self.plant_max_health / 2
            self.seeds_left -= 1
            # Sound: Plant seed
            pos = self._get_plot_center(self.cursor_pos)
            self._create_particles(pos[0], pos[1], self.color_seed, 20, "burst")
            reward += 0.5

        if shift_held and plot['state'] == 'planted' and self.water_level > 0:
            # Water action
            self.water_level -= 1
            is_full_health = plot['health'] >= self.plant_max_health
            plot['health'] = min(self.plant_max_health, plot['health'] + 25)
            # Sound: Water plant
            pos = self._get_plot_center(self.cursor_pos)
            self._create_particles(pos[0], pos[1] - 20, self.color_water, 15, "drip")
            if is_full_health:
                reward -= 0.2
            else:
                reward += 0.1
        
        return reward

    def _update_plants(self):
        total_growth_reward = 0
        plant_death = False
        for plot in self.plots:
            if plot['state'] == 'planted':
                plot['health'] -= self.wither_rate
                if plot['health'] <= 0:
                    plot['state'] = 'dead'
                    plant_death = True
                    # Sound: Plant dies
                    pos = self._get_plot_center(self.plots.index(plot))
                    self._create_particles(pos[0], pos[1], self.color_plant_dead, 30, "burst")

                new_stage = 0
                for i, threshold in enumerate(self.plant_growth_thresholds):
                    if plot['health'] >= threshold:
                        new_stage = i + 1
                
                if new_stage > plot['growth_stage']:
                    plot['growth_stage'] = new_stage
                    total_growth_reward += 1.0
                    # Sound: Plant grows
                    pos = self._get_plot_center(self.plots.index(plot))
                    self._create_particles(pos[0], pos[1], self.color_plant_healthy, 25, "upward_burst")
        return total_growth_reward, plant_death

    def _check_win_condition(self):
        if self.seeds_left > 0: return False
        planted_plots = [p for p in self.plots if p['state'] != 'empty']
        if len(planted_plots) != self.num_plots: return False
        return all(p['growth_stage'] == self.plant_max_growth_stage for p in planted_plots)

    def _get_observation(self):
        self.screen.fill(self.color_bg)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid()
        self._render_cursor()
        self._render_particles()

    def _render_grid(self):
        for i in range(self.num_plots):
            rect = self._get_plot_rect(i)
            pygame.draw.rect(self.screen, self.color_grid_bg, rect, border_radius=5)
            pygame.draw.rect(self.screen, self.color_grid_line, rect, width=2, border_radius=5)
            
            plot = self.plots[i]
            if plot['state'] != 'empty':
                self._render_plant(plot, rect)
            
            if plot['state'] == 'planted' and plot['health'] < self.thirsty_threshold:
                self._render_thirsty_icon(rect)

    def _render_plant(self, plot, rect):
        center_x, center_y = rect.center
        health_ratio = max(0, plot['health']) / self.plant_max_health
        
        if plot['state'] == 'dead':
            color = self.color_plant_dead
        else:
            color = self._lerp_color(self.color_plant_thirsty, self.color_plant_healthy, health_ratio)

        stage = plot['growth_stage']
        if stage == 0: # Sprout
            pygame.draw.circle(self.screen, color, (center_x, rect.bottom - 10), 5)
        elif stage == 1:
            pygame.draw.rect(self.screen, color, (center_x - 3, rect.bottom - 25, 6, 20))
            pygame.draw.circle(self.screen, color, (center_x, rect.bottom - 25), 8)
        elif stage == 2:
            pygame.draw.rect(self.screen, color, (center_x - 4, rect.bottom - 40, 8, 35))
            pygame.draw.circle(self.screen, color, (center_x - 10, rect.bottom - 25), 7)
            pygame.draw.circle(self.screen, color, (center_x + 10, rect.bottom - 25), 7)
            pygame.draw.circle(self.screen, color, (center_x, rect.bottom - 40), 10)
        elif stage == self.plant_max_growth_stage:
            pygame.draw.rect(self.screen, color, (center_x - 5, rect.bottom - 50, 10, 45))
            pygame.draw.circle(self.screen, color, (center_x - 15, rect.bottom - 20), 8)
            pygame.draw.circle(self.screen, color, (center_x + 15, rect.bottom - 20), 8)
            pygame.draw.circle(self.screen, self.color_flower, (center_x, rect.bottom - 50), 14)
            pygame.draw.circle(self.screen, self.color_cursor, (center_x, rect.bottom - 50), 5)

    def _render_thirsty_icon(self, rect):
        pos = (rect.centerx, rect.top + 12)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 6, self.color_water)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 6, self.color_water)
        
    def _render_cursor(self):
        rect = self._get_plot_rect(self.cursor_pos)
        pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
        alpha = int(155 + pulse * 100)
        
        # Create a temporary surface for the glowing border
        glow_surface = pygame.Surface((rect.width + 8, rect.height + 8), pygame.SRCALPHA)
        glow_rect = glow_surface.get_rect()
        
        # Draw the glowing border
        pygame.draw.rect(glow_surface, (*self.color_cursor, alpha), glow_rect, width=4, border_radius=8)
        
        self.screen.blit(glow_surface, (rect.left - 4, rect.top - 4))

    def _render_ui(self):
        ui_surf = pygame.Surface((self.screen_width, 60), pygame.SRCALPHA)
        ui_surf.fill(self.color_ui_bg)
        self.screen.blit(ui_surf, (0, 0))

        texts = [
            f"Score: {int(self.score)}",
            f"Water: {self.water_level}",
            f"Seeds: {self.seeds_left}",
            f"Day: {self.steps}/{self.max_steps}",
        ]
        spacing = self.screen_width // (len(texts) + 1)
        for i, text in enumerate(texts):
            self._draw_text(text, ((i + 1) * spacing, 30), self.font_title, self.color_ui_text, center=True)

    def _render_game_over(self):
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        text = "VICTORY!" if self.win_state else "GAME OVER"
        color = self.color_plant_healthy if self.win_state else self.color_plant_thirsty
        self._draw_text(text, (self.screen_width/2, self.screen_height/2), self.font_game_over, color, center=True)

    def _create_particles(self, x, y, color, count, p_type):
        for _ in range(count):
            if p_type == "burst":
                angle = self.rng.uniform(0, 2 * math.pi)
                speed = self.rng.uniform(1, 4)
                dx = math.cos(angle) * speed
                dy = math.sin(angle) * speed
                lifetime = self.rng.integers(20, 40)
            elif p_type == "drip":
                dx = self.rng.uniform(-0.5, 0.5)
                dy = self.rng.uniform(1, 3)
                lifetime = self.rng.integers(15, 30)
            elif p_type == "upward_burst":
                angle = self.rng.uniform(-math.pi * 0.75, -math.pi * 0.25)
                speed = self.rng.uniform(1, 4)
                dx = math.cos(angle) * speed
                dy = math.sin(angle) * speed
                lifetime = self.rng.integers(25, 50)
            self.particles.append([x, y, dx, dy, lifetime, color])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1]  # x += dx
            p[1] += p[2]  # y += dy
            p[3] -= 1     # lifetime -= 1
        self.particles = [p for p in self.particles if p[3] > 0]

    def _render_particles(self):
        for x, y, dx, dy, lifetime, color in self.particles:
            radius = max(0, int(lifetime * 0.15))
            if radius > 0:
                pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}
    
    def _get_plot_rect(self, index):
        row = index // self.grid_cols
        col = index % self.grid_cols
        x = self.grid_offset_x + col * (self.plot_size + self.plot_padding)
        y = self.grid_offset_y + row * (self.plot_size + self.plot_padding)
        return pygame.Rect(x, y, self.plot_size, self.plot_size)
    
    def _get_plot_center(self, index):
        return self._get_plot_rect(index).center

    def _lerp_color(self, color1, color2, t):
        t = max(0, min(1, t))
        return (
            int(color1[0] + (color2[0] - color1[0]) * t),
            int(color1[1] + (color2[1] - color1[1]) * t),
            int(color1[2] + (color2[2] - color1[2]) * t),
        )

    def _draw_text(self, text, pos, font, color, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    action = np.array([0, 0, 0]) # No-op
    
    # Create a window to display the game
    pygame.display.set_caption("Grid Garden")
    display_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get keyboard input for human player
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        current_action = np.array([movement, space, shift])
        
        # In auto_advance=False, we only step when an action is taken.
        # For human play, we step on every key press or every few frames to feel responsive.
        # A simple way is to always step.
        obs, reward, terminated, truncated, info = env.step(current_action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before resetting
            pygame.time.wait(3000)
            obs, info = env.reset()

        # Update the display
        # Pygame uses a different coordinate system, so we need to transpose the observation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Cap the frame rate
        env.clock.tick(30)
        
    env.close()