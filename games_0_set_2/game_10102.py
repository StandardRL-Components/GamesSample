import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
import math
from itertools import combinations
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:55:06.347038
# Source Brief: brief_00102.md
# Brief Index: 102
# """import gymnasium as gym
class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Balance the fluid levels in four pipes by adjusting their flow rates. "
        "Keep the levels synchronized within the safe zone to win, but avoid overflowing."
    )
    user_guide = "Controls: Use ←→ to select a pipe and ↑↓ to adjust its flow rate."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000
        self.WIN_SYNC_DURATION = 10.0  # seconds

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 55)
        self.COLOR_PIPE = (50, 60, 80)
        self.COLOR_PIPE_SELECTED = (255, 255, 100)
        self.COLOR_FLUID = (0, 120, 255)
        self.COLOR_SAFE_ZONE = (0, 255, 150, 30) # RGBA for transparency
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TIMER = (100, 255, 180)
        self.COLOR_FAIL = (255, 80, 80)

        # Game Mechanics
        self.LEVEL_FILL_RATE = 20.0  # units per second at 100% flow
        self.FLOW_RATE_STEP = 5.0
        self.SYNC_THRESHOLD = 5.0
        self.SAFE_LEVEL_MIN, self.SAFE_LEVEL_MAX = 20.0, 80.0
        self.INITIAL_DRAIN_RATE = 0.01 # drains per second
        self.DRAIN_RATE_INCREASE = 0.001 # per second

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
        self.font_small = pygame.font.SysFont('Consolas', 16, bold=True)
        self.font_large = pygame.font.SysFont('Consolas', 48, bold=True)
        self.font_medium = pygame.font.SysFont('Consolas', 24, bold=True)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.pipe_levels = np.zeros(4, dtype=float)
        self.pipe_flow_rates = np.zeros(4, dtype=float)
        self.selected_pipe_index = 0
        self.time_in_sync = 0.0
        self.is_in_sync_period = False
        self.random_drain_rate_per_sec = 0.0
        self.particles = []
        self.last_movement_action = 0 # For debouncing pipe selection

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.pipe_levels = self.np_random.uniform(low=30.0, high=70.0, size=4)
        self.pipe_flow_rates = np.full(4, 50.0, dtype=float)
        self.selected_pipe_index = 0
        self.time_in_sync = 0.0
        self.is_in_sync_period = False
        self.random_drain_rate_per_sec = self.INITIAL_DRAIN_RATE
        self.particles = []
        self.last_movement_action = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False

        self._handle_input(action)
        self._update_game_state()

        # Calculate continuous reward before checking for termination
        reward = self._calculate_continuous_reward()

        # Check for terminal states and get terminal rewards
        terminated, terminal_reward = self._check_termination()
        if terminated:
            self.game_over = True
            reward = terminal_reward

        self.score += reward
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]

        # Debounced left/right selection
        if movement in [3, 4] and movement != self.last_movement_action:
            if movement == 3:  # Left
                self.selected_pipe_index = (self.selected_pipe_index - 1) % 4
            elif movement == 4:  # Right
                self.selected_pipe_index = (self.selected_pipe_index + 1) % 4
        elif movement in [1, 2]:
            if movement == 1:  # Up
                self.pipe_flow_rates[self.selected_pipe_index] += self.FLOW_RATE_STEP
            elif movement == 2:  # Down
                self.pipe_flow_rates[self.selected_pipe_index] -= self.FLOW_RATE_STEP
            self.pipe_flow_rates[self.selected_pipe_index] = np.clip(
                self.pipe_flow_rates[self.selected_pipe_index], 0, 100
            )

        self.last_movement_action = movement

    def _update_game_state(self):
        # Update random drain probability
        self.random_drain_rate_per_sec = self.INITIAL_DRAIN_RATE + self.DRAIN_RATE_INCREASE * (self.steps / self.FPS)
        prob_this_step = self.random_drain_rate_per_sec / self.FPS
        if self.np_random.random() < prob_this_step:
            drain_pipe_idx = self.np_random.integers(0, 4)
            drain_y = self._get_pipe_y(drain_pipe_idx, self.pipe_levels[drain_pipe_idx])
            drain_x = self._get_pipe_x(drain_pipe_idx) + self.pipe_width / 2
            self._create_particles((drain_x, drain_y), self.COLOR_FAIL, 20)
            self.pipe_levels[drain_pipe_idx] = 0.0

        # Update pipe levels based on flow rates
        fill_amount = (self.pipe_flow_rates / 100.0) * (self.LEVEL_FILL_RATE / self.FPS)
        self.pipe_levels += fill_amount

        # Update particles
        for p in self.particles[:]:
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)
            else:
                p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
                p['vel'] = (p['vel'][0], p['vel'][1] + 0.1) # Gravity

    def _calculate_continuous_reward(self):
        reward = 0.0
        # +0.1 for each pipe in safe range
        for level in self.pipe_levels:
            if self.SAFE_LEVEL_MIN <= level <= self.SAFE_LEVEL_MAX:
                reward += 0.1

        # +0.5 for each synchronized pair
        for i, j in combinations(range(4), 2):
            if abs(self.pipe_levels[i] - self.pipe_levels[j]) <= self.SYNC_THRESHOLD:
                reward += 0.5
        return reward

    def _check_termination(self):
        # Player-caused overflow
        if np.any(self.pipe_levels > 100.0):
            return True, -10.0

        all_in_safe_range = np.all((self.pipe_levels >= self.SAFE_LEVEL_MIN) & (self.pipe_levels <= self.SAFE_LEVEL_MAX))
        is_synchronized = (np.max(self.pipe_levels) - np.min(self.pipe_levels)) <= self.SYNC_THRESHOLD

        if self.is_in_sync_period:
            if not all_in_safe_range or not is_synchronized:
                # Failed the sync attempt
                return True, -10.0
            else:
                self.time_in_sync += 1.0 / self.FPS
                if self.time_in_sync >= self.WIN_SYNC_DURATION:
                    # Win condition met
                    return True, 100.0
        else:
            if all_in_safe_range and is_synchronized:
                self.is_in_sync_period = True
                self.time_in_sync = 1.0 / self.FPS
            else:
                self.time_in_sync = 0.0

        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "pipe_levels": self.pipe_levels,
            "pipe_flow_rates": self.pipe_flow_rates,
            "time_in_sync": self.time_in_sync,
        }

    def _render_game(self):
        self._render_background()
        self._render_pipes()
        self._render_particles()

    def _render_background(self):
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _get_pipe_x(self, i):
        return self.pipe_margin + i * (self.pipe_width + self.pipe_spacing)

    def _get_pipe_y(self, i, level):
        return self.pipe_y_bottom - (level / 100.0) * self.pipe_draw_height

    def _render_pipes(self):
        num_pipes = 4
        total_pipe_width = self.WIDTH * 0.8
        self.pipe_width = total_pipe_width / (num_pipes * 1.5)
        self.pipe_spacing = self.pipe_width * 0.5
        self.pipe_margin = (self.WIDTH - (num_pipes * self.pipe_width + (num_pipes - 1) * self.pipe_spacing)) / 2

        self.pipe_y_top = 50
        self.pipe_y_bottom = self.HEIGHT - 80
        self.pipe_draw_height = self.pipe_y_bottom - self.pipe_y_top

        for i in range(num_pipes):
            pipe_x = self._get_pipe_x(i)
            
            # Safe zone rectangle
            safe_zone_rect = pygame.Rect(
                pipe_x,
                self._get_pipe_y(i, self.SAFE_LEVEL_MAX),
                self.pipe_width,
                (self.SAFE_LEVEL_MAX - self.SAFE_LEVEL_MIN) / 100.0 * self.pipe_draw_height
            )
            safe_surface = pygame.Surface(safe_zone_rect.size, pygame.SRCALPHA)
            safe_surface.fill(self.COLOR_SAFE_ZONE)
            self.screen.blit(safe_surface, safe_zone_rect.topleft)

            # Fluid level
            level = self.pipe_levels[i]
            fluid_height = np.clip(level / 100.0, 0, 1) * self.pipe_draw_height
            fluid_rect = pygame.Rect(pipe_x, self.pipe_y_bottom - fluid_height, self.pipe_width, fluid_height)

            # Fluid color changes if out of bounds
            fluid_color = self.COLOR_FLUID
            if self.is_in_sync_period and not (self.SAFE_LEVEL_MIN <= level <= self.SAFE_LEVEL_MAX):
                fluid_color = self.COLOR_FAIL

            # Draw fluid with wavy top
            if fluid_height > 0:
                pygame.draw.rect(self.screen, fluid_color, fluid_rect)
                for x_offset in range(int(self.pipe_width)):
                    angle = (self.steps * 0.1 + x_offset * 0.2)
                    y_offset = math.sin(angle) * 2
                    pygame.draw.line(self.screen, fluid_color,
                                     (pipe_x + x_offset, fluid_rect.top + y_offset),
                                     (pipe_x + x_offset, fluid_rect.top + y_offset - 3))

            # Pipe container
            pygame.draw.rect(self.screen, self.COLOR_PIPE, (pipe_x, self.pipe_y_top, self.pipe_width, self.pipe_draw_height), 2)

            # Selection highlight
            if i == self.selected_pipe_index:
                pygame.draw.rect(self.screen, self.COLOR_PIPE_SELECTED, (pipe_x-2, self.pipe_y_top-2, self.pipe_width+4, self.pipe_draw_height+4), 2, border_radius=2)

            # Flow rate text
            flow_text = f"{self.pipe_flow_rates[i]:.0f}%"
            text_surf = self.font_small.render(flow_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(pipe_x + self.pipe_width / 2, self.pipe_y_bottom + 20))
            self.screen.blit(text_surf, text_rect)

            # Flow rate indicator bar
            flow_bar_width = self.pipe_flow_rates[i] / 100.0 * self.pipe_width
            pygame.draw.rect(self.screen, self.COLOR_PIPE, (pipe_x, self.pipe_y_bottom + 35, self.pipe_width, 8), 1)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, (pipe_x, self.pipe_y_bottom + 35, flow_bar_width, 8))


    def _render_particles(self):
        for p in self.particles:
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]),
                int(p['radius'] * (p['lifetime'] / p['max_lifetime'])),
                p['color']
            )

    def _render_ui(self):
        # Score and Steps
        score_text = f"SCORE: {self.score:.1f}"
        steps_text = f"STEPS: {self.steps}/{self.MAX_STEPS}"
        score_surf = self.font_medium.render(score_text, True, self.COLOR_TEXT)
        steps_surf = self.font_medium.render(steps_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 10))
        self.screen.blit(steps_surf, (self.WIDTH - steps_surf.get_width() - 20, 10))

        # Sync Timer
        if self.is_in_sync_period:
            timer_text = f"{self.time_in_sync:.1f}s / {self.WIN_SYNC_DURATION:.1f}s"
            timer_surf = self.font_large.render(timer_text, True, self.COLOR_TIMER)
            timer_rect = timer_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            # Background for timer
            bg_rect = timer_rect.inflate(20, 10)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            self.screen.blit(bg_surf, bg_rect)
            self.screen.blit(timer_surf, timer_rect)

            # Progress arc
            progress_angle = (self.time_in_sync / self.WIN_SYNC_DURATION) * 360
            if progress_angle > 0:
                 pygame.draw.arc(self.screen, self.COLOR_TIMER, bg_rect.inflate(10,10), math.radians(-90), math.radians(-90 + progress_angle), 4)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'radius': self.np_random.uniform(2, 5),
                'lifetime': self.np_random.integers(15, 30),
                'max_lifetime': 30
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the environment directly and play it manually.
    # This is for testing and demonstration purposes.
    
    # Unset the dummy video driver to allow for a display window.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pipe Equilibrium")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # To avoid rapid-fire pipe selection, we track the last frame a key was pressed.
    last_lr_press_time = -1
    KEY_DEBOUNCE_MS = 150 

    while running:
        movement_action = 0 # No-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    
        keys = pygame.key.get_pressed()
        
        # Continuous actions (up/down)
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        
        # Debounced actions (left/right)
        current_time = pygame.time.get_ticks()
        if current_time - last_lr_press_time > KEY_DEBOUNCE_MS:
            if keys[pygame.K_LEFT]:
                movement_action = 3
                last_lr_press_time = current_time
            elif keys[pygame.K_RIGHT]:
                movement_action = 4
                last_lr_press_time = current_time

        # The action space is MultiDiscrete, but for manual play we only care about movement
        # The other two dimensions of the action are not used in this game logic.
        action = [movement_action, 0, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to restart.")
            
        clock.tick(env.FPS)
        
    env.close()