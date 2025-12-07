import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:50:56.697973
# Source Brief: brief_02415.md
# Brief Index: 2415
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manage water flow from multiple pipes to keep a reservoir at a stable level. "
        "Watch out for random pipe leaks that disrupt the balance!"
    )
    user_guide = (
        "Controls: Use ←→ to select a pipe and ↑↓ to adjust its flow slider. "
        "Keep the reservoir level in the green zone to win."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (30, 35, 40)
    COLOR_PIPE = (70, 80, 90)
    COLOR_WATER = (50, 150, 255)
    COLOR_WATER_HIGHLIGHT = (150, 200, 255)
    COLOR_LEAK = (255, 50, 50)
    COLOR_LEAK_PARTICLE = (200, 40, 40)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_SLIDER_BG = (40, 45, 50)
    COLOR_SLIDER_HANDLE = (150, 160, 170)
    COLOR_SLIDER_SELECTED = (255, 200, 0)
    COLOR_TARGET_ZONE = (0, 255, 100, 50) # RGBA for transparency
    COLOR_SAFE_ZONE = (255, 200, 0, 30)  # RGBA for transparency
    
    # Game Parameters
    MAX_EPISODE_STEPS = 2000
    WIN_DURATION_SECS = 30
    WIN_DURATION_STEPS = WIN_DURATION_SECS * FPS
    
    NUM_PIPES = 4
    SLIDER_INCREMENT = 0.05
    LEAK_CHANCE_PER_SEC = 0.1
    LEAK_PROB_PER_STEP = 1 - (1 - LEAK_CHANCE_PER_SEC)**(1 / FPS)
    LEAK_FLOW_MULTIPLIER = 0.75 # A leak removes 75% of the pipe's flow

    RESERVOIR_CAPACITY = 100.0
    PIPE_MAX_FLOW_PER_STEP = 0.04
    # Outflow is set to balance the system when all sliders are at 0.5
    OUTFLOW_RATE = NUM_PIPES * 0.5 * PIPE_MAX_FLOW_PER_STEP

    LEVEL_SAFE_MIN = 0.4
    LEVEL_SAFE_MAX = 0.6
    LEVEL_TARGET_MIN = 0.48
    LEVEL_TARGET_MAX = 0.52

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
        self.font_ui = pygame.font.SysFont("monospace", 16)
        self.font_game_over = pygame.font.SysFont("impact", 60)

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reservoir_level = 0.0
        self.pipe_sliders = np.zeros(self.NUM_PIPES, dtype=np.float32)
        self.pipe_leaks = np.zeros(self.NUM_PIPES, dtype=bool)
        self.newly_leaked = np.zeros(self.NUM_PIPES, dtype=bool)
        self.leak_flash_timers = np.zeros(self.NUM_PIPES, dtype=np.float32)
        self.win_timer = 0
        self.selected_pipe = 0
        self.particles = []
        
        # self.reset() # reset is called by the environment wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reservoir_level = 0.5
        self.pipe_sliders = np.full(self.NUM_PIPES, 0.5, dtype=np.float32)
        self.pipe_leaks = np.zeros(self.NUM_PIPES, dtype=bool)
        self.newly_leaked = np.zeros(self.NUM_PIPES, dtype=bool)
        self.leak_flash_timers = np.zeros(self.NUM_PIPES, dtype=np.float32)

        self.win_timer = 0
        self.selected_pipe = 0
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_input(action)
        self._update_leaks()
        self._update_reservoir()
        self._update_particles()
        
        self.steps += 1
        
        terminated = self._check_termination()
        reward = self._calculate_reward(terminated)
        self.score += reward
        
        if terminated:
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        
        if movement == 3: # Left
            self.selected_pipe = (self.selected_pipe - 1 + self.NUM_PIPES) % self.NUM_PIPES
        elif movement == 4: # Right
            self.selected_pipe = (self.selected_pipe + 1) % self.NUM_PIPES
        elif movement == 1: # Up
            self.pipe_sliders[self.selected_pipe] += self.SLIDER_INCREMENT
        elif movement == 2: # Down
            self.pipe_sliders[self.selected_pipe] -= self.SLIDER_INCREMENT
            
        self.pipe_sliders[self.selected_pipe] = np.clip(self.pipe_sliders[self.selected_pipe], 0, 1)

    def _update_leaks(self):
        self.newly_leaked.fill(False)
        for i in range(self.NUM_PIPES):
            if not self.pipe_leaks[i] and self.np_random.random() < self.LEAK_PROB_PER_STEP:
                self.pipe_leaks[i] = True
                self.newly_leaked[i] = True
                self.leak_flash_timers[i] = 0
                # sfx: leak_start
            if self.pipe_leaks[i]:
                self.leak_flash_timers[i] += 1
                if self.np_random.random() < 0.1: # Chance to spawn leak particle
                    pipe_x = 100 + i * 120
                    self._spawn_particles(1, (pipe_x, 200), self.COLOR_LEAK_PARTICLE, (0, 2), 0.1, 30)


    def _update_reservoir(self):
        total_inflow = 0
        for i in range(self.NUM_PIPES):
            flow = self.pipe_sliders[i] * self.PIPE_MAX_FLOW_PER_STEP
            if self.pipe_leaks[i]:
                flow *= (1 - self.LEAK_FLOW_MULTIPLIER)
            total_inflow += flow
            
            # Spawn water particles based on flow
            pipe_x = 100 + i * 120
            num_particles = int(flow / self.PIPE_MAX_FLOW_PER_STEP * 3)
            if num_particles > 0:
                self._spawn_particles(num_particles, (pipe_x, 260), self.COLOR_WATER, (0, 1), 0.05, 40)

        delta_level = (total_inflow - self.OUTFLOW_RATE) / self.RESERVOIR_CAPACITY
        self.reservoir_level = np.clip(self.reservoir_level + delta_level, 0, 1)

    def _spawn_particles(self, num, pos, color, vel_y_range, gravity, lifetime):
        for _ in range(num):
            vel_x = self.np_random.uniform(-1, 1)
            vel_y = self.np_random.uniform(vel_y_range[0], vel_y_range[1])
            particle = {
                "pos": pygame.Vector2(pos),
                "vel": pygame.Vector2(vel_x, vel_y),
                "color": color,
                "gravity": gravity,
                "lifetime": lifetime,
                "life": float(lifetime)
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p["vel"].y += p["gravity"]
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        is_out_of_bounds = not (self.LEVEL_SAFE_MIN <= self.reservoir_level <= self.LEVEL_SAFE_MAX)
        if is_out_of_bounds:
            # sfx: fail_sound
            return True

        is_in_target = self.LEVEL_TARGET_MIN <= self.reservoir_level <= self.LEVEL_TARGET_MAX
        if is_in_target:
            self.win_timer += 1
        else:
            self.win_timer = 0
            
        if self.win_timer >= self.WIN_DURATION_STEPS:
            # sfx: win_sound
            return True
            
        if self.steps >= self.MAX_EPISODE_STEPS:
            return True
            
        return False

    def _calculate_reward(self, terminated):
        reward = 0
        
        # Leak penalty
        reward -= np.sum(self.newly_leaked) * 5.0
        
        # Level-based continuous reward
        if self.LEVEL_TARGET_MIN <= self.reservoir_level <= self.LEVEL_TARGET_MAX:
            reward += 1.0
        elif not (self.LEVEL_SAFE_MIN <= self.reservoir_level <= self.LEVEL_SAFE_MAX):
            reward -= 1.0
        
        # Terminal rewards
        if terminated:
            if self.win_timer >= self.WIN_DURATION_STEPS:
                reward += 100.0
            elif not (self.LEVEL_SAFE_MIN <= self.reservoir_level <= self.LEVEL_SAFE_MAX):
                reward -= 100.0
                
        return reward

    def _get_observation(self):
        self._render_background()
        self._render_pipes_and_reservoir()
        self._render_water()
        self._render_leaks()
        self._render_sliders()
        self._render_particles()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for i in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

    def _render_pipes_and_reservoir(self):
        self.pipe_rects = []
        for i in range(self.NUM_PIPES):
            x = 100 + i * 120 - 25
            rect = pygame.Rect(x, 50, 50, 200)
            pygame.draw.rect(self.screen, self.COLOR_PIPE, rect, 2, border_radius=5)
            self.pipe_rects.append(rect)

        self.reservoir_rect = pygame.Rect(40, 270, self.SCREEN_WIDTH - 80, 80)
        pygame.draw.rect(self.screen, self.COLOR_PIPE, self.reservoir_rect, 2, border_radius=5)

    def _render_water(self):
        # Pipes
        for i in range(self.NUM_PIPES):
            pipe_rect = self.pipe_rects[i]
            water_height = int(pipe_rect.height * self.pipe_sliders[i])
            water_rect = pygame.Rect(
                pipe_rect.left, pipe_rect.bottom - water_height,
                pipe_rect.width, water_height
            )
            pygame.draw.rect(self.screen, self.COLOR_WATER, water_rect, border_bottom_left_radius=5, border_bottom_right_radius=5)

        # Reservoir
        res_rect = self.reservoir_rect
        water_height = int(res_rect.height * self.reservoir_level)
        water_rect = pygame.Rect(
            res_rect.left, res_rect.bottom - water_height,
            res_rect.width, water_height
        )
        pygame.draw.rect(self.screen, self.COLOR_WATER, water_rect, border_bottom_left_radius=5, border_bottom_right_radius=5)
        
        # Water surface highlight
        if water_height > 0:
            y = res_rect.bottom - water_height
            pygame.draw.line(self.screen, self.COLOR_WATER_HIGHLIGHT, (res_rect.left + 1, y), (res_rect.right - 1, y), 1)

        # Target/Safe zones
        target_min_y = res_rect.bottom - int(res_rect.height * self.LEVEL_TARGET_MAX)
        target_max_y = res_rect.bottom - int(res_rect.height * self.LEVEL_TARGET_MIN)
        target_zone_rect = pygame.Rect(res_rect.left, target_min_y, res_rect.width, target_max_y - target_min_y)
        
        safe_min_y = res_rect.bottom - int(res_rect.height * self.LEVEL_SAFE_MAX)
        safe_max_y = res_rect.bottom - int(res_rect.height * self.LEVEL_SAFE_MIN)
        safe_zone_rect = pygame.Rect(res_rect.left, safe_min_y, res_rect.width, safe_max_y - safe_min_y)
        
        # Use a surface for transparency
        s = pygame.Surface((res_rect.width, res_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_SAFE_ZONE, (0, safe_min_y - res_rect.top, res_rect.width, safe_zone_rect.height))
        pygame.draw.rect(s, self.COLOR_TARGET_ZONE, (0, target_min_y - res_rect.top, res_rect.width, target_zone_rect.height))
        self.screen.blit(s, (res_rect.left, res_rect.top))

    def _render_leaks(self):
        for i in range(self.NUM_PIPES):
            if self.pipe_leaks[i]:
                pipe_rect = self.pipe_rects[i]
                center_x = pipe_rect.centerx
                center_y = pipe_rect.centery
                
                # Flashing effect
                alpha = int(128 + 127 * math.sin(self.leak_flash_timers[i] * 0.2))
                radius = 15
                
                # Use gfxdraw for smooth, alpha-blended circle
                pygame.gfxdraw.filled_circle(self.screen, int(center_x), int(center_y), radius, (*self.COLOR_LEAK, alpha))
                pygame.gfxdraw.aacircle(self.screen, int(center_x), int(center_y), radius, (*self.COLOR_LEAK, alpha))


    def _render_sliders(self):
        for i in range(self.NUM_PIPES):
            pipe_rect = self.pipe_rects[i]
            slider_x = pipe_rect.centerx
            slider_y_start = self.reservoir_rect.bottom + 20
            slider_y_end = self.SCREEN_HEIGHT - 15
            
            # Track
            pygame.draw.line(self.screen, self.COLOR_SLIDER_BG, (slider_x, slider_y_start), (slider_x, slider_y_end), 4)

            # Handle
            handle_y = slider_y_start + (1 - self.pipe_sliders[i]) * (slider_y_end - slider_y_start)
            
            # Selection Glow
            if i == self.selected_pipe:
                pygame.gfxdraw.filled_circle(self.screen, int(slider_x), int(handle_y), 12, (*self.COLOR_SLIDER_SELECTED, 80))
                pygame.gfxdraw.aacircle(self.screen, int(slider_x), int(handle_y), 12, (*self.COLOR_SLIDER_SELECTED, 150))

            pygame.gfxdraw.filled_circle(self.screen, int(slider_x), int(handle_y), 8, self.COLOR_SLIDER_HANDLE)
            pygame.gfxdraw.aacircle(self.screen, int(slider_x), int(handle_y), 8, self.COLOR_UI_TEXT)

    def _render_particles(self):
        for p in self.particles:
            size = int(p["life"] / p["lifetime"] * 4)
            if size > 0:
                color_with_alpha = (*p["color"], int(p["life"] / p["lifetime"] * 255))
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), size, color_with_alpha)

    def _render_ui(self):
        # Win timer bar
        win_progress = self.win_timer / self.WIN_DURATION_STEPS
        bar_width = self.SCREEN_WIDTH * win_progress
        pygame.draw.rect(self.screen, self.COLOR_TARGET_ZONE, (0, 0, bar_width, 10))
        
        # Reservoir level text
        level_text = self.font_ui.render(f"LEVEL: {self.reservoir_level:.3f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.reservoir_rect.left + 5, self.reservoir_rect.top - 20))

        # Score text
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 15))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        
        if self.win_timer >= self.WIN_DURATION_STEPS:
            text = "SYSTEM STABLE"
            color = self.COLOR_TARGET_ZONE
        else:
            text = "LEVEL CRITICAL"
            color = self.COLOR_LEAK

        text_surface = self.font_game_over.render(text, True, color)
        text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "reservoir_level": self.reservoir_level,
            "win_timer": self.win_timer,
        }
        
    def close(self):
        pygame.font.quit()
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
    # This block allows you to play the game manually for testing
    # For manual play, we need a real display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Pygame setup for manual play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Pipe Balance")
    clock = pygame.time.Clock()

    while running:
        action = np.array([0, 0, 0]) # Default action: do nothing
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward}")
            print("Press 'R' to reset.")

        clock.tick(GameEnv.FPS)

    env.close()