import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control the flow of a volatile liquid by adjusting three levers to fill a container to a target level before time runs out."
    )
    user_guide = (
        "Controls: Use ↑↓ and ←→ to adjust levers A and B. Use space and shift to adjust lever C."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 6000
    TARGET_LIQUID = 50.0
    MAX_FLOW_RATE = 0.5
    INITIAL_FLOW_RATE = 0.05
    DIFFICULTY_INTERVAL = 500
    DIFFICULTY_INCREASE = 0.005

    # --- Colors ---
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (25, 35, 45)
    COLOR_PIPE = (70, 80, 90)
    COLOR_PIPE_BORDER = (90, 100, 110)
    COLOR_LIQUID = (50, 150, 255)
    COLOR_PARTICLE = (150, 200, 255)
    COLOR_LEVER_BG = (40, 50, 60)
    COLOR_LEVER_HANDLE = (255, 180, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TARGET_CONTAINER = (40, 100, 60)
    COLOR_TARGET_BORDER = (60, 150, 90)
    COLOR_TIMER_GOOD = (60, 180, 90)
    COLOR_TIMER_BAD = (220, 50, 50)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Exact spaces as required
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font_big = pygame.font.SysFont('Consolas', 36)
            self.font_medium = pygame.font.SysFont('Consolas', 24)
        except pygame.error:
            self.font_big = pygame.font.SysFont(None, 48)
            self.font_medium = pygame.font.SysFont(None, 32)

        # Initialize state variables to avoid attribute errors
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.liquid_level = 0.0
        self.last_liquid_level = 0.0
        self.lever_positions = [0.5, 0.5, 0.5]
        self.flow_rate = 0.0
        self.fluctuation_range = (0.05, 0.10)
        self.flow_particles = []
        self.win = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        
        self.liquid_level = 0.0
        self.last_liquid_level = 0.0
        self.lever_positions = [0.5, 0.5, 0.5] # A, B, C
        self.flow_rate = self.INITIAL_FLOW_RATE
        self.fluctuation_range = (0.05, 0.10)
        
        self.flow_particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._update_game_state(action)
        
        reward = self._calculate_reward()
        terminated = self._check_termination()
        
        self.score += reward
        
        # Gymnasium API requires a boolean for truncated, not a tuple
        truncated = False
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_game_state(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        flow_rate_change_factor = 0.0

        # --- Handle Actions ---
        # Lever A (Up/Down)
        if movement == 1: # Up
            flow_rate_change_factor += self.np_random.uniform(self.fluctuation_range[0], self.fluctuation_range[1])
            self.lever_positions[0] = min(1.0, self.lever_positions[0] + 0.1)
        elif movement == 2: # Down
            flow_rate_change_factor -= self.np_random.uniform(self.fluctuation_range[0], self.fluctuation_range[1])
            self.lever_positions[0] = max(0.0, self.lever_positions[0] - 0.1)

        # Lever B (Left/Right)
        if movement == 3: # Left
            flow_rate_change_factor -= self.np_random.uniform(self.fluctuation_range[0], self.fluctuation_range[1])
            self.lever_positions[1] = max(0.0, self.lever_positions[1] - 0.1)
        elif movement == 4: # Right
            flow_rate_change_factor += self.np_random.uniform(self.fluctuation_range[0], self.fluctuation_range[1])
            self.lever_positions[1] = min(1.0, self.lever_positions[1] + 0.1)
            
        # Lever C (Space/Shift)
        if space_held:
            flow_rate_change_factor += self.np_random.uniform(self.fluctuation_range[0], self.fluctuation_range[1])
            self.lever_positions[2] = min(1.0, self.lever_positions[2] + 0.1)
        if shift_held:
            flow_rate_change_factor -= self.np_random.uniform(self.fluctuation_range[0], self.fluctuation_range[1])
            self.lever_positions[2] = max(0.0, self.lever_positions[2] - 0.1)

        # --- Update State ---
        self.steps += 1
        
        self.flow_rate += self.flow_rate * flow_rate_change_factor
        self.flow_rate = max(0, min(self.MAX_FLOW_RATE, self.flow_rate))
        
        self.liquid_level += self.flow_rate
        self.liquid_level = min(self.TARGET_LIQUID, self.liquid_level)

        # Update difficulty
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            new_min = self.fluctuation_range[0] + self.DIFFICULTY_INCREASE
            new_max = self.fluctuation_range[1] + self.DIFFICULTY_INCREASE
            self.fluctuation_range = (new_min, new_max)

        self._update_particles()

    def _calculate_reward(self):
        liquid_added = self.liquid_level - self.last_liquid_level
        reward = liquid_added * 0.1

        if self.liquid_level >= self.TARGET_LIQUID:
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            reward -= 100

        self.last_liquid_level = self.liquid_level
        return reward

    def _check_termination(self):
        terminated = False
        if self.liquid_level >= self.TARGET_LIQUID:
            terminated = True
            self.game_over = True
            self.win = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win = False
        
        return terminated

    def _get_observation(self):
        self._render_frame()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def render(self):
        return self._get_observation()

    def _render_frame(self):
        self._render_background()
        self._render_pipes_and_containers()
        self._render_liquid()
        self._render_particles()
        self._render_levers()
        self._render_ui()

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_pipes_and_containers(self):
        # Source Container (left)
        pygame.draw.rect(self.screen, self.COLOR_PIPE, (20, 100, 80, 200))
        pygame.draw.rect(self.screen, self.COLOR_PIPE_BORDER, (20, 100, 80, 200), 2)
        
        # Target Container (right)
        target_rect = pygame.Rect(540, 100, 80, 200)
        pygame.draw.rect(self.screen, self.COLOR_TARGET_CONTAINER, target_rect)
        pygame.draw.rect(self.screen, self.COLOR_TARGET_BORDER, target_rect, 2)

        # Pipes
        pipe_y = 190
        pipe_width = 20
        # Horizontal pipe
        pygame.draw.rect(self.screen, self.COLOR_PIPE, (100, pipe_y, 440, pipe_width))
        pygame.draw.rect(self.screen, self.COLOR_PIPE_BORDER, (100, pipe_y, 440, pipe_width), 2)
        # Inlet pipe
        pygame.draw.rect(self.screen, self.COLOR_PIPE, (530, pipe_y - 20, 20, 40))
        pygame.draw.rect(self.screen, self.COLOR_PIPE_BORDER, (530, pipe_y-20, 20, 40), 2)

    def _render_liquid(self):
        # Source liquid (always full)
        pygame.draw.rect(self.screen, self.COLOR_LIQUID, (22, 102, 76, 196))
        
        # Target liquid
        liquid_height = (self.liquid_level / self.TARGET_LIQUID) * 196
        if liquid_height > 0:
            pygame.draw.rect(self.screen, self.COLOR_LIQUID, 
                             (542, 102 + (196 - liquid_height), 76, liquid_height))

    def _update_particles(self):
        # Spawn new particles based on flow rate
        if self.np_random.random() < self.flow_rate * 2:
            # Spawn particle at the start of the pipe
            particle = {
                "x": 110, 
                "y": self.np_random.uniform(195, 205),
                "life": 200,
                "speed": 1 + self.flow_rate * 10
            }
            self.flow_particles.append(particle)

        # Update and draw existing particles
        for p in self.flow_particles[:]:
            p['x'] += p['speed']
            p['life'] -= 1
            if p['life'] <= 0 or p['x'] > 530:
                self.flow_particles.remove(p)

    def _render_particles(self):
        for p in self.flow_particles:
            size = int(max(1, 4 * self.flow_rate))
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), size, self.COLOR_PARTICLE)

    def _render_levers(self):
        lever_data = [
            {"label": "A", "pos": (180, 320)},
            {"label": "B", "pos": (320, 320)},
            {"label": "C", "pos": (460, 320)},
        ]
        
        for i, data in enumerate(lever_data):
            x, y = data["pos"]
            value = self.lever_positions[i]
            
            # Background slot
            pygame.draw.rect(self.screen, self.COLOR_LEVER_BG, (x - 10, y - 50, 20, 100), border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_PIPE, (x - 10, y - 50, 20, 100), 2, border_radius=5)
            
            # Handle
            handle_y = y + 40 - (value * 80)
            pygame.draw.circle(self.screen, self.COLOR_LEVER_HANDLE, (x, int(handle_y)), 12)
            pygame.draw.circle(self.screen, (255,255,255), (x, int(handle_y)), 12, 1)

            # Label
            label_text = self.font_medium.render(data["label"], True, self.COLOR_TEXT)
            self.screen.blit(label_text, (x - label_text.get_width() // 2, y + 60))

    def _render_ui(self):
        # Timer Bar
        time_ratio = self.steps / self.MAX_STEPS
        bar_width = self.SCREEN_WIDTH * (1 - time_ratio)
        bar_color = self._interpolate_color(self.COLOR_TIMER_GOOD, self.COLOR_TIMER_BAD, time_ratio)
        pygame.draw.rect(self.screen, bar_color, (0, 0, bar_width, 10))

        # Flow Rate Display
        flow_text = self.font_medium.render(f"Flow: {self.flow_rate:.3f} u/s", True, self.COLOR_TEXT)
        self.screen.blit(flow_text, (20, self.SCREEN_HEIGHT - 40))

        # Liquid Level Display
        level_text = self.font_medium.render(f"Level: {self.liquid_level:.2f}/{self.TARGET_LIQUID:.0f} U", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 20, self.SCREEN_HEIGHT - 40))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                msg = "TARGET REACHED"
                color = self.COLOR_TARGET_BORDER
            else:
                msg = "TIME'S UP"
                color = self.COLOR_TIMER_BAD
                
            end_text = self.font_big.render(msg, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH // 2 - end_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - end_text.get_height() // 2))

    def _interpolate_color(self, color1, color2, factor):
        r = color1[0] + (color2[0] - color1[0]) * factor
        g = color1[1] + (color2[1] - color1[1]) * factor
        b = color1[2] + (color2[2] - color1[2]) * factor
        return (int(r), int(g), int(b))
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "liquid_level": self.liquid_level,
            "flow_rate": self.flow_rate
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Un-dummy the video driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Use a display for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Liquid Flow Control")
    clock = pygame.time.Clock()

    print("\n--- Human Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset Environment")
    print("----------------------\n")

    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # Wait for a moment then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS for smooth human experience

    env.close()