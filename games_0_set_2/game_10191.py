import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:05:15.828617
# Source Brief: brief_00191.md
# Brief Index: 191
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: A puzzle game where the player manages oscillating colored liquids
    to fill target containers.

    The goal is to fill three target containers (Red, Green, Blue) to at least
    75% capacity within a 15-second time limit. The player controls which liquid
    is selected and when to pour it. The available amount of each liquid in the
    main reservoirs oscillates, adding a timing challenge.

    Visuals are minimalist and clean, with vibrant colors and smooth animations
    to provide clear feedback and a polished experience.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "A puzzle game where you manage oscillating colored liquids to fill target containers before time runs out."
    user_guide = "Controls: Hold SPACE to pour the selected liquid. Press SHIFT to cycle through the red, green, and blue liquids."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.TIME_LIMIT_SECONDS = 15
        self.TARGET_FILL_GOAL = 0.75
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS + 5 # A little buffer

        # --- Colors and Style ---
        self.COLOR_BG = (26, 26, 46)
        self.COLOR_OUTLINE = (230, 230, 250)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLORS_LIQUID = [
            (233, 69, 96),   # Red
            (80, 200, 120),  # Green
            (65, 105, 225)   # Blue
        ]
        self.COLORS_LIQUID_DARK = [
            (c[0]//2, c[1]//2, c[2]//2) for c in self.COLORS_LIQUID
        ]

        # --- Physics and Mechanics ---
        self.OSC_PERIODS = [2.0, 3.0, 4.0]  # seconds
        self.OSC_AMPLITUDE = 0.1  # as a percentage of container height
        self.OUTFLOW_RATE = 0.008  # units of fill per frame

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Initialization ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_small = pygame.font.SysFont("Arial", 16)
            self.font_large = pygame.font.SysFont("Arial", 32)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 20)
            self.font_large = pygame.font.Font(None, 40)

        # --- State Variables ---
        self.steps = None
        self.game_over = None
        self.time_left_frames = None
        self.main_container_volumes = None
        self.target_container_fills = None
        self.selected_liquid_idx = None
        self.prev_shift_state = None
        self.target_reached_bonus = None
        self.last_target_fills_for_reward = None
        self.particles = None
        
        # Initialize state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.game_over = False
        self.time_left_frames = self.TIME_LIMIT_SECONDS * self.FPS

        # 1.0 = 100% volume
        self.main_container_volumes = [1.0, 1.0, 1.0]
        self.target_container_fills = [0.0, 0.0, 0.0]

        self.selected_liquid_idx = 0
        self.prev_shift_state = 0  # 0=released, 1=held

        # Reward tracking
        self.target_reached_bonus = [False, False, False]
        self.last_target_fills_for_reward = [0.0, 0.0, 0.0]

        # Visuals
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Unpack and handle actions
        space_held, shift_held = action[1] == 1, action[2] == 1

        # Handle shift press to cycle liquid
        if shift_held and not self.prev_shift_state:
            self.selected_liquid_idx = (self.selected_liquid_idx + 1) % 3
            # sfx_switch_liquid()
        self.prev_shift_state = shift_held

        # Handle space hold to transfer liquid
        if space_held:
            self._handle_outflow()

        # 2. Update game state
        self.steps += 1
        self.time_left_frames -= 1
        self._update_particles()
        
        # 3. Calculate reward
        reward = self._calculate_reward()

        # 4. Check for termination
        terminated = False
        truncated = False
        win_condition = all(fill >= self.TARGET_FILL_GOAL for fill in self.target_container_fills)
        lose_condition = self.time_left_frames <= 0
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            if not win_condition:
                reward -= 100 # Penalize for running out of steps

        if win_condition:
            reward += 100
            terminated = True
            self.game_over = True
            # sfx_win()
        elif lose_condition and not self.game_over:
            reward -= 100
            terminated = True
            self.game_over = True
            # sfx_lose()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_outflow(self):
        idx = self.selected_liquid_idx
        
        # Available liquid is based on volume and oscillation
        effective_level = self._get_oscillating_level(idx)
        
        if self.main_container_volumes[idx] > 0 and effective_level > 0:
            transfer_amount = min(self.main_container_volumes[idx], self.OUTFLOW_RATE)
            space_in_target = 1.0 - self.target_container_fills[idx]
            transfer_amount = min(transfer_amount, space_in_target)

            if transfer_amount > 0:
                self.main_container_volumes[idx] -= transfer_amount
                self.target_container_fills[idx] += transfer_amount
                # sfx_liquid_flow()
                self._create_particles(idx, space_held=True)

    def _get_oscillating_level(self, liquid_idx):
        time_s = self.steps / self.FPS
        period = self.OSC_PERIODS[liquid_idx]
        base_level = self.main_container_volumes[liquid_idx]
        offset = self.OSC_AMPLITUDE * math.sin(2 * math.pi * time_s / period)
        return max(0, base_level + offset)

    def _calculate_reward(self):
        reward = 0
        for i in range(3):
            current_fill = self.target_container_fills[i]
            last_fill = self.last_target_fills_for_reward[i]
            
            if current_fill <= self.TARGET_FILL_GOAL and current_fill > last_fill:
                 improvement_percent = (current_fill - last_fill) * 100
                 reward += improvement_percent * 0.01

            if current_fill >= self.TARGET_FILL_GOAL and not self.target_reached_bonus[i]:
                reward += 5
                self.target_reached_bonus[i] = True
        
        self.last_target_fills_for_reward = list(self.target_container_fills)
        return reward

    def _get_info(self):
        score = sum(min(f / self.TARGET_FILL_GOAL, 1.0) for f in self.target_container_fills) * 100 / 3
        return {
            "score": score,
            "steps": self.steps,
            "time_left": max(0, self.time_left_frames / self.FPS),
            "target_fills": self.target_container_fills,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_main_containers()
        self._render_target_containers()
        self._render_pipes()
        self._render_particles()

    def _render_main_containers(self):
        num_containers = 3
        c_width, c_height = 80, 200
        total_width = num_containers * (c_width + 20) - 20
        start_x = (self.WIDTH / 4) - (total_width / 2)
        y_pos = 80
        
        for i in range(num_containers):
            x_pos = start_x + i * (c_width + 20)
            rect = pygame.Rect(x_pos, y_pos, c_width, c_height)

            if i == self.selected_liquid_idx and not self.game_over:
                self._draw_glow(rect.center, c_width / 2 + 10, self.COLORS_LIQUID[i])

            liquid_volume = self.main_container_volumes[i]
            liquid_height = int(c_height * liquid_volume)
            liquid_rect = pygame.Rect(x_pos, y_pos + c_height - liquid_height, c_width, liquid_height)
            
            pygame.draw.rect(self.screen, self.COLORS_LIQUID_DARK[i], liquid_rect, border_bottom_left_radius=5, border_bottom_right_radius=5)

            surface_y = y_pos + c_height - liquid_height
            if liquid_volume > 0.01:
                self._draw_wave_surface(x_pos, surface_y, c_width, self.COLORS_LIQUID[i], i)

            pygame.draw.rect(self.screen, self.COLOR_OUTLINE, rect, 2, border_radius=5)

    def _draw_wave_surface(self, x, y, width, color, liquid_idx):
        time_s = self.steps / self.FPS
        period = self.OSC_PERIODS[liquid_idx]
        amplitude = self.OSC_AMPLITUDE * 200 * 0.5
        phase = liquid_idx * math.pi / 2
        
        points = []
        for i in range(width + 1):
            wave_offset = amplitude * math.sin(2 * math.pi * time_s / period + (i / 20.0) + phase)
            points.append((int(x + i), int(y + wave_offset)))
        
        if len(points) > 1:
            pygame.draw.aalines(self.screen, color, False, points)
            pygame.draw.aalines(self.screen, color, False, [(p[0], p[1]+1) for p in points])

    def _render_target_containers(self):
        num_containers = 3
        c_width, c_height = 80, 200
        total_width = num_containers * (c_width + 20) - 20
        start_x = (self.WIDTH * 3 / 4) - (total_width / 2)
        y_pos = 80
        
        for i in range(num_containers):
            x_pos = start_x + i * (c_width + 20)
            rect = pygame.Rect(x_pos, y_pos, c_width, c_height)

            fill_height = int(c_height * self.target_container_fills[i])
            fill_rect = pygame.Rect(x_pos, y_pos + c_height - fill_height, c_width, fill_height)
            pygame.draw.rect(self.screen, self.COLORS_LIQUID[i], fill_rect, border_bottom_left_radius=5, border_bottom_right_radius=5)
            
            pygame.draw.rect(self.screen, self.COLOR_OUTLINE, rect, 2, border_radius=5)
            
            target_y = y_pos + c_height * (1 - self.TARGET_FILL_GOAL)
            pygame.draw.line(self.screen, self.COLOR_OUTLINE, (x_pos - 5, target_y), (x_pos + c_width + 5, target_y), 2)
            
            text = f"{self.target_container_fills[i]*100:.1f}%"
            text_surf = self.font_small.render(text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(rect.centerx, rect.top - 15))
            self.screen.blit(text_surf, text_rect)

    def _render_pipes(self):
        for i in range(3):
            start_x = (self.WIDTH / 4) + (i - 1) * 100
            end_x = (self.WIDTH * 3 / 4) + (i - 1) * 100
            y1, y2 = 280, 320
            
            pygame.draw.line(self.screen, self.COLOR_OUTLINE, (start_x, y1), (start_x, y2), 2)
            pygame.draw.line(self.screen, self.COLOR_OUTLINE, (end_x, y1), (end_x, y2), 2)
            pygame.draw.line(self.screen, self.COLOR_OUTLINE, (start_x, y2), (end_x, y2), 2)

    def _create_particles(self, liquid_idx, space_held):
        if not space_held: return
        start_x = (self.WIDTH / 4) + (liquid_idx - 1) * 100
        end_x = (self.WIDTH * 3 / 4) + (liquid_idx - 1) * 100
        y = 320
        
        for _ in range(3):
            p = {
                "pos": [start_x + random.uniform(-1, 1), y],
                "vel": [(end_x - start_x) / 40 + random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)],
                "lifetime": 40,
                "color": self.COLORS_LIQUID[liquid_idx],
                "radius": random.uniform(3, 5)
            }
            self.particles.append(p)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifetime"] > 0 and p["radius"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifetime"] -= 1
            p["radius"] -= 0.08

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            radius = int(p["radius"])
            if radius > 0:
                self._draw_glow(pos, radius, p["color"])

    def _render_ui(self):
        time_sec = max(0, self.time_left_frames / self.FPS)
        timer_text = f"Time: {time_sec:.2f}"
        color = self.COLOR_TEXT if time_sec > 5 or int(time_sec * 2) % 2 == 0 else (255, 100, 100)
        timer_surf = self.font_large.render(timer_text, True, color)
        timer_rect = timer_surf.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(timer_surf, timer_rect)

        if self.game_over:
            win_condition = all(fill >= self.TARGET_FILL_GOAL for fill in self.target_container_fills)
            end_text = "SUCCESS!" if win_condition else "TIME UP!"
            end_color = (100, 255, 100) if win_condition else (255, 100, 100)
            
            end_surf = self.font_large.render(end_text, True, end_color)
            end_rect = end_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            
            bg_rect = end_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 150))
            self.screen.blit(s, bg_rect)
            self.screen.blit(end_surf, end_rect)

    def _draw_glow(self, center, radius, color):
        max_radius = int(radius)
        for i in range(max_radius, 0, -2):
            alpha = int(255 * (1 - (i / max_radius))**2 * 0.3)
            if alpha > 0:
                pygame.gfxdraw.aacircle(self.screen, int(center[0]), int(center[1]), i, (*color, alpha))

    def close(self):
        pygame.quit()

def validate_implementation(env_class):
    """Call this to verify the environment's implementation."""
    print("Running validation...")
    try:
        env = env_class()
        
        assert env.action_space.shape == (3,)
        assert env.action_space.nvec.tolist() == [5, 2, 2]
        print("✓ Action space is correct.")

        test_obs = env._get_observation()
        assert test_obs.shape == (400, 640, 3) and test_obs.dtype == np.uint8
        print("✓ Observation space is correct.")
        
        obs, info = env.reset()
        assert obs.shape == (400, 640, 3) and isinstance(info, dict)
        print("✓ reset() method is correct.")
        
        test_action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(test_action)
        assert obs.shape == (400, 640, 3) and isinstance(reward, (int, float))
        assert isinstance(term, bool) and not trunc and isinstance(info, dict)
        print("✓ step() method is correct.")

        env.reset()
        env.target_container_fills = [0.8, 0.8, 0.8]
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert terminated, "Win condition did not terminate."
        print("✓ Win condition works.")

        env.reset()
        env.time_left_frames = 1
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert terminated, "Lose condition did not terminate."
        print("✓ Lose condition works.")

        env.close()
        print("\n✓ Implementation validated successfully")

    except Exception as e:
        print(f"\n❌ Validation Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # The validation part can be commented out if you just want to play
    # validate_implementation(GameEnv)
    
    # This part below is for human gameplay
    # It requires a display, so it won't work in a headless environment
    # where SDL_VIDEODRIVER is "dummy".
    # To run this, you might need to comment out the os.environ line at the top.
    try:
        os.environ.pop("SDL_VIDEODRIVER")
    except KeyError:
        pass # It was not set, which is fine.

    print("\nStarting interactive human gameplay session...")
    print("Controls:\n  - SHIFT: Cycle selected liquid\n  - SPACE: Pour liquid\n  - Q or ESC: Quit")
    
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.init()
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Liquid Puzzle")
    
    running = True
    while running:
        space_held, shift_held = 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and (event.key == pygame.K_q or event.key == pygame.K_ESCAPE)):
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]: space_held = 1
        # Use a simple toggle for shift to avoid rapid cycling
        if any(event.type == pygame.KEYDOWN and (event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT) for event in pygame.event.get(pygame.KEYDOWN)):
            shift_held = 1

        action = [0, space_held, shift_held] # movement is action[0], unused
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # The env.screen surface was updated in the step call via _get_observation
        # We need to get the observation again to render it to the display screen
        # Or, more simply, just blit the env's internal screen.
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Final Info: {info}")
            pygame.time.wait(3000)
            obs, info = env.reset()

        env.clock.tick(env.FPS)
        
    env.close()