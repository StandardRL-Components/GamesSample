import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:39:04.372040
# Source Brief: brief_00012.md
# Brief Index: 12
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Shadow:
    """Represents a single moving shadow."""
    def __init__(self, screen_width, screen_height, np_random):
        self.np_random = np_random
        self.size = pygame.Vector2(self.np_random.uniform(20, 40), self.np_random.uniform(20, 40))
        self.rect = pygame.Rect(0, 0, self.size.x, self.size.y)
        
        self.path_type = self.np_random.choice(['circle', 'ellipse', 'figure_eight'])
        
        margin = 100
        self.center = pygame.Vector2(
            self.np_random.uniform(margin, screen_width - margin),
            self.np_random.uniform(margin, screen_height - margin)
        )
        self.radius_x = self.np_random.uniform(40, 120)
        self.radius_y = self.np_random.uniform(40, 120) if self.path_type != 'circle' else self.radius_x
        
        self.speed = self.np_random.uniform(0.01, 0.03)
        self.phase = self.np_random.uniform(0, 2 * math.pi)
        
        self.a = self.np_random.choice([1, 2, 3])
        self.b = self.np_random.choice([1, 2, 3])
        if self.a == self.b: self.b += 1
        
        self.illuminated_time = 0.0
        self.is_fully_illuminated = False

    def update(self, time_step):
        angle = time_step * self.speed + self.phase
        
        if self.path_type == 'circle' or self.path_type == 'ellipse':
            self.rect.centerx = self.center.x + self.radius_x * math.cos(angle)
            self.rect.centery = self.center.y + self.radius_y * math.sin(angle)
        elif self.path_type == 'figure_eight':
            self.rect.centerx = self.center.x + self.radius_x * math.sin(self.a * angle)
            self.rect.centery = self.center.y + self.radius_y * math.cos(self.b * angle)
            
    def check_illumination(self, lights, light_radius, dt):
        is_lit = False
        for light_pos, light_on in lights:
            if light_on:
                dist_sq = (light_pos.x - self.rect.centerx)**2 + (light_pos.y - self.rect.centery)**2
                if dist_sq < light_radius**2:
                    is_lit = True
                    break
        
        if is_lit:
            self.illuminated_time += dt
        else:
            self.illuminated_time = 0
        
        return is_lit

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Use two movable spotlights to find and fully illuminate all the elusive shadows before time runs out."
    user_guide = "Controls: Use arrow keys (↑↓←→) to move the selected spotlight. Press space to toggle the light on/off and shift to switch between the two spotlights."
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    GAME_DURATION_SECONDS = 90
    
    COLOR_BG = (20, 25, 35)
    COLOR_LIGHT_OFF = (80, 80, 80)
    COLOR_LIGHT_ON = (255, 255, 180)
    COLOR_LIGHT_SELECTED_BORDER = (255, 255, 255)
    COLOR_SHADOW = (100, 110, 120)
    COLOR_SHADOW_ILLUMINATED = (210, 220, 230)
    COLOR_UI_TEXT = (230, 230, 240)
    COLOR_PROGRESS_BAR = (60, 200, 60)
    COLOR_PROGRESS_BAR_BG = (60, 60, 60)

    LIGHT_RADIUS = 60
    LIGHT_SPEED = 5
    NUM_SHADOWS = 12

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
        self.font_small = pygame.font.SysFont("Consolas", 20)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.light1_pos = pygame.Vector2(0, 0)
        self.light2_pos = pygame.Vector2(0, 0)
        self.light1_on = False
        self.light2_on = False
        self.selected_light = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.shadows = []
        
        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # this is for dev, not needed in prod

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.GAME_DURATION_SECONDS
        
        self.light1_pos = pygame.Vector2(self.SCREEN_WIDTH * 0.25, self.SCREEN_HEIGHT * 0.5)
        self.light2_pos = pygame.Vector2(self.SCREEN_WIDTH * 0.75, self.SCREEN_HEIGHT * 0.5)
        self.light1_on = False
        self.light2_on = False
        self.selected_light = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.shadows = [Shadow(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.np_random) for _ in range(self.NUM_SHADOWS)]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_action, shift_action = action
        current_space_held = (space_action == 1)
        current_shift_held = (shift_action == 1)

        reward = 0
        
        # --- Handle Input ---
        # 1. Switch selected light
        if current_shift_held and not self.prev_shift_held:
            self.selected_light = 1 - self.selected_light
            # sfx: switch_sound

        # 2. Toggle active light on/off
        if current_space_held and not self.prev_space_held:
            if self.selected_light == 0:
                self.light1_on = not self.light1_on
            else:
                self.light2_on = not self.light2_on
            # sfx: toggle_light_sound

        # 3. Move selected light
        light_to_move = self.light1_pos if self.selected_light == 0 else self.light2_pos
        if movement == 1: light_to_move.y -= self.LIGHT_SPEED
        elif movement == 2: light_to_move.y += self.LIGHT_SPEED
        elif movement == 3: light_to_move.x -= self.LIGHT_SPEED
        elif movement == 4: light_to_move.x += self.LIGHT_SPEED
        
        # Clamp light positions to screen
        self.light1_pos.x = np.clip(self.light1_pos.x, 0, self.SCREEN_WIDTH)
        self.light1_pos.y = np.clip(self.light1_pos.y, 0, self.SCREEN_HEIGHT)
        self.light2_pos.x = np.clip(self.light2_pos.x, 0, self.SCREEN_WIDTH)
        self.light2_pos.y = np.clip(self.light2_pos.y, 0, self.SCREEN_HEIGHT)

        self.prev_space_held = current_space_held
        self.prev_shift_held = current_shift_held

        # --- Update Game State ---
        self.steps += 1
        dt = 1.0 / self.FPS
        self.time_left -= dt

        lights_state = [(self.light1_pos, self.light1_on), (self.light2_pos, self.light2_on)]
        
        fully_illuminated_count = 0
        for shadow in self.shadows:
            shadow.update(self.steps)
            
            was_fully_illuminated = shadow.is_fully_illuminated
            is_lit_this_frame = shadow.check_illumination(lights_state, self.LIGHT_RADIUS, dt)
            
            if is_lit_this_frame:
                reward += 0.1 # Continuous reward for illumination
            
            if shadow.illuminated_time >= 1.0:
                shadow.is_fully_illuminated = True
                if not was_fully_illuminated:
                    reward += 1.0 # Event reward for completing a shadow
                    self.score += 1
                    # sfx: success_chime
                fully_illuminated_count += 1
            else:
                shadow.is_fully_illuminated = False

        # --- Check Termination ---
        terminated = False
        if self.time_left <= 0:
            terminated = True
            reward -= 100 # Penalty for timeout
            # sfx: failure_sound
        
        if fully_illuminated_count == self.NUM_SHADOWS:
            terminated = True
            reward += 100 # Bonus for winning
            # sfx: victory_fanfare
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render shadows
        for shadow in self.shadows:
            color = self.COLOR_SHADOW_ILLUMINATED if shadow.illuminated_time > 0 else self.COLOR_SHADOW
            pygame.draw.rect(self.screen, color, shadow.rect, border_radius=3)
            
            if shadow.is_fully_illuminated:
                # Progress bar
                bar_pos = (shadow.rect.left, shadow.rect.top - 8)
                bar_size = (shadow.rect.width, 5)
                pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_BG, (*bar_pos, *bar_size))
                pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR, (*bar_pos, *bar_size))

        # Render lights
        self._render_light(self.light1_pos, self.light1_on, self.selected_light == 0)
        self._render_light(self.light2_pos, self.light2_on, self.selected_light == 1)

    def _render_light(self, pos, is_on, is_selected):
        int_pos = (int(pos.x), int(pos.y))
        
        if is_on:
            # Additive glow effect
            glow_surface = pygame.Surface((self.LIGHT_RADIUS * 4, self.LIGHT_RADIUS * 4), pygame.SRCALPHA)
            for i in range(self.LIGHT_RADIUS, 0, -2):
                alpha = int(100 * (1 - (i / self.LIGHT_RADIUS))**2)
                pygame.draw.circle(
                    glow_surface,
                    (*self.COLOR_LIGHT_ON, alpha),
                    (self.LIGHT_RADIUS * 2, self.LIGHT_RADIUS * 2),
                    i
                )
            self.screen.blit(glow_surface, (int_pos[0] - self.LIGHT_RADIUS * 2, int_pos[1] - self.LIGHT_RADIUS * 2), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Main light circle
            pygame.gfxdraw.filled_circle(self.screen, int_pos[0], int_pos[1], self.LIGHT_RADIUS, self.COLOR_LIGHT_ON)
            pygame.gfxdraw.aacircle(self.screen, int_pos[0], int_pos[1], self.LIGHT_RADIUS, self.COLOR_LIGHT_ON)
        else:
            pygame.gfxdraw.filled_circle(self.screen, int_pos[0], int_pos[1], self.LIGHT_RADIUS, self.COLOR_LIGHT_OFF)
            pygame.gfxdraw.aacircle(self.screen, int_pos[0], int_pos[1], self.LIGHT_RADIUS, self.COLOR_LIGHT_OFF)
            
        if is_selected:
            pygame.gfxdraw.aacircle(self.screen, int_pos[0], int_pos[1], self.LIGHT_RADIUS + 3, self.COLOR_LIGHT_SELECTED_BORDER)
            pygame.gfxdraw.aacircle(self.screen, int_pos[0], int_pos[1], self.LIGHT_RADIUS + 4, self.COLOR_LIGHT_SELECTED_BORDER)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"Illuminated: {self.score} / {self.NUM_SHADOWS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_text = self.font_small.render(f"Time: {max(0, self.time_left):.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))
        
        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score == self.NUM_SHADOWS:
                end_text = self.font_large.render("SUCCESS", True, self.COLOR_PROGRESS_BAR)
            else:
                end_text = self.font_large.render("TIME UP", True, (200, 50, 50))
                
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "illuminated_shadows": self.score,
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
    # It will not work with the "dummy" video driver, so we unset it.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Shadow Illuminator")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
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
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- ENV RESET ---")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # In a real game loop, you might wait for a key press before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(env.FPS)
        
    env.close()