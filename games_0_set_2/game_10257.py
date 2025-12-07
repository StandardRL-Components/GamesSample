import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:05:12.244927
# Source Brief: brief_00257.md
# Brief Index: 257
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gear Sync: A Gymnasium environment where the agent must synchronize two rotating gears
    to achieve a speed boost and complete 7 rotations within a time limit.

    The game prioritizes visual quality and satisfying game feel, with smooth,
    anti-aliased graphics, particle effects, and clear visual feedback for all actions.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Synchronize two rotating gears to gain a speed boost. "
        "Complete the required number of rotations before time runs out."
    )
    user_guide = (
        "Use arrow keys to control the gears. ↑/↓ rotate the left gear, and ←/→ rotate the right gear. "
        "Match their rotation markers to achieve sync for a speed boost."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_DURATION_SECONDS = 60
    FPS = 30  # Assumed for game logic timing
    MAX_STEPS = GAME_DURATION_SECONDS * FPS # 1800 steps

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 40, 55)
    COLOR_GEAR_BODY = (210, 220, 230)
    COLOR_GEAR_OUTLINE = (160, 170, 180)
    COLOR_SYNC_GLOW = (40, 200, 120, 100)
    COLOR_DESYNC_GLOW = (200, 40, 80, 100)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_PROGRESS_BAR_BG = (40, 60, 80)
    COLOR_PROGRESS_BAR_FG = (60, 180, 240)

    # Game Mechanics
    ROTATION_INPUT_SPEED = 1.5  # Degrees per step on input
    WIN_ROTATIONS = 7
    SYNC_TOLERANCE_DEG = 12.0
    SYNC_SPEED_BONUS = 1.25  # 25% speed boost

    GEAR_SIZE_CHANGE_INTERVAL_SECONDS = 3
    GEAR_SIZE_CHANGE_INTERVAL_STEPS = GEAR_SIZE_CHANGE_INTERVAL_SECONDS * FPS

    BASE_RADIUS = 75
    LARGE_RADIUS_MULT = 1.2
    SMALL_RADIUS_MULT = 0.8
    LARGE_SPEED_MULT = 0.66 # Slower when large
    SMALL_SPEED_MULT = 1.5  # Faster when small

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 18)
        self.font_small = pygame.font.SysFont("monospace", 14)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.terminated = False
        
        self.gear1_angle = 0.0
        self.gear2_angle = 0.0
        self.gear1_total_rotation = 0.0
        self.gear2_total_rotation = 0.0

        self.gear1_radius = self.BASE_RADIUS
        self.gear2_radius = self.BASE_RADIUS
        self.gear1_speed_mult = 1.0
        self.gear2_speed_mult = 1.0
        
        self.is_synced = False
        self.last_size_change_step = 0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.terminated = False

        self.gear1_angle = self.np_random.uniform(0, 360)
        self.gear2_angle = self.np_random.uniform(0, 360)
        self.gear1_total_rotation = 0.0
        self.gear2_total_rotation = 0.0

        self._set_gear_properties(1, 'medium')
        self._set_gear_properties(2, 'medium')

        self.is_synced = self._check_sync()
        self.last_size_change_step = 0

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, self.terminated, False, self._get_info()

        # --- 1. Unpack Action & Store Old State ---
        movement = action[0]
        was_synced = self.is_synced
        old_total_rot1 = self.gear1_total_rotation
        old_total_rot2 = self.gear2_total_rotation
        
        # --- 2. Update Game Logic ---
        self.steps += 1

        # Apply player input
        gear1_input_rot = 0.0
        gear2_input_rot = 0.0
        if movement == 1: gear1_input_rot = self.ROTATION_INPUT_SPEED
        elif movement == 2: gear1_input_rot = -self.ROTATION_INPUT_SPEED
        elif movement == 3: gear2_input_rot = self.ROTATION_INPUT_SPEED
        elif movement == 4: gear2_input_rot = -self.ROTATION_INPUT_SPEED

        # Calculate effective rotation speed
        sync_bonus = self.SYNC_SPEED_BONUS if self.is_synced else 1.0
        effective_rot1 = gear1_input_rot * self.gear1_speed_mult * sync_bonus
        effective_rot2 = gear2_input_rot * self.gear2_speed_mult * sync_bonus

        # Update gear angles and total rotation
        self.gear1_angle = (self.gear1_angle + effective_rot1) % 360
        self.gear2_angle = (self.gear2_angle + effective_rot2) % 360
        self.gear1_total_rotation += abs(effective_rot1)
        self.gear2_total_rotation += abs(effective_rot2)

        # Handle periodic gear size changes
        if self.steps - self.last_size_change_step >= self.GEAR_SIZE_CHANGE_INTERVAL_STEPS:
            self.last_size_change_step = self.steps
            self._set_gear_properties(1, self.np_random.choice(['small', 'large']))
            self._set_gear_properties(2, self.np_random.choice(['small', 'large']))
            # SFX Placeholder: whoosh or mechanical change sound

        # Update synchronization status
        self.is_synced = self._check_sync()

        # --- 3. Calculate Reward ---
        reward = 0.0
        
        # Continuous reward for rotation
        rot_delta1 = self.gear1_total_rotation - old_total_rot1
        rot_delta2 = self.gear2_total_rotation - old_total_rot2
        reward += (rot_delta1 + rot_delta2) * 0.01

        # Event-based reward for sync change
        if self.is_synced and not was_synced:
            reward += 1.0 # Gained sync
            # SFX Placeholder: positive chime, click sound
        elif not self.is_synced and was_synced:
            reward -= 0.5 # Lost sync
            # SFX Placeholder: negative buzz, dissonant sound

        # --- 4. Check Termination ---
        self.terminated = self._check_termination()
        
        # Goal-oriented terminal rewards
        if self.terminated:
            progress = min(self.gear1_total_rotation, self.gear2_total_rotation) / 360
            if progress >= self.WIN_ROTATIONS:
                reward += 100.0 # Win
                # SFX Placeholder: victory fanfare
            else:
                reward -= 100.0 # Lose by timeout
                # SFX Placeholder: failure buzzer
        
        self.score += reward

        # --- 5. Return Step Tuple ---
        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _set_gear_properties(self, gear_num, size_str):
        if size_str == 'large':
            radius = self.BASE_RADIUS * self.LARGE_RADIUS_MULT
            speed_mult = self.LARGE_SPEED_MULT
        elif size_str == 'small':
            radius = self.BASE_RADIUS * self.SMALL_RADIUS_MULT
            speed_mult = self.SMALL_SPEED_MULT
        else: # medium
            radius = self.BASE_RADIUS
            speed_mult = 1.0

        if gear_num == 1:
            self.gear1_radius = radius
            self.gear1_speed_mult = speed_mult
        else:
            self.gear2_radius = radius
            self.gear2_speed_mult = speed_mult

    def _check_sync(self):
        diff = abs(self.gear1_angle - self.gear2_angle)
        angle_diff = min(diff, 360 - diff)
        return angle_diff < self.SYNC_TOLERANCE_DEG
    
    def _check_termination(self):
        progress = min(self.gear1_total_rotation, self.gear2_total_rotation) / 360
        if progress >= self.WIN_ROTATIONS:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "progress": min(self.gear1_total_rotation, self.gear2_total_rotation) / 360,
            "is_synced": self.is_synced,
        }

    def _get_observation(self):
        # --- Clear screen ---
        self.screen.fill(self.COLOR_BG)

        # --- Render all game elements ---
        self._render_background()
        self._render_sync_glow()
        self._render_gears()
        
        # --- Render UI overlay ---
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for i in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

    def _render_sync_glow(self):
        color = self.COLOR_SYNC_GLOW if self.is_synced else self.COLOR_DESYNC_GLOW
        center = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        max_radius = self.BASE_RADIUS * 1.8
        
        # Draw multiple circles for a soft glow effect
        for i in range(15):
            alpha = color[3] * (1 - i / 15)
            radius = max_radius * (1 - i / 20)
            pygame.gfxdraw.filled_circle(
                self.screen, center[0], center[1], int(radius),
                (color[0], color[1], color[2], int(alpha))
            )

    def _render_gears(self):
        pos1 = (self.SCREEN_WIDTH // 4, self.SCREEN_HEIGHT // 2)
        pos2 = (self.SCREEN_WIDTH * 3 // 4, self.SCREEN_HEIGHT // 2)
        
        self._draw_gear(self.screen, pos1, int(self.gear1_radius), self.gear1_angle, 12)
        self._draw_gear(self.screen, pos2, int(self.gear2_radius), self.gear2_angle, 12)

    def _draw_gear(self, surface, center, radius, angle, num_teeth):
        # --- Draw teeth ---
        tooth_width = 10
        tooth_height = 10
        angle_step = 360 / num_teeth

        for i in range(num_teeth):
            tooth_angle = angle + i * angle_step
            
            p1 = self._point_on_circle(center, radius - tooth_width / 2, tooth_angle - 5)
            p2 = self._point_on_circle(center, radius + tooth_height, tooth_angle - 5)
            p3 = self._point_on_circle(center, radius + tooth_height, tooth_angle + 5)
            p4 = self._point_on_circle(center, radius - tooth_width / 2, tooth_angle + 5)
            
            points = [p1, p2, p3, p4]
            pygame.gfxdraw.aapolygon(surface, points, self.COLOR_GEAR_OUTLINE)
            pygame.gfxdraw.filled_polygon(surface, points, self.COLOR_GEAR_OUTLINE)

        # --- Draw body ---
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, self.COLOR_GEAR_BODY)
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, self.COLOR_GEAR_OUTLINE)
        
        # --- Draw inner details ---
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], int(radius * 0.3), self.COLOR_BG)
        pygame.gfxdraw.aacircle(surface, center[0], center[1], int(radius * 0.3), self.COLOR_GEAR_OUTLINE)
        
        # --- Draw rotation marker ---
        spoke_end = self._point_on_circle(center, radius, angle)
        pygame.draw.line(surface, self.COLOR_DESYNC_GLOW[:3], center, spoke_end, 3)

    def _point_on_circle(self, center, radius, angle_deg):
        angle_rad = math.radians(angle_deg)
        x = center[0] + radius * math.cos(angle_rad)
        y = center[1] + radius * math.sin(angle_rad)
        return (int(x), int(y))

    def _render_ui(self):
        # --- Timer ---
        time_elapsed = self.steps / self.FPS
        time_left = max(0, self.GAME_DURATION_SECONDS - time_elapsed)
        minutes, seconds = divmod(int(time_left), 60)
        timer_text = f"{minutes:02}:{seconds:02}"
        self._draw_text(timer_text, (self.SCREEN_WIDTH // 2, 40), self.font_large, self.COLOR_UI_TEXT)

        # --- Rotation Progress ---
        progress = min(self.gear1_total_rotation, self.gear2_total_rotation) / 360
        progress_text = f"ROTATIONS: {progress:.2f} / {self.WIN_ROTATIONS:.2f}"
        self._draw_text(progress_text, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 50), self.font_medium, self.COLOR_UI_TEXT)
        
        # --- Progress Bar ---
        bar_width = self.SCREEN_WIDTH - 80
        bar_height = 20
        bar_x = 40
        bar_y = self.SCREEN_HEIGHT - 30
        
        fill_ratio = min(1.0, progress / self.WIN_ROTATIONS)
        fill_width = int(bar_width * fill_ratio)
        
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_FG, (bar_x, bar_y, fill_width, bar_height), border_radius=5)

        # --- Sync Status Text ---
        sync_status_text = "SYNCED" if self.is_synced else "DE-SYNCED"
        sync_color = self.COLOR_SYNC_GLOW[:3] if self.is_synced else self.COLOR_DESYNC_GLOW[:3]
        self._draw_text(sync_status_text, (self.SCREEN_WIDTH // 2, 80), self.font_medium, sync_color)
        
    def _draw_text(self, text, pos, font, color):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=pos)
        self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    # The main loop needs a display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play Controls ---
    # W: Gear 1 CW, S: Gear 1 CCW
    # A: Gear 2 CW, D: Gear 2 CCW
    
    done = False
    total_reward = 0
    
    # Use pygame for human interaction
    pygame.display.set_caption("Gear Sync - Manual Control")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    while running:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action[0] = 1 # Gear 1 CW
        elif keys[pygame.K_s]:
            action[0] = 2 # Gear 1 CCW
            
        if keys[pygame.K_a]:
            action[0] = 3 # Gear 2 CW
        elif keys[pygame.K_d]:
            action[0] = 4 # Gear 2 CCW

        if keys[pygame.K_r]: # Reset
            obs, info = env.reset()
            total_reward = 0
            done = False
            continue

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # Wait for reset
            pass

        env.clock.tick(GameEnv.FPS)
        
    env.close()