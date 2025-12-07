import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:08:49.216776
# Source Brief: brief_01548.md
# Brief Index: 1548
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a clock synchronization puzzle game.

    The agent's goal is to manipulate the hour, minute, and second hands of a
    clock to align them at specific target times (12:00:00 or 3:00:00).
    The agent has a limited number of moves (steps) to achieve a target
    number of synchronizations.

    **Visuals:**
    The environment renders a minimalist, high-contrast clock face.
    - A dark background with a white clock face.
    - Hour, minute, and second hands are clearly distinguishable by size and color.
    - Target synchronization times are marked on the clock face.
    - Successful synchronizations trigger a green flash effect.
    - The hand that was just moved receives a temporary glow for immediate feedback.
    - A UI overlay displays the digital time, score, turns remaining, and syncs achieved.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `action[0]` (Movement):
        - 0: No operation
        - 1: Increment Hour Hand (+1 hour)
        - 2: Increment Minute Hand (+1 minute)
        - 3: Increment Second Hand (+1 second)
        - 4: No operation
    - `action[1]` (Space): Unused
    - `action[2]` (Shift): Unused

    **Reward Structure:**
    - +10 for each successful synchronization.
    - +50 bonus for winning the game (achieving the sync goal).
    - A continuous shaping reward based on the change in distance to the
      nearest target time. Moving closer gives a small positive reward,
      while moving further away gives a small negative reward.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Manipulate the hour, minute, and second hands of a clock to align them at specific target times."
    user_guide = "Use ↑ to advance the hour, ↓ for the minute, and ← for the second hand to match the target times."
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 60
    SYNC_GOAL = 5
    
    # --- Colors ---
    COLOR_BG = (15, 18, 26)
    COLOR_CLOCK_FACE = (200, 200, 210)
    COLOR_HOUR_HAND = (230, 230, 240)
    COLOR_MINUTE_HAND = (230, 230, 240)
    COLOR_SECOND_HAND = (255, 80, 80)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_TARGET_MARKER = (60, 180, 120, 150)
    COLOR_SYNC_FLASH = (40, 220, 150)
    COLOR_GLOW = (255, 255, 100, 100)
    COLOR_GAMEOVER_BG = (0, 0, 0, 180)

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
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_digital = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_gameover = pygame.font.SysFont("sans-serif", 60, bold=True)
        
        # --- Game State ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.sync_count = 0
        self.hour = 0
        self.minute = 0
        self.second = 0
        self.last_distance = 0.0
        self.last_action_type = 0 # 1=h, 2=m, 3=s
        self.flash_timer = 0
        self.action_glow_timer = 0

        # --- Clock Geometry ---
        self.center_x = self.SCREEN_WIDTH // 2
        self.center_y = self.SCREEN_HEIGHT // 2
        self.clock_radius = 140
        self.hour_len = self.clock_radius * 0.5
        self.minute_len = self.clock_radius * 0.75
        self.second_len = self.clock_radius * 0.9

        # Initialize state variables
        # self.reset() is called in validate_implementation

        # Critical self-check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.sync_count = 0
        self.flash_timer = 0
        self.last_action_type = 0
        self.action_glow_timer = 0

        # Randomize time, ensuring it's not a solution state
        self._scramble_time()
        while self._check_sync():
            self._scramble_time()

        self.last_distance = self._calculate_distance_to_nearest_target()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        movement = action[0]
        
        reward = 0.0
        self.steps += 1
        self.last_action_type = 0
        self.action_glow_timer = 0

        # --- Update Game Logic ---
        if movement in [1, 2, 3]:
            # Sound effect placeholder: pygame.mixer.Sound("tick.wav").play()
            self.last_action_type = movement
            self.action_glow_timer = 10 # Glow for 10 frames
            if movement == 1: # Increment Hour
                self.hour = (self.hour + 1) % 12
            elif movement == 2: # Increment Minute
                self.minute = (self.minute + 1) % 60
            elif movement == 3: # Increment Second
                self.second = (self.second + 1) % 60

        # --- Calculate Rewards ---
        # 1. Continuous shaping reward
        current_distance = self._calculate_distance_to_nearest_target()
        distance_delta = self.last_distance - current_distance
        shaping_reward = 0.1 * distance_delta
        reward += shaping_reward
        self.last_distance = current_distance

        # 2. Event-based reward for synchronization
        if self._check_sync():
            # Sound effect placeholder: pygame.mixer.Sound("sync_success.wav").play()
            self.sync_count += 1
            sync_reward = 10.0
            reward += sync_reward
            self.flash_timer = 15 # Flash for 15 frames
            self._scramble_time() # Create a new puzzle
            self.last_distance = self._calculate_distance_to_nearest_target()

        # --- Update Score ---
        self.score += reward

        # --- Check Termination ---
        terminated = self.steps >= self.MAX_STEPS or self.sync_count >= self.SYNC_GOAL
        truncated = False # This game is not truncated
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.sync_count >= self.SYNC_GOAL:
                # Sound effect placeholder: pygame.mixer.Sound("win_game.wav").play()
                win_bonus = 50.0
                reward += win_bonus
                self.score += win_bonus
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        # Handle flash effect after all rendering
        if self.flash_timer > 0:
            flash_surface = self.screen.copy()
            flash_surface.fill(self.COLOR_SYNC_FLASH)
            alpha = int(255 * (self.flash_timer / 15.0))
            flash_surface.set_alpha(alpha)
            self.screen.blit(flash_surface, (0, 0))
            self.flash_timer -= 1
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Clock Face ---
        pygame.gfxdraw.aacircle(self.screen, self.center_x, self.center_y, self.clock_radius, self.COLOR_CLOCK_FACE)
        pygame.gfxdraw.aacircle(self.screen, self.center_x, self.center_y, self.clock_radius - 1, self.COLOR_CLOCK_FACE)
        pygame.gfxdraw.filled_circle(self.screen, self.center_x, self.center_y, 5, self.COLOR_CLOCK_FACE)

        # --- Draw Tick Marks and Target Markers ---
        for i in range(12):
            angle = math.radians(i * 30 - 90)
            is_target = i == 0 or i == 3
            is_major = i % 3 == 0
            
            if is_target:
                tx = self.center_x + int(math.cos(angle) * (self.clock_radius - 10))
                ty = self.center_y + int(math.sin(angle) * (self.clock_radius - 10))
                pygame.gfxdraw.filled_circle(self.screen, tx, ty, 6, self.COLOR_TARGET_MARKER)
                pygame.gfxdraw.aacircle(self.screen, tx, ty, 6, self.COLOR_TARGET_MARKER)

            start_r = self.clock_radius - (15 if is_major else 8)
            end_r = self.clock_radius
            start_pos = (self.center_x + int(math.cos(angle) * start_r), self.center_y + int(math.sin(angle) * start_r))
            end_pos = (self.center_x + int(math.cos(angle) * end_r), self.center_y + int(math.sin(angle) * end_r))
            pygame.draw.aaline(self.screen, self.COLOR_CLOCK_FACE, start_pos, end_pos, 1)

        # --- Calculate Hand Angles (0 degrees is UP) ---
        hour_angle_deg = (self.hour % 12 + self.minute / 60) * 30
        minute_angle_deg = (self.minute + self.second / 60) * 6
        second_angle_deg = self.second * 6
        
        # --- Draw Hands ---
        if self.action_glow_timer > 0:
            glow_alpha = int(self.COLOR_GLOW[3] * (self.action_glow_timer / 10.0))
            glow_color = (*self.COLOR_GLOW[:3], glow_alpha)
            if self.last_action_type == 1: self._draw_hand(hour_angle_deg, self.hour_len, 10, glow_color)
            if self.last_action_type == 2: self._draw_hand(minute_angle_deg, self.minute_len, 8, glow_color)
            if self.last_action_type == 3: self._draw_hand(second_angle_deg, self.second_len, 6, glow_color)
            self.action_glow_timer -= 1

        self._draw_hand(hour_angle_deg, self.hour_len, 6, self.COLOR_HOUR_HAND)
        self._draw_hand(minute_angle_deg, self.minute_len, 4, self.COLOR_MINUTE_HAND)
        self._draw_hand(second_angle_deg, self.second_len, 2, self.COLOR_SECOND_HAND)
        pygame.gfxdraw.filled_circle(self.screen, self.center_x, self.center_y, 7, self.COLOR_SECOND_HAND)
        pygame.gfxdraw.aacircle(self.screen, self.center_x, self.center_y, 7, self.COLOR_SECOND_HAND)

    def _draw_hand(self, angle_deg, length, width, color):
        angle_rad = math.radians(angle_deg - 90)
        end_x = self.center_x + length * math.cos(angle_rad)
        end_y = self.center_y + length * math.sin(angle_rad)
        
        # Create a list of points for a polygon to represent the hand
        points = []
        for i in range(-1, 2, 2):
            offset_angle = angle_rad + math.pi/2 * i
            points.append((
                self.center_x + (width/2) * math.cos(offset_angle),
                self.center_y + (width/2) * math.sin(offset_angle)
            ))
        points.append((end_x, end_y))
        
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_ui(self):
        # --- Digital Clock ---
        display_hour = 12 if self.hour == 0 else self.hour
        time_str = f"{display_hour:02d}:{self.minute:02d}:{self.second:02d}"
        self._render_text(time_str, (self.center_x, 40), self.font_digital, self.COLOR_UI_TEXT, center_x=True)

        # --- Game Stats ---
        score_str = f"Score: {int(self.score)}"
        turns_str = f"Turns: {self.steps}/{self.MAX_STEPS}"
        syncs_str = f"Syncs: {self.sync_count}/{self.SYNC_GOAL}"
        self._render_text(score_str, (20, 20), self.font_ui, self.COLOR_UI_TEXT)
        self._render_text(turns_str, (20, 40), self.font_ui, self.COLOR_UI_TEXT)
        self._render_text(syncs_str, (20, 60), self.font_ui, self.COLOR_UI_TEXT)
        
        # --- Action Helper ---
        self._render_text("UP: Hour", (self.SCREEN_WIDTH - 120, 20), self.font_ui, self.COLOR_UI_TEXT)
        self._render_text("DOWN: Minute", (self.SCREEN_WIDTH - 120, 40), self.font_ui, self.COLOR_UI_TEXT)
        self._render_text("LEFT: Second", (self.SCREEN_WIDTH - 120, 60), self.font_ui, self.COLOR_UI_TEXT)

        # --- Game Over Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_GAMEOVER_BG)
            self.screen.blit(overlay, (0, 0))
            
            if self.sync_count >= self.SYNC_GOAL:
                msg = "SYNCHRONIZED!"
                color = self.COLOR_SYNC_FLASH
            else:
                msg = "OUT OF TIME"
                color = self.COLOR_SECOND_HAND
            self._render_text(msg, (self.center_x, self.center_y), self.font_gameover, color, center_x=True, center_y=True)

    def _render_text(self, text, position, font, color, center_x=False, center_y=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center_x: text_rect.centerx = position[0]
        else: text_rect.x = position[0]
        if center_y: text_rect.centery = position[1]
        else: text_rect.y = position[1]
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sync_count": self.sync_count,
            "time": (self.hour, self.minute, self.second),
        }

    def _calculate_distance_to_nearest_target(self):
        # Target 1: 12:00:00 (h=0, m=0, s=0)
        # Target 2: 03:00:00 (h=3, m=0, s=0)
        
        dist_h1 = min(abs(self.hour - 0), 12 - abs(self.hour - 0))
        dist_m1 = min(abs(self.minute - 0), 60 - abs(self.minute - 0))
        dist_s1 = min(abs(self.second - 0), 60 - abs(self.second - 0))
        total_dist1 = dist_h1 + dist_m1 + dist_s1

        dist_h2 = min(abs(self.hour - 3), 12 - abs(self.hour - 3))
        dist_m2 = min(abs(self.minute - 0), 60 - abs(self.minute - 0))
        dist_s2 = min(abs(self.second - 0), 60 - abs(self.second - 0))
        total_dist2 = dist_h2 + dist_m2 + dist_s2

        return min(total_dist1, total_dist2)

    def _check_sync(self):
        is_sync1 = self.hour == 0 and self.minute == 0 and self.second == 0
        is_sync2 = self.hour == 3 and self.minute == 0 and self.second == 0
        return is_sync1 or is_sync2

    def _scramble_time(self):
        """Sets the clock to a new random time."""
        self.hour = self.np_random.integers(0, 12)
        self.minute = self.np_random.integers(0, 60)
        self.second = self.np_random.integers(0, 60)
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        obs, info = self.reset() # Reset to get a valid initial state
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        
        # Test reset again to ensure it's idempotent
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")
        self.reset() # Reset again to leave the env in a clean state

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human interaction
    pygame.display.set_caption("Clock Sync Puzzle - Manual Control")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("--- Manual Control ---")
    print("UP ARROW:   Increment Hour")
    print("DOWN ARROW: Increment Minute")
    print("LEFT ARROW: Increment Second")
    print("R:          Reset Environment")
    print("Q:          Quit")

    while running:
        action = np.array([0, 0, 0]) # Default action is no-op
        
        # Get observation before processing events
        current_obs = env._get_observation()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("\n--- Environment Reset ---")
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
        
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Syncs: {info['sync_count']}")
            
            if terminated:
                print("\n--- Episode Finished ---")
                print(f"Final Score: {info['score']:.2f}, Syncs: {info['sync_count']}/{env.SYNC_GOAL}")
                # Update obs for final frame render
                obs = env._get_observation()
        else:
            obs = current_obs

        # Render the environment observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS
        
    env.close()