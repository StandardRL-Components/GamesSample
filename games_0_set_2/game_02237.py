
# Generated: 2025-08-28T04:18:11.635013
# Source Brief: brief_02237.md
# Brief Index: 2237

        
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
        "Controls: Use ↑/↓ to select a gear. Press Space to rotate it clockwise, "
        "or Shift to rotate counter-clockwise. Set the clock to 12:00."
    )

    game_description = (
        "A steampunk puzzle. Manipulate interconnected gears to set the clock "
        "to 12:00 before you run out of moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_clock = pygame.font.SysFont("serif", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("serif", 48, bold=True)

        # Colors
        self.COLOR_BG = (25, 20, 20)
        self.COLOR_BG_PATTERN = (35, 30, 30)
        self.COLOR_GEAR_BRASS = (205, 127, 50)
        self.COLOR_GEAR_BRASS_DARK = (165, 97, 20)
        self.COLOR_CLOCK_FACE = (10, 10, 10)
        self.COLOR_CLOCK_BORDER = (50, 40, 30)
        self.COLOR_HAND_HOUR = (220, 50, 50)
        self.COLOR_HAND_MINUTE = (200, 80, 80)
        self.COLOR_UI_TEXT = (220, 220, 200)
        self.COLOR_SELECT_GLOW = (255, 255, 150)

        # Game constants
        self.MAX_MOVES = 20
        self.TOTAL_MINUTES_IN_CLOCK = 12 * 60

        # Game entities
        self.gears = [
            {
                "pos": (self.width * 0.25, self.height * 0.35),
                "radius": 50, "teeth": 12, "ratio": 30, # Coarse control: 30 mins
            },
            {
                "pos": (self.width * 0.5, self.height * 0.65),
                "radius": 70, "teeth": 20, "ratio": -10, # Medium control: -10 mins
            },
            {
                "pos": (self.width * 0.75, self.height * 0.35),
                "radius": 35, "teeth": 8, "ratio": 1, # Fine control: 1 min
            },
        ]
        self.clock_center = (self.width // 2, self.height // 2)
        self.clock_radius = 120

        # Initialize state variables
        self.total_minutes = 0
        self.moves_remaining = 0
        self.selected_gear_idx = 0
        self.gear_angles = []
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.last_action_was_rotation = False
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.total_minutes = self.np_random.integers(1, self.TOTAL_MINUTES_IN_CLOCK)
        self.moves_remaining = self.MAX_MOVES
        self.selected_gear_idx = 0
        self.gear_angles = [self.np_random.uniform(0, 360) for _ in self.gears]
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.last_action_was_rotation = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return (
                self._get_observation(),
                0,
                True,
                False,
                self._get_info(),
            )

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        self.last_action_was_rotation = False
        reward = 0

        # 1. Handle gear selection
        if movement == 1:  # Up
            self.selected_gear_idx = (self.selected_gear_idx - 1) % len(self.gears)
        elif movement == 2:  # Down
            self.selected_gear_idx = (self.selected_gear_idx + 1) % len(self.gears)

        # 2. Handle gear rotation
        rotation_dir = 0
        if space_held:
            rotation_dir = 1  # Clockwise
        elif shift_held:
            rotation_dir = -1 # Counter-Clockwise

        if rotation_dir != 0:
            self.last_action_was_rotation = True
            
            # Store state for reward calculation
            old_dist_to_target = self._minutes_distance_to_target(self.total_minutes)
            old_hour = (self.total_minutes // 60) % 12

            # Update game state
            gear = self.gears[self.selected_gear_idx]
            self.total_minutes = (self.total_minutes + rotation_dir * gear["ratio"]) % self.TOTAL_MINUTES_IN_CLOCK
            self.gear_angles[self.selected_gear_idx] += rotation_dir * (360 / gear["teeth"])
            self.moves_remaining -= 1

            # Calculate continuous reward
            new_dist_to_target = self._minutes_distance_to_target(self.total_minutes)
            reward = old_dist_to_target - new_dist_to_target # Positive if closer
            
            # Calculate event-based reward
            new_hour = (self.total_minutes // 60) % 12
            if new_hour == 0 and old_hour != 0:
                reward += 5

        self.score += reward
        terminated = self._check_termination()
        
        # Calculate terminal rewards
        if terminated:
            if self.win:
                reward += 100
                self.score += 100
            else: # Ran out of moves
                reward -= 50
                self.score -= 50

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _minutes_distance_to_target(self, minutes):
        # Target is 0 (12:00). Distance on a circle.
        return min(minutes, self.TOTAL_MINUTES_IN_CLOCK - minutes)

    def _check_termination(self):
        is_win = self.total_minutes == 0
        is_loss = self.moves_remaining <= 0 and not is_win
        
        if is_win:
            self.game_over = True
            self.win = True
        elif is_loss:
            self.game_over = True
            self.win = False
            
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_pattern()
        self._render_clock()
        self._render_gears()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_pattern(self):
        for i in range(0, self.width, 50):
            for j in range(0, self.height, 50):
                pygame.gfxdraw.aacircle(self.screen, i, j, 20, self.COLOR_BG_PATTERN)

    def _render_clock(self):
        # Face
        pygame.gfxdraw.filled_circle(self.screen, self.clock_center[0], self.clock_center[1], self.clock_radius, self.COLOR_CLOCK_FACE)
        pygame.gfxdraw.aacircle(self.screen, self.clock_center[0], self.clock_center[1], self.clock_radius, self.COLOR_CLOCK_BORDER)
        pygame.gfxdraw.aacircle(self.screen, self.clock_center[0], self.clock_center[1], self.clock_radius-1, self.COLOR_CLOCK_BORDER)

        # Numerals and ticks
        for i in range(12):
            angle = math.radians(i * 30 - 90)
            text_x = self.clock_center[0] + (self.clock_radius - 20) * math.cos(angle)
            text_y = self.clock_center[1] + (self.clock_radius - 20) * math.sin(angle)
            numeral = str(i if i != 0 else 12)
            text_surf = self.font_clock.render(numeral, True, self.COLOR_UI_TEXT)
            self.screen.blit(text_surf, text_surf.get_rect(center=(int(text_x), int(text_y))))
            
            tick_start_x = self.clock_center[0] + (self.clock_radius - 5) * math.cos(angle)
            tick_start_y = self.clock_center[1] + (self.clock_radius - 5) * math.sin(angle)
            tick_end_x = self.clock_center[0] + (self.clock_radius - 10) * math.cos(angle)
            tick_end_y = self.clock_center[1] + (self.clock_radius - 10) * math.sin(angle)
            pygame.draw.aaline(self.screen, self.COLOR_UI_TEXT, (tick_start_x, tick_start_y), (tick_end_x, tick_end_y))

        # Hands
        minute_angle = math.radians((self.total_minutes % 60) * 6 - 90)
        hour_angle = math.radians(((self.total_minutes / 60) % 12) * 30 - 90)
        
        self._draw_hand(self.clock_radius * 0.85, minute_angle, self.COLOR_HAND_MINUTE, 3)
        self._draw_hand(self.clock_radius * 0.6, hour_angle, self.COLOR_HAND_HOUR, 5)

        # Center pin
        pygame.gfxdraw.filled_circle(self.screen, self.clock_center[0], self.clock_center[1], 8, self.COLOR_GEAR_BRASS)
        pygame.gfxdraw.aacircle(self.screen, self.clock_center[0], self.clock_center[1], 8, self.COLOR_GEAR_BRASS_DARK)

    def _draw_hand(self, length, angle, color, width):
        end_x = self.clock_center[0] + length * math.cos(angle)
        end_y = self.clock_center[1] + length * math.sin(angle)
        pygame.draw.line(self.screen, color, self.clock_center, (end_x, end_y), width)
        # Antialiasing line cap
        pygame.gfxdraw.filled_circle(self.screen, int(end_x), int(end_y), width // 2, color)
        pygame.gfxdraw.aacircle(self.screen, int(end_x), int(end_y), width // 2, color)


    def _render_gears(self):
        for i, gear in enumerate(self.gears):
            is_selected = (i == self.selected_gear_idx)
            self._draw_gear(
                pos=gear["pos"],
                radius=gear["radius"],
                teeth=gear["teeth"],
                angle=self.gear_angles[i],
                color=self.COLOR_GEAR_BRASS,
                dark_color=self.COLOR_GEAR_BRASS_DARK,
                is_selected=is_selected,
            )

    def _draw_gear(self, pos, radius, teeth, angle, color, dark_color, is_selected):
        # Selection glow
        if is_selected:
            glow_radius = radius + 10
            for i in range(5):
                alpha = 80 - i * 15
                glow_color = (*self.COLOR_SELECT_GLOW, alpha)
                temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, glow_radius - i*2, glow_color)
                self.screen.blit(temp_surf, (pos[0] - glow_radius, pos[1] - glow_radius))

        # Teeth
        tooth_length = radius * 1.15
        for i in range(teeth):
            current_angle_rad = math.radians(angle + (i * 360 / teeth))
            start_pos = (
                pos[0] + radius * 0.8 * math.cos(current_angle_rad),
                pos[1] + radius * 0.8 * math.sin(current_angle_rad),
            )
            end_pos = (
                pos[0] + tooth_length * math.cos(current_angle_rad),
                pos[1] + tooth_length * math.sin(current_angle_rad),
            )
            pygame.draw.line(self.screen, color, start_pos, end_pos, max(3, int(radius / 10)))

        # Body
        pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(radius), color)
        pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), int(radius), dark_color)
        
        # Hub
        hub_radius = int(radius * 0.3)
        pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), hub_radius, dark_color)
        pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), hub_radius, color)


    def _render_ui(self):
        # Time display
        hours = (self.total_minutes // 60) % 12
        if hours == 0: hours = 12
        minutes = self.total_minutes % 60
        time_str = f"Time: {hours:02d}:{minutes:02d}"
        time_surf = self.font_ui.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (10, 10))
        
        target_surf = self.font_ui.render("Target: 12:00", True, self.COLOR_UI_TEXT)
        self.screen.blit(target_surf, (10, 35))

        # Moves display
        moves_str = f"Moves: {self.moves_remaining}"
        moves_surf = self.font_ui.render(moves_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_surf, (self.width - moves_surf.get_width() - 10, 10))
        
        # Action feedback
        if self.last_action_was_rotation:
            gear = self.gears[self.selected_gear_idx]
            ratio = gear["ratio"]
            feedback_str = f"Time {'+' if ratio > 0 else ''}{ratio} min"
            feedback_color = (100, 255, 100) if ratio > 0 else (255, 100, 100)
            feedback_surf = self.font_ui.render(feedback_str, True, feedback_color)
            self.screen.blit(feedback_surf, (self.width - feedback_surf.get_width() - 10, 35))

    def _render_game_over(self):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        text = "PUZZLE SOLVED" if self.win else "OUT OF MOVES"
        color = (150, 255, 150) if self.win else (255, 150, 150)
        
        text_surf = self.font_game_over.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.width / 2, self.height / 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "total_minutes": self.total_minutes,
            "win": self.win,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Clockwork Enigma")
    
    running = True
    terminated = False
    
    # Game loop
    while running:
        action = np.array([0, 0, 0])  # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            
            # Map keys to MultiDiscrete action space
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            
            if keys[pygame.K_SPACE]:
                action[1] = 1
            
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Since auto_advance is False, we need a delay to make it playable by humans
        pygame.time.wait(100) # 10 FPS for human playability
        
    env.close()