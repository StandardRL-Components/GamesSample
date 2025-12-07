import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent must synchronize three interconnected gears.

    The agent's goal is to adjust the target speeds of three gears to make them
    all spin at exactly 50% speed simultaneously. The challenge lies in the
    interconnections: when a gear reaches 100% speed, its size changes, which
    in turn imparts a speed change to its adjacent gears, creating a chain reaction.
    The agent must manage these interactions to solve the puzzle within a 90-second time limit.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `action[0]` (Gear Selection): 1 for Gear 1, 2 for Gear 2, 3 for Gear 3. 0 and 4 are no-ops.
    - `action[1]` (Increase Speed): 1 to increase the selected gear's target speed (Spacebar).
    - `action[2]` (Decrease Speed): 1 to decrease the selected gear's target speed (Shift).

    **Observation Space:** `Box(0, 255, (400, 640, 3), dtype=np.uint8)`
    - An RGB image of the game state.

    **Reward Structure:**
    - **Continuous:** +0.1 per step for each gear within 5% of the 50% target speed, -0.01 otherwise.
    - **Event-based:** +5 for a gear's speed entering the target zone for the first time.
    - **Terminal:** +100 for winning (all gears synchronized), -100 for timeout.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Synchronize three interconnected gears to 50% speed by adjusting their spin rates, managing chain reactions from size changes."
    user_guide = "Controls: Use keys 1, 2, and 3 to select a gear. Hold space to increase speed and shift to decrease it."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_TIME_SECONDS = 90
    MAX_STEPS = MAX_TIME_SECONDS * FPS

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_TIMER_NORMAL = (100, 200, 100)
    COLOR_TIMER_WARN = (220, 50, 50)
    COLOR_SLOW = (50, 100, 255)
    COLOR_FAST = (255, 80, 80)
    COLOR_TARGET = (80, 255, 80)
    COLOR_GEAR_BODY = (120, 130, 150)
    COLOR_GEAR_OUTLINE = (80, 90, 110)
    COLOR_SELECTION_GLOW = (255, 255, 100, 50) # RGBA for transparency

    # Game Mechanics
    WIN_SPEED_TARGET = 50.0
    WIN_SPEED_TOLERANCE = 1.0
    REWARD_ZONE_TOLERANCE = 5.0
    SPEED_ADJUST_RATE = 0.5
    SPEED_INTERP_RATE = 0.05
    GEAR_SIZES = [40, 60, 80] # small, medium, large radii
    GEAR_POSITIONS = [
        (SCREEN_WIDTH * 0.25, SCREEN_HEIGHT * 0.5),
        (SCREEN_WIDTH * 0.50, SCREEN_HEIGHT * 0.5),
        (SCREEN_WIDTH * 0.75, SCREEN_HEIGHT * 0.5)
    ]
    GEAR_TEETH_COUNT = [10, 15, 20]
    SIZE_CHANGE_NUDGE_FACTOR = 15.0


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.ui_font_large = pygame.font.SysFont("Consolas", 30, bold=True)
            self.ui_font_small = pygame.font.SysFont("Consolas", 18)
            self.ui_font_tiny = pygame.font.SysFont("Consolas", 14)
        except pygame.error:
            self.ui_font_large = pygame.font.Font(None, 40)
            self.ui_font_small = pygame.font.Font(None, 24)
            self.ui_font_tiny = pygame.font.Font(None, 20)

        # Initialize state variables
        self.gears = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0.0
        self.selected_gear_idx = 0
        self._was_in_reward_zone = [False, False, False]


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_TIME_SECONDS
        self.selected_gear_idx = 0
        self._initialize_gears()

        return self._get_observation(), self._get_info()

    def _initialize_gears(self):
        self.gears = []
        for i in range(3):
            initial_speed = self.np_random.uniform(20.0, 80.0)
            initial_size_idx = self.np_random.integers(0, len(self.GEAR_SIZES))
            gear = {
                "pos": self.GEAR_POSITIONS[i],
                "current_speed": initial_speed,
                "target_speed": initial_speed,
                "size_idx": initial_size_idx,
                "radius": self.GEAR_SIZES[initial_size_idx],
                "num_teeth": self.GEAR_TEETH_COUNT[initial_size_idx],
                "angle": self.np_random.uniform(0, 360),
                "hit_100_flag": False,
            }
            self.gears.append(gear)
        
        self._was_in_reward_zone = [self._is_in_reward_zone(i) for i in range(3)]


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self._handle_action(action)
        self._update_game_state()

        reward = self._calculate_reward()
        self.score += reward
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.timer <= 0:
                reward -= 100 # Timeout penalty
            elif terminated:
                reward += 100 # Win bonus
            self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if 1 <= movement <= 3:
            self.selected_gear_idx = movement - 1

        if self.selected_gear_idx is not None:
            gear = self.gears[self.selected_gear_idx]
            if space_held and not shift_held:
                gear["target_speed"] += self.SPEED_ADJUST_RATE
            elif shift_held and not space_held:
                gear["target_speed"] -= self.SPEED_ADJUST_RATE
            
            gear["target_speed"] = np.clip(gear["target_speed"], 0, 100)

    def _update_game_state(self):
        self.timer = max(0, self.timer - 1.0 / self.FPS)

        # Update speeds and check for size changes
        for i, gear in enumerate(self.gears):
            # Interpolate speed
            speed_diff = gear["target_speed"] - gear["current_speed"]
            gear["current_speed"] += speed_diff * self.SPEED_INTERP_RATE
            gear["current_speed"] = np.clip(gear["current_speed"], 0, 100)

            # Check for 100% speed trigger
            if gear["current_speed"] >= 99.9 and not gear["hit_100_flag"]:
                gear["hit_100_flag"] = True
                self._trigger_size_change(i)
            elif gear["current_speed"] < 99.9:
                gear["hit_100_flag"] = False

            # Update rotation angle
            rotation_direction = 1 if i % 2 == 0 else -1
            gear["angle"] += (gear["current_speed"] / 10.0) * rotation_direction
            gear["angle"] %= 360

    def _trigger_size_change(self, gear_idx):
        gear = self.gears[gear_idx]
        old_radius = gear["radius"]

        gear["size_idx"] = (gear["size_idx"] + 1) % len(self.GEAR_SIZES)
        new_radius = self.GEAR_SIZES[gear["size_idx"]]
        gear["radius"] = new_radius
        gear["num_teeth"] = self.GEAR_TEETH_COUNT[gear["size_idx"]]
        
        radius_ratio = old_radius / new_radius if new_radius > 0 else 1
        nudge_amount = self.SIZE_CHANGE_NUDGE_FACTOR * (1 - radius_ratio)

        if gear_idx > 0: # Nudge left neighbor
            self.gears[gear_idx - 1]["current_speed"] -= nudge_amount
            self.gears[gear_idx - 1]["current_speed"] = np.clip(self.gears[gear_idx - 1]["current_speed"], 0, 100)

        if gear_idx < len(self.gears) - 1: # Nudge right neighbor
            self.gears[gear_idx + 1]["current_speed"] -= nudge_amount
            self.gears[gear_idx + 1]["current_speed"] = np.clip(self.gears[gear_idx + 1]["current_speed"], 0, 100)

    def _is_in_reward_zone(self, gear_idx):
        return abs(self.gears[gear_idx]["current_speed"] - self.WIN_SPEED_TARGET) <= self.REWARD_ZONE_TOLERANCE

    def _calculate_reward(self):
        reward = 0.0
        for i in range(3):
            is_in_zone = self._is_in_reward_zone(i)
            if is_in_zone:
                reward += 0.1 # Continuous reward for being in the zone
                if not self._was_in_reward_zone[i]:
                    reward += 5.0 # Bonus for entering the zone
            else:
                reward -= 0.01 # Small penalty for being out of zone
            self._was_in_reward_zone[i] = is_in_zone
        return reward

    def _check_termination(self):
        if self.timer <= 0:
            return True
        
        win = all(
            abs(g["current_speed"] - self.WIN_SPEED_TARGET) <= self.WIN_SPEED_TOLERANCE
            for g in self.gears
        )
        if win:
            return True

        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "gear_speeds": [g["current_speed"] for g in self.gears],
            "win_condition": self._check_termination() and self.timer > 0
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw connecting lines
        p1, p2, p3 = self.GEAR_POSITIONS
        pygame.draw.line(self.screen, self.COLOR_GEAR_OUTLINE, (p1[0], p1[1]-5), (p2[0], p2[1]-5), 2)
        pygame.draw.line(self.screen, self.COLOR_GEAR_OUTLINE, (p2[0], p2[1]+5), (p3[0], p3[1]+5), 2)

        for i, gear in enumerate(self.gears):
            # Draw selection glow
            if i == self.selected_gear_idx:
                glow_radius = gear["radius"] + 15
                s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, self.COLOR_SELECTION_GLOW, (glow_radius, glow_radius), glow_radius)
                self.screen.blit(s, (int(gear["pos"][0] - glow_radius), int(gear["pos"][1] - glow_radius)))
            
            # Determine color based on speed
            speed_fraction = gear["current_speed"] / 100.0
            gear_color = (
                int(self.COLOR_SLOW[0] + (self.COLOR_FAST[0] - self.COLOR_SLOW[0]) * speed_fraction),
                int(self.COLOR_SLOW[1] + (self.COLOR_FAST[1] - self.COLOR_SLOW[1]) * speed_fraction),
                int(self.COLOR_SLOW[2] + (self.COLOR_FAST[2] - self.COLOR_SLOW[2]) * speed_fraction),
            )
            
            # Draw the gear
            self._draw_gear(
                self.screen, gear["pos"], gear["radius"], gear["angle"], gear["num_teeth"], gear_color
            )

            # Draw speed indicator arc
            self._draw_speed_arc(gear, gear_color)

    def _draw_gear(self, surface, pos, radius, angle_deg, num_teeth, color):
        tooth_height = radius * 0.2
        outer_radius = radius
        inner_radius = radius - tooth_height
        
        angle_rad = math.radians(angle_deg)
        tooth_angle = 2 * math.pi / (num_teeth * 2)

        points = []
        for i in range(num_teeth * 2):
            current_angle = angle_rad + i * tooth_angle
            r = outer_radius if i % 2 == 0 else inner_radius
            x = pos[0] + r * math.cos(current_angle)
            y = pos[1] + r * math.sin(current_angle)
            points.append((int(x), int(y)))

        pygame.draw.polygon(surface, self.COLOR_GEAR_BODY, points)
        pygame.draw.polygon(surface, self.COLOR_GEAR_OUTLINE, points, 2)
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius * 0.3), self.COLOR_GEAR_OUTLINE)
        pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]), int(radius * 0.3), self.COLOR_GEAR_OUTLINE)


    def _draw_speed_arc(self, gear, color):
        pos_x, pos_y = int(gear["pos"][0]), int(gear["pos"][1])
        radius = gear["radius"] + 8
        speed_fraction = gear["current_speed"] / 100.0
        target_fraction = gear["target_speed"] / 100.0
        
        # Draw background arc
        pygame.gfxdraw.arc(self.screen, pos_x, pos_y, radius, -90, 270, self.COLOR_GEAR_OUTLINE)
        
        # Draw current speed arc
        if speed_fraction > 0:
            end_angle = int(-90 + 360 * speed_fraction)
            pygame.gfxdraw.arc(self.screen, pos_x, pos_y, radius, -90, end_angle, color)
            pygame.gfxdraw.arc(self.screen, pos_x, pos_y, radius-1, -90, end_angle, color)

        # Draw target speed marker
        target_angle_rad = math.radians(-90 + 360 * target_fraction)
        tx1 = pos_x + (radius - 4) * math.cos(target_angle_rad)
        ty1 = pos_y + (radius - 4) * math.sin(target_angle_rad)
        tx2 = pos_x + (radius + 4) * math.cos(target_angle_rad)
        ty2 = pos_y + (radius + 4) * math.sin(target_angle_rad)
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (tx1, ty1), (tx2, ty2), 2)

        # Draw 50% target zone
        zone_radius = self.REWARD_ZONE_TOLERANCE / 100.0
        start_angle = int(-90 + 360 * (0.5 - zone_radius))
        end_angle = int(-90 + 360 * (0.5 + zone_radius))
        pygame.gfxdraw.arc(self.screen, pos_x, pos_y, radius + 2, start_angle, end_angle, self.COLOR_TARGET)
        pygame.gfxdraw.arc(self.screen, pos_x, pos_y, radius + 3, start_angle, end_angle, self.COLOR_TARGET)


    def _render_ui(self):
        # Timer
        timer_color = self.COLOR_TIMER_NORMAL if self.timer > self.MAX_TIME_SECONDS * 0.2 else self.COLOR_TIMER_WARN
        timer_text = self.ui_font_large.render(f"{self.timer:.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 5))
        
        # Score
        score_text = self.ui_font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Gear speed text
        for i, gear in enumerate(self.gears):
            speed_text = self.ui_font_small.render(f"{gear['current_speed']:.1f}%", True, self.COLOR_UI_TEXT)
            text_rect = speed_text.get_rect(center=(gear["pos"][0], gear["pos"][1] + gear["radius"] + 35))
            self.screen.blit(speed_text, text_rect)
            
            # Gear label
            label_text = self.ui_font_small.render(f"Gear {i+1}", True, self.COLOR_UI_TEXT)
            label_rect = label_text.get_rect(center=(gear["pos"][0], gear["pos"][1] - gear["radius"] - 30))
            self.screen.blit(label_text, label_rect)

        # Controls hint
        controls_text = self.ui_font_tiny.render(
            "Controls: 1/2/3 to Select | Space to Accelerate | Shift to Decelerate", True, self.COLOR_UI_TEXT
        )
        self.screen.blit(controls_text, (10, self.SCREEN_HEIGHT - controls_text.get_height() - 5))

        # Win/Loss Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.timer <= 0:
                msg = "TIME OUT"
                color = self.COLOR_TIMER_WARN
            else:
                msg = "SYNCHRONIZED!"
                color = self.COLOR_TARGET
            
            end_text = self.ui_font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)


    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    game_window = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gear Sync")
    clock = pygame.time.Clock()

    action = [0, 0, 0] # [movement, space, shift]
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()

        keys = pygame.key.get_pressed()
        
        # Map keyboard to MultiDiscrete action space
        move_action = 0
        if keys[pygame.K_1]: move_action = 1
        elif keys[pygame.K_2]: move_action = 2
        elif keys[pygame.K_3]: move_action = 3
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [move_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_window.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Win: {info['win_condition']}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(GameEnv.FPS)

    env.close()