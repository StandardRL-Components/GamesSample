import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:53:47.093024
# Source Brief: brief_01278.md
# Brief Index: 1278
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
        "Align the colored segments of the interconnected gears to match the target pattern at the top before the timer expires."
    )
    user_guide = (
        "Controls: Use ↑/↓ to rotate the left gear, ←/→ for the middle gear, and Space/Shift for the rightmost gear. "
        "Match the top colors to the target pattern."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 90 * self.FPS  # 90 seconds

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GEAR_BODY = (90, 100, 110)
        self.COLOR_GEAR_OUTLINE = (70, 80, 90)
        self.COLOR_AXIS = (120, 130, 140)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_TIMER_WARN = (255, 100, 100)
        self.COLOR_LOCK_OVERLAY = (255, 0, 0, 100)
        self.SEGMENT_COLORS = [
            (227, 52, 47),   # Red
            (64, 191, 64),   # Green
            (52, 152, 219),  # Blue
            (241, 196, 15)   # Yellow
        ]

        # Gear properties
        self.GEAR_SEGMENTS = 8
        self.GEAR_POSITIONS = [(160, 200), (320, 200), (480, 200)]
        self.GEAR_RADII = [100, 100, 50]
        self.GEAR_RATIOS = [1.0, -1.0, 2.0] # Relative to Gear 1

        # Gameplay
        self.LOCK_CHANCE = 0.20
        self.LOCK_DURATION = 2 * self.FPS # 2 seconds
        self.RAPID_ROTATION_WINDOW = 5 # steps

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
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # --- Persistent State (survives reset) ---
        self.base_rotation_speed = 2.0 # degrees per step

        # --- Initialize State ---
        self.gear_angles = [0.0, 0.0, 0.0]
        self.gear_colors = []
        self.gear_lock_timers = [0, 0, 0]
        self.target_pattern = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.last_rotation_info = {'step': -self.RAPID_ROTATION_WINDOW, 'gear': -1}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset episode-specific state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.gear_lock_timers = [0, 0, 0]
        self.last_rotation_info = {'step': -self.RAPID_ROTATION_WINDOW, 'gear': -1}

        # Randomize gear starting positions
        self.gear_angles = [self.np_random.uniform(0, 360) for _ in range(3)]

        # Generate new gear colors and target pattern
        self.gear_colors = [
            self.np_random.choice(len(self.SEGMENT_COLORS), size=self.GEAR_SEGMENTS).tolist()
            for _ in range(3)
        ]
        self.target_pattern = self.np_random.choice(len(self.SEGMENT_COLORS), size=3).tolist()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action: [movement, space, shift]
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- Action to Game Logic Mapping ---
        # Priority: Shift > Space > Movement
        # `controlled_gear` is the gear the player is directly trying to turn
        # `rotation_dir` is the direction: 1 for CW, -1 for CCW
        controlled_gear = -1
        rotation_dir = 0

        if shift_held:
            controlled_gear, rotation_dir = 2, -1 # Gear 3 CCW
        elif space_held:
            controlled_gear, rotation_dir = 2, 1  # Gear 3 CW
        elif movement == 4: # Right
            controlled_gear, rotation_dir = 1, -1 # Gear 2 CCW
        elif movement == 3: # Left
            controlled_gear, rotation_dir = 1, 1  # Gear 2 CW
        elif movement == 2: # Down
            controlled_gear, rotation_dir = 0, -1 # Gear 1 CCW
        elif movement == 1: # Up
            controlled_gear, rotation_dir = 0, 1  # Gear 1 CW

        # --- Update Game State ---
        prev_lock_timers = list(self.gear_lock_timers)

        # Update lock timers
        for i in range(3):
            if self.gear_lock_timers[i] > 0:
                self.gear_lock_timers[i] -= 1
                if self.gear_lock_timers[i] == 0:
                    reward += 1.0 # Reward for unlocking
                    # SFX: unlock_chime.wav

        # Apply rotation if a gear is controlled and not locked
        if controlled_gear != -1 and self.gear_lock_timers[controlled_gear] == 0:
            # SFX: gear_rotate_click.wav
            
            # Check for rapid rotation to apply lock
            if self.steps - self.last_rotation_info['step'] < self.RAPID_ROTATION_WINDOW and \
               self.last_rotation_info['gear'] == controlled_gear:
                if self.np_random.random() < self.LOCK_CHANCE:
                    self.gear_lock_timers[controlled_gear] = self.LOCK_DURATION
                    # SFX: gear_lock_fail.wav
            
            self.last_rotation_info = {'step': self.steps, 'gear': controlled_gear}

            # Calculate base rotation delta based on which gear is controlled
            base_delta = rotation_dir * self.base_rotation_speed
            if controlled_gear == 0: # Controlling Gear 1
                delta_g1 = base_delta
            elif controlled_gear == 1: # Controlling Gear 2
                delta_g1 = base_delta / self.GEAR_RATIOS[1]
            else: # Controlling Gear 3
                delta_g1 = base_delta / self.GEAR_RATIOS[2]

            # Apply rotations based on gear ratios relative to Gear 1
            for i in range(3):
                self.gear_angles[i] += delta_g1 * self.GEAR_RATIOS[i]
                self.gear_angles[i] %= 360

        # --- Calculate Continuous Reward ---
        # Alignment between Gear 1 and 2 (at G1's 0 deg, G2's 180 deg)
        if self._get_segment_color(0, 0) == self._get_segment_color(1, 180):
            reward += 0.1
        # Alignment between Gear 2 and 3 (at G2's 0 deg, G3's 180 deg)
        if self._get_segment_color(1, 0) == self._get_segment_color(2, 180):
            reward += 0.1
        
        self.score += reward
        self.steps += 1

        # --- Check Termination Conditions ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            current_top_colors = self._get_current_top_colors()
            is_win = current_top_colors == self.target_pattern
            
            if is_win:
                reward = 100.0
                # SFX: win_fanfare.wav
            else: # Time ran out
                reward = -100.0
                self.base_rotation_speed *= 1.20 # Increase difficulty
                # SFX: loss_buzzer.wav
            self.score += reward

        truncated = self.steps >= self.MAX_STEPS
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_segment_color(self, gear_index, world_angle_deg):
        """Gets the color index of a segment at a specific world angle."""
        gear_angle = self.gear_angles[gear_index]
        segment_angle = (world_angle_deg - gear_angle + 360) % 360
        segment_size = 360 / self.GEAR_SEGMENTS
        segment_index = int(segment_angle / segment_size)
        color_index = self.gear_colors[gear_index][segment_index]
        return color_index

    def _get_current_top_colors(self):
        """Gets the color indices of the top-most segments of all gears."""
        # Top of the screen is world angle 270 degrees in Pygame coords
        return [self._get_segment_color(i, 270) for i in range(3)]

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        current_top_colors = self._get_current_top_colors()
        if current_top_colors == self.target_pattern:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Draw connecting axes
        pygame.draw.line(self.screen, self.COLOR_AXIS, (0, 200), (self.WIDTH, 200), 4)

        current_top_colors = self._get_current_top_colors()
        is_win_state = current_top_colors == self.target_pattern

        for i in range(3):
            is_aligned_top = current_top_colors[i] == self.target_pattern[i] or is_win_state
            self._draw_gear(
                self.screen,
                self.GEAR_POSITIONS[i],
                self.GEAR_RADII[i],
                self.gear_angles[i],
                self.gear_colors[i],
                self.gear_lock_timers[i],
                is_aligned_top
            )

    def _draw_gear(self, surface, center, radius, angle_deg, colors, lock_timer, is_aligned_top):
        # Draw gear body
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, self.COLOR_GEAR_BODY)
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, self.COLOR_GEAR_OUTLINE)
        
        # Draw teeth
        num_teeth = self.GEAR_SEGMENTS * 2
        for i in range(num_teeth):
            tooth_angle = math.radians(i * (360 / num_teeth) + angle_deg)
            outer_p = (center[0] + (radius + 5) * math.cos(tooth_angle), center[1] + (radius + 5) * math.sin(tooth_angle))
            inner_p = (center[0] + radius * math.cos(tooth_angle), center[1] + radius * math.sin(tooth_angle))
            pygame.draw.line(surface, self.COLOR_GEAR_BODY, inner_p, outer_p, 5)

        # Draw colored segments
        segment_angle_size = 360 / self.GEAR_SEGMENTS
        for i in range(self.GEAR_SEGMENTS):
            color_index = colors[i]
            segment_color = self.SEGMENT_COLORS[color_index]
            start_angle = angle_deg + i * segment_angle_size
            end_angle = start_angle + segment_angle_size
            
            # Use gfxdraw pie for antialiased filled arcs
            pygame.gfxdraw.pie(surface, center[0], center[1], radius-2, int(start_angle), int(end_angle), segment_color)

        # Draw glow effect for top aligned segment
        if is_aligned_top:
            gear_idx = self.GEAR_POSITIONS.index(center)
            top_segment_color_idx = self._get_segment_color(gear_idx, 270)
            glow_color = list(self.SEGMENT_COLORS[top_segment_color_idx]) + [70] # Add alpha
            
            relative_top_angle = (270 - angle_deg + 360) % 360
            segment_idx = int(relative_top_angle / segment_angle_size)
            
            segment_start_angle = angle_deg + (segment_idx * segment_angle_size)
            segment_end_angle = segment_start_angle + segment_angle_size
            
            # Create a temporary surface for the glow
            glow_surface = pygame.Surface((radius*2+20, radius*2+20), pygame.SRCALPHA)
            pygame.gfxdraw.pie(glow_surface, radius+10, radius+10, radius+5,
                               int(segment_start_angle - 5),
                               int(segment_end_angle + 5),
                               glow_color)
            surface.blit(glow_surface, (center[0]-radius-10, center[1]-radius-10))

        # Draw central axis
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], 15, self.COLOR_AXIS)
        pygame.gfxdraw.aacircle(surface, center[0], center[1], 15, self.COLOR_GEAR_OUTLINE)

        # Draw lock overlay
        if lock_timer > 0:
            lock_surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(lock_surface, radius, radius, radius, self.COLOR_LOCK_OVERLAY)
            surface.blit(lock_surface, (center[0]-radius, center[1]-radius))

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        timer_color = self.COLOR_TIMER_WARN if time_left < 10 else self.COLOR_TEXT
        timer_text = self.font_large.render(f"{time_left:.2f}", True, timer_color)
        timer_rect = timer_text.get_rect(center=(self.WIDTH // 2, 30))
        self.screen.blit(timer_text, timer_rect)

        # Target Pattern
        target_label = self.font_small.render("Target:", True, self.COLOR_TEXT)
        self.screen.blit(target_label, (self.WIDTH - 150, 10))
        for i, color_index in enumerate(self.target_pattern):
            color = self.SEGMENT_COLORS[color_index]
            pygame.draw.rect(self.screen, color, (self.WIDTH - 150 + i * 40, 35, 30, 30))
            pygame.draw.rect(self.screen, self.COLOR_TEXT, (self.WIDTH - 150 + i * 40, 35, 30, 30), 1)

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gear Aligner")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Human Controls ---
        # Default action is "do nothing"
        movement, space, shift = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.1f}")
            # Visual feedback for game over
            font = pygame.font.SysFont("Consolas", 60, bold=True)
            win_text = font.render("ALIGNED!", True, (100, 255, 100))
            loss_text = font.render("TIME UP!", True, (255, 100, 100))
            
            # We need to get the observation again to draw the final text on it
            final_obs_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            
            if info['score'] > 0 and terminated and not truncated: # Win condition
                text_rect = win_text.get_rect(center=(env.WIDTH/2, env.HEIGHT/2 - 50))
                final_obs_surface.blit(win_text, text_rect)
            else:
                text_rect = loss_text.get_rect(center=(env.WIDTH/2, env.HEIGHT/2 - 50))
                final_obs_surface.blit(loss_text, text_rect)

            reset_prompt = env.font_medium.render("Press 'R' to restart", True, env.COLOR_TEXT)
            prompt_rect = reset_prompt.get_rect(center=(env.WIDTH/2, env.HEIGHT/2 + 20))
            final_obs_surface.blit(reset_prompt, prompt_rect)

            screen.blit(final_obs_surface, (0, 0))
            pygame.display.flip()

            # Wait for reset command
            waiting_for_reset = True
            while waiting_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            obs, info = env.reset()
                            waiting_for_reset = False
                        if event.key == pygame.K_ESCAPE:
                            waiting_for_reset = False
                            running = False
                clock.tick(15)
            continue

        # --- Rendering ---
        # The observation is already the rendered screen
        # We just need to convert it back to a surface to blit
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(env.FPS)
        
    pygame.quit()