
# Generated: 2025-08-28T06:42:46.522330
# Source Brief: brief_02997.md
# Brief Index: 2997

        
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
        "Controls: ↑↓ to aim, ←→ to set power. Press Space to shoot. Hold Shift to reset arm."
    )

    game_description = (
        "Score 10 points by skillfully flinging a basketball into a hoop using a single, articulated robotic arm."
    )

    auto_advance = False

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_TOP = (100, 150, 255)
    COLOR_BG_BOTTOM = (150, 200, 255)
    COLOR_PLATFORM = (90, 100, 110)
    COLOR_PLATFORM_SHADOW = (60, 70, 80)
    COLOR_ARM = (180, 190, 200)
    COLOR_ARM_OUTLINE = (120, 130, 140)
    COLOR_BALL = (255, 100, 0)
    COLOR_BALL_OUTLINE = (200, 60, 0)
    COLOR_BACKBOARD = (230, 230, 230)
    COLOR_BACKBOARD_OUTLINE = (150, 150, 150)
    COLOR_HOOP = (255, 60, 30)
    COLOR_HOOP_FLASH = (255, 255, 255)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_SHADOW = (0, 0, 0, 128)
    COLOR_X = (255, 0, 0)

    # Physics
    GRAVITY = 0.4
    BOUNCE_DAMPENING = 0.7
    MAX_STEPS = 1000

    # Game State
    WIN_SCORE = 10
    LOSE_MISSES = 5

    # Arm
    ARM_BASE_POS = (100, 300)
    ARM_SEGMENT_LEN = 60
    ARM_MIN_ANGLE = 10  # Degrees from horizontal
    ARM_MAX_ANGLE = 85
    ARM_MIN_POWER = 0.2 # Extension multiplier
    ARM_MAX_POWER = 1.0
    ARM_ANGLE_SPEED = 2.0
    ARM_POWER_SPEED = 0.05
    
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
        
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # Hoop and backboard setup
        self.hoop_pos = (self.SCREEN_WIDTH - 100, 150)
        self.backboard_rect = pygame.Rect(self.hoop_pos[0] + 10, self.hoop_pos[1] - 40, 10, 80)
        self.hoop_rim_left = (self.hoop_pos[0] - 25, self.hoop_pos[1])
        self.hoop_rim_right = (self.hoop_pos[0] + 5, self.hoop_pos[1])
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.missed_shots = 0
        self.game_over = False
        self.arm_angle = 0.0
        self.arm_power = 0.0
        self.ball_state = "ready"
        self.ball_pos = np.array([0.0, 0.0])
        self.ball_vel = np.array([0.0, 0.0])
        self.ball_trail = []
        self.flash_timer = 0
        self.last_dist_to_hoop = float('inf')

        self.np_random = None

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.missed_shots = 0
        self.game_over = False
        
        self.arm_angle = 45.0
        self.arm_power = (self.ARM_MIN_POWER + self.ARM_MAX_POWER) / 2
        
        self._reset_ball()
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_state = "ready"
        self.ball_pos = np.array([0.0, 0.0]) # Position relative to arm tip
        self.ball_vel = np.array([0.0, 0.0])
        self.ball_trail = []
        self.last_dist_to_hoop = self._get_dist_to_hoop(self._get_arm_tip())
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Phase 1: Update based on action (Aiming or Shooting)
        if self.ball_state == "ready":
            if shift_held:
                self.arm_angle = 45.0
                self.arm_power = (self.ARM_MIN_POWER + self.ARM_MAX_POWER) / 2
            
            if movement == 1: # Up
                self.arm_angle = min(self.ARM_MAX_ANGLE, self.arm_angle + self.ARM_ANGLE_SPEED)
            elif movement == 2: # Down
                self.arm_angle = max(self.ARM_MIN_ANGLE, self.arm_angle - self.ARM_ANGLE_SPEED)
            elif movement == 3: # Left
                self.arm_power = max(self.ARM_MIN_POWER, self.arm_power - self.ARM_POWER_SPEED)
            elif movement == 4: # Right
                self.arm_power = min(self.ARM_MAX_POWER, self.arm_power + self.ARM_POWER_SPEED)

            if space_held:
                self._shoot()

        # Phase 2: Update physics if ball is in flight
        elif self.ball_state == "in_flight":
            self.ball_vel[1] += self.GRAVITY
            self.ball_pos += self.ball_vel
            self.ball_trail.append(self.ball_pos.copy())
            if len(self.ball_trail) > 20:
                self.ball_trail.pop(0)
            
            # Continuous reward for getting closer
            dist_to_hoop = self._get_dist_to_hoop(self.ball_pos)
            if dist_to_hoop < self.last_dist_to_hoop:
                reward += 0.1
            self.last_dist_to_hoop = dist_to_hoop

            # Check collisions
            reward += self._handle_collisions()

        # Phase 3: Handle post-shot states
        elif self.ball_state in ["scored", "missed"]:
            self._reset_ball()

        # Update game state
        self.steps += 1
        if self.flash_timer > 0:
            self.flash_timer -= 1
            
        if self.score >= self.WIN_SCORE:
            terminated = True
            self.game_over = True
            reward += 100
        elif self.missed_shots >= self.LOSE_MISSES:
            terminated = True
            self.game_over = True
            reward -= 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _shoot(self):
        # sfx_shoot
        self.ball_state = "in_flight"
        self.ball_pos = self._get_arm_tip()
        
        launch_angle_rad = math.radians(self.arm_angle)
        launch_speed = 3 + 17 * self.arm_power

        self.ball_vel = np.array([
            math.cos(launch_angle_rad) * launch_speed,
            -math.sin(launch_angle_rad) * launch_speed
        ])
        self.ball_trail = [self.ball_pos.copy()]
        self.last_dist_to_hoop = self._get_dist_to_hoop(self.ball_pos)

    def _handle_collisions(self):
        reward = 0
        ball_radius = 10
        
        # Backboard collision
        if self.ball_pos[0] + ball_radius > self.backboard_rect.left and self.ball_vel[0] > 0:
            if self.backboard_rect.top < self.ball_pos[1] < self.backboard_rect.bottom:
                self.ball_pos[0] = self.backboard_rect.left - ball_radius
                self.ball_vel[0] *= -self.BOUNCE_DAMPENING
                reward += 1 # sfx_backboard_hit

        # Rim collisions
        for rim_pos in [self.hoop_rim_left, self.hoop_rim_right]:
            dist = np.linalg.norm(self.ball_pos - np.array(rim_pos))
            if dist < ball_radius:
                # sfx_rim_hit
                normal = (self.ball_pos - np.array(rim_pos)) / dist
                self.ball_vel = self.ball_vel - 2 * np.dot(self.ball_vel, normal) * normal
                self.ball_vel *= self.BOUNCE_DAMPENING

        # Ground collision
        if self.ball_pos[1] + ball_radius > self.SCREEN_HEIGHT - 50:
            self.ball_state = "missed"
            self.missed_shots += 1
            # sfx_miss

        # Off-screen miss
        if self.ball_pos[0] > self.SCREEN_WIDTH or self.ball_pos[0] < 0:
            self.ball_state = "missed"
            self.missed_shots += 1
            # sfx_miss

        # Score check
        is_above = self.ball_pos[1] < self.hoop_pos[1]
        is_moving_down = self.ball_vel[1] > 0
        is_within_hoop_x = self.hoop_rim_left[0] < self.ball_pos[0] < self.hoop_rim_right[0]
        
        if is_moving_down and is_within_hoop_x:
            ball_y_next = self.ball_pos[1] + self.ball_vel[1]
            if is_above and ball_y_next >= self.hoop_pos[1]:
                self.ball_state = "scored"
                self.score += 1
                reward += 5
                self.flash_timer = 15
                # sfx_score
        
        return reward

    def _get_observation(self):
        self._draw_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "missed_shots": self.missed_shots,
            "ball_state": self.ball_state,
        }

    def _get_arm_tip(self):
        angle_rad = math.radians(self.arm_angle)
        length = self.ARM_SEGMENT_LEN * self.arm_power
        tip_x = self.ARM_BASE_POS[0] + self.ARM_SEGMENT_LEN + length * math.cos(angle_rad)
        tip_y = self.ARM_BASE_POS[1] - length * math.sin(angle_rad)
        return np.array([tip_x, tip_y])

    def _get_dist_to_hoop(self, pos):
        hoop_center = ( (self.hoop_rim_left[0] + self.hoop_rim_right[0]) / 2, self.hoop_pos[1] )
        return np.linalg.norm(pos - np.array(hoop_center))

    def _draw_background(self):
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Platform
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM_SHADOW, (0, self.SCREEN_HEIGHT - 45, self.SCREEN_WIDTH, 50))
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, (0, self.SCREEN_HEIGHT - 50, self.SCREEN_WIDTH, 50))
        
        # Backboard and Hoop
        pygame.draw.rect(self.screen, self.COLOR_BACKBOARD_OUTLINE, self.backboard_rect.inflate(4, 4))
        pygame.draw.rect(self.screen, self.COLOR_BACKBOARD, self.backboard_rect)
        pygame.draw.line(self.screen, self.COLOR_HOOP, self.hoop_rim_left, self.hoop_rim_right, 5)
        
        # Hoop flash on score
        if self.flash_timer > 0:
            alpha = 255 * (self.flash_timer / 15)
            flash_surf = pygame.Surface((60, 20), pygame.SRCALPHA)
            flash_surf.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surf, (self.hoop_rim_left[0], self.hoop_rim_left[1] - 10))

        # Arm
        angle_rad = math.radians(self.arm_angle)
        joint_pos = (
            self.ARM_BASE_POS[0] + self.ARM_SEGMENT_LEN * math.cos(0),
            self.ARM_BASE_POS[1] - self.ARM_SEGMENT_LEN * math.sin(0)
        )
        ext_len = self.ARM_SEGMENT_LEN * self.arm_power
        tip_pos = (
            joint_pos[0] + ext_len * math.cos(angle_rad),
            joint_pos[1] - ext_len * math.sin(angle_rad)
        )
        
        pygame.draw.line(self.screen, self.COLOR_ARM_OUTLINE, self.ARM_BASE_POS, joint_pos, 14)
        pygame.draw.line(self.screen, self.COLOR_ARM, self.ARM_BASE_POS, joint_pos, 10)
        pygame.draw.line(self.screen, self.COLOR_ARM_OUTLINE, joint_pos, tip_pos, 14)
        pygame.draw.line(self.screen, self.COLOR_ARM, joint_pos, tip_pos, 10)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ARM_BASE_POS[0]), int(self.ARM_BASE_POS[1]), 10, self.COLOR_ARM_OUTLINE)
        pygame.gfxdraw.filled_circle(self.screen, int(joint_pos[0]), int(joint_pos[1]), 10, self.COLOR_ARM_OUTLINE)

        # Ball Trail
        if self.ball_state == "in_flight":
            for i, pos in enumerate(self.ball_trail):
                alpha = int(200 * (i / len(self.ball_trail)))
                radius = int(10 * (i / len(self.ball_trail)))
                if radius > 1:
                    pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, (*self.COLOR_BALL, alpha))

        # Ball
        ball_radius = 10
        ball_draw_pos = self._get_arm_tip() if self.ball_state == "ready" else self.ball_pos
        pygame.gfxdraw.filled_circle(self.screen, int(ball_draw_pos[0]), int(ball_draw_pos[1]), ball_radius, self.COLOR_BALL_OUTLINE)
        pygame.gfxdraw.filled_circle(self.screen, int(ball_draw_pos[0]), int(ball_draw_pos[1]), ball_radius - 2, self.COLOR_BALL)
        
        # Power/Angle indicator
        if self.ball_state == "ready":
            self._render_indicators(joint_pos)

    def _render_indicators(self, joint_pos):
        # Angle indicator arc
        rect = pygame.Rect(joint_pos[0] - 40, joint_pos[1] - 40, 80, 80)
        pygame.gfxdraw.arc(self.screen, rect.centerx, rect.centery, 40, int(self.ARM_MIN_ANGLE), int(self.ARM_MAX_ANGLE), (*self.COLOR_UI_TEXT, 100))
        
        angle_rad = math.radians(self.arm_angle)
        indicator_end = (joint_pos[0] + 45 * math.cos(angle_rad), joint_pos[1] - 45 * math.sin(angle_rad))
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, joint_pos, indicator_end, 2)

        # Power bar
        bar_width = 100
        bar_height = 10
        bar_x = self.ARM_BASE_POS[0] - 50
        bar_y = self.SCREEN_HEIGHT - 25
        fill_width = bar_width * ((self.arm_power - self.ARM_MIN_POWER) / (self.ARM_MAX_POWER - self.ARM_MIN_POWER))
        
        pygame.draw.rect(self.screen, self.COLOR_UI_SHADOW, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HOOP, (bar_x, bar_y, fill_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        shadow = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_SHADOW)
        self.screen.blit(shadow, (17, 17))
        self.screen.blit(score_text, (15, 15))
        
        # Misses
        miss_text = "MISSES: "
        for i in range(self.LOSE_MISSES):
            color = self.COLOR_X if i < self.missed_shots else self.COLOR_UI_SHADOW
            char = self.font_medium.render("X", True, color)
            self.screen.blit(char, (15 + self.font_medium.size(miss_text)[0] + i * 20, 50))
        
        # Game Over Text
        if self.game_over:
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = (100, 255, 100) if self.score >= self.WIN_SCORE else (255, 100, 100)
            
            text = self.font_large.render(msg, True, color)
            shadow = self.font_large.render(msg, True, self.COLOR_UI_SHADOW)
            
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            shadow_rect = shadow.get_rect(center=(self.SCREEN_WIDTH / 2 + 3, self.SCREEN_HEIGHT / 2 + 3))
            
            self.screen.blit(shadow, shadow_rect)
            self.screen.blit(text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("RoboBall")
    clock = pygame.time.Clock()
    
    # --- Game loop for human play ---
    while not done:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # Only step if action is taken OR ball is in flight
        if any(action) or env.ball_state == "in_flight":
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Game Over. Score: {info['score']}, Steps: {info['steps']}")
                # Wait for a moment before auto-resetting
                pygame.time.wait(2000)
                obs, info = env.reset()
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()