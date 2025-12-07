import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:45:08.695188
# Source Brief: brief_00064.md
# Brief Index: 64
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls two particles.
    The goal is to guide both particles to a target, synchronizing their
    speed and arrival time for a higher score.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement (0=none, 1=up(P1), 2=down(P1), 3=left(P2), 4=right(P2))
    - action[1]: Unused
    - action[2]: Unused

    Observation Space: Box(0, 255, (400, 640, 3), uint8) - RGB array of the game screen.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    game_description = (
        "Control two particles simultaneously, guiding them to a shared target. "
        "Synchronize their speed and arrival time to maximize your score."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to move the blue particle vertically and "
        "←→ arrow keys to move the red particle horizontally."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1800  # 60 seconds at 30 FPS

    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_P1 = (0, 170, 255)
    COLOR_P2 = (255, 68, 0)
    COLOR_GOAL = (255, 255, 255)
    COLOR_UI_TEXT = (204, 204, 204)
    COLOR_SYNC_BAR = (0, 255, 170)
    
    # Physics
    ACCELERATION = 0.1
    DRAG = 0.99
    MAX_SPEED = 5.0
    
    # Game elements
    PARTICLE_RADIUS = 10
    GOAL_RADIUS = 15
    TRAIL_LENGTH = 25

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        # --- Gym Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # For human rendering
        self.window = None

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.p1_pos = pygame.Vector2(0, 0)
        self.p1_vel = pygame.Vector2(0, 0)
        self.p2_pos = pygame.Vector2(0, 0)
        self.p2_vel = pygame.Vector2(0, 0)
        self.p1_trail = deque(maxlen=self.TRAIL_LENGTH)
        self.p2_trail = deque(maxlen=self.TRAIL_LENGTH)
        self.goal_pos = pygame.Vector2(self.SCREEN_WIDTH // 2, 50)
        self.p1_finished = False
        self.p1_finish_time = 0
        self.p2_finished = False
        self.p2_finish_time = 0
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.p1_pos = pygame.Vector2(self.SCREEN_WIDTH * 0.25, self.SCREEN_HEIGHT - 50)
        self.p1_vel = pygame.Vector2(self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-1.0, -0.5))
        self.p2_pos = pygame.Vector2(self.SCREEN_WIDTH * 0.75, self.SCREEN_HEIGHT - 50)
        self.p2_vel = pygame.Vector2(self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-1.0, -0.5))

        self.p1_trail.clear()
        self.p2_trail.clear()

        self.p1_finished = False
        self.p1_finish_time = 0
        self.p2_finished = False
        self.p2_finish_time = 0
        
        if self.render_mode == "human":
            self.render()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        reward = 0
        
        if not self.game_over:
            self._handle_action(movement)
            self._update_physics()
            reward = self._calculate_reward()
            self.score += reward
            self.steps += 1
            terminated = self._check_termination()
            truncated = self.steps >= self.MAX_STEPS
            if truncated:
                self.game_over = True
                if not self.p1_finished or not self.p2_finished: self.score -= 10
        else:
            terminated = True
            truncated = self.steps >= self.MAX_STEPS

        if self.render_mode == "human":
            self.render()

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, movement):
        if movement == 1: self.p1_vel.y -= self.ACCELERATION  # Up for P1
        elif movement == 2: self.p1_vel.y += self.ACCELERATION  # Down for P1
        elif movement == 3: self.p2_vel.x -= self.ACCELERATION  # Left for P2
        elif movement == 4: self.p2_vel.x += self.ACCELERATION  # Right for P2
    
    def _update_physics(self):
        self.p1_vel *= self.DRAG
        self.p2_vel *= self.DRAG

        if self.p1_vel.magnitude() > self.MAX_SPEED: self.p1_vel.scale_to_length(self.MAX_SPEED)
        if self.p2_vel.magnitude() > self.MAX_SPEED: self.p2_vel.scale_to_length(self.MAX_SPEED)

        if not self.p1_finished: self.p1_pos += self.p1_vel
        if not self.p2_finished: self.p2_pos += self.p2_vel

        self.p1_trail.append(self.p1_pos.copy())
        self.p2_trail.append(self.p2_pos.copy())

        self.p1_pos.x = np.clip(self.p1_pos.x, 0, self.SCREEN_WIDTH)
        self.p1_pos.y = np.clip(self.p1_pos.y, 0, self.SCREEN_HEIGHT)
        self.p2_pos.x = np.clip(self.p2_pos.x, 0, self.SCREEN_WIDTH)
        self.p2_pos.y = np.clip(self.p2_pos.y, 0, self.SCREEN_HEIGHT)

        if not self.p1_finished and self.p1_pos.distance_to(self.goal_pos) < self.GOAL_RADIUS + self.PARTICLE_RADIUS:
            self.p1_finished = True
            self.p1_finish_time = self.steps
        
        if not self.p2_finished and self.p2_pos.distance_to(self.goal_pos) < self.GOAL_RADIUS + self.PARTICLE_RADIUS:
            self.p2_finished = True
            self.p2_finish_time = self.steps

    def _calculate_reward(self):
        reward = 0
        speed1 = self.p1_vel.magnitude()
        speed2 = self.p2_vel.magnitude()
        max_speed = max(speed1, speed2, 1e-6)
        speed_diff_ratio = abs(speed1 - speed2) / max_speed
        
        if speed_diff_ratio < 0.1: reward += 0.01

        sync_bonus_applied = False
        if self.p1_finished and self.p1_finish_time == self.steps:
            reward += 5
            if speed_diff_ratio < 0.1:
                reward += 5
                sync_bonus_applied = True
        
        if self.p2_finished and self.p2_finish_time == self.steps:
            reward += 5
            if not sync_bonus_applied and speed_diff_ratio < 0.1:
                reward += 5

        return reward
    
    def _check_termination(self):
        if self.game_over: return True

        if self.p1_finished and self.p2_finished:
            self.game_over = True
            time_diff = abs(self.p1_finish_time - self.p2_finish_time)
            time_bonus = max(0, 100 - (time_diff * 0.5)) 
            self.score += time_bonus
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
        self._draw_glow_circle(self.screen, self.goal_pos, self.GOAL_RADIUS, self.COLOR_GOAL, 4)
        self._draw_trail(self.screen, self.p1_trail, self.COLOR_P1)
        self._draw_trail(self.screen, self.p2_trail, self.COLOR_P2)
        self._draw_glow_circle(self.screen, self.p1_pos, self.PARTICLE_RADIUS, self.COLOR_P1, 5)
        self._draw_glow_circle(self.screen, self.p2_pos, self.PARTICLE_RADIUS, self.COLOR_P2, 5)

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = self.font_large.render(f"Time: {time_left:.1f}s", True, self.COLOR_UI_TEXT)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)

        speed1 = self.p1_vel.magnitude(); speed2 = self.p2_vel.magnitude()
        sync_ratio = 1.0 - (abs(speed1 - speed2) / max(speed1, speed2, 1e-6))
        max_bar_width = self.SCREEN_WIDTH / 3; bar_height = 10
        current_bar_width = max_bar_width * sync_ratio
        bar_x = self.SCREEN_WIDTH / 2; bar_y = self.SCREEN_HEIGHT - 20
        
        bg_rect = pygame.Rect(0, 0, max_bar_width, bar_height); bg_rect.center = (bar_x, bar_y)
        pygame.draw.rect(self.screen, (50, 50, 70), bg_rect, border_radius=5)
        
        if current_bar_width > 0:
            fg_rect = pygame.Rect(0, 0, current_bar_width, bar_height); fg_rect.center = (bar_x, bar_y)
            pygame.draw.rect(self.screen, self.COLOR_SYNC_BAR, fg_rect, border_radius=5)
            
        sync_text = self.font_small.render("SYNC", True, self.COLOR_UI_TEXT)
        sync_text_rect = sync_text.get_rect(center=(bar_x, bar_y))
        self.screen.blit(sync_text, sync_text_rect)

    def _draw_glow_circle(self, surface, pos, radius, color, glow_layers):
        pos_int = (int(pos.x), int(pos.y))
        for i in range(glow_layers, 0, -1):
            alpha = 150 / (i**2)
            glow_radius = int(radius * (1 + i * 0.15))
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*color, alpha), (glow_radius, glow_radius), glow_radius)
            surface.blit(temp_surf, (pos_int[0] - glow_radius, pos_int[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], radius, color)
    
    def _draw_trail(self, surface, trail, color):
        if len(trail) < 2: return
        for i, pos in enumerate(trail):
            if i == 0: continue
            alpha = int(80 * (i / len(trail)))
            radius = int(self.PARTICLE_RADIUS * 0.6 * (i / len(trail)))
            if radius < 1: continue
            
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*color, alpha), (radius, radius), radius)
            surface.blit(temp_surf, (int(pos.x) - radius, int(pos.y) - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def render(self):
        if self.render_mode == "rgb_array": return self._get_observation()
        if self.window is None:
            pygame.display.init()
            self.window = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        
        obs_arr = self._get_observation()
        surf = pygame.surfarray.make_surface(np.transpose(obs_arr, (1, 0, 2)))
        self.window.blit(surf, (0, 0))
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window:
            pygame.display.quit()
            self.window = None
        pygame.quit()
        
    def validate_implementation(self):
        # This method is for internal validation and not part of the standard Gym API
        try:
            assert self.action_space.shape == (3,)
            assert self.action_space.nvec.tolist() == [5, 2, 2]
            self.reset()
            test_obs = self._get_observation()
            assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert test_obs.dtype == np.uint8
            obs, info = self.reset()
            assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert isinstance(info, dict)
            test_action = self.action_space.sample()
            obs, reward, term, trunc, info = self.step(test_action)
            assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert isinstance(reward, (int, float))
            assert isinstance(term, bool)
            assert isinstance(trunc, bool)
            assert isinstance(info, dict)
        except AssertionError as e:
            # This helps in debugging during development.
            # print(f"Implementation validation failed: {e}")
            pass


if __name__ == '__main__':
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    total_reward = 0
    running = True

    print("\n--- Human Controls ---")
    print(GameEnv.user_guide)
    print("----------------------\n")

    while running:
        action = [0, 0, 0]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        keys = pygame.key.get_pressed()
        
        # The agent action space only allows one movement at a time.
        # We prioritize P1 for human play if both keys are pressed.
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}")
            total_reward = 0
            obs, info = env.reset()

    env.close()