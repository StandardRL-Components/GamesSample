
# Generated: 2025-08-28T00:17:19.561790
# Source Brief: brief_03742.md
# Brief Index: 3742

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to aim, ↑↓ to adjust power. Space to swing. Shift to aim at the hole."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A procedural mini-golf game. Sink the ball in the fewest strokes possible while navigating obstacles."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STROKES = 10
    MAX_EPISODE_STEPS = 1000
    
    # Colors
    COLOR_GRASS_DARK = (25, 80, 50)
    COLOR_GRASS_LIGHT = (35, 110, 70)
    COLOR_BALL = (255, 255, 255)
    COLOR_SHADOW = (0, 0, 0, 50)
    COLOR_HOLE_FLAG = (255, 50, 50)
    COLOR_WATER = (60, 120, 180)
    COLOR_WATER_BORDER = (90, 150, 210)
    COLOR_OBSTACLE = (100, 100, 110)
    COLOR_OBSTACLE_SHADOW = (40, 40, 45)
    COLOR_UI_TEXT = (240, 240, 240)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("sans-serif", 24)
        self.font_small = pygame.font.SysFont("sans-serif", 14)
        
        # Game state variables
        self.ball_pos = np.array([0.0, 0.0])
        self.ball_vel = np.array([0.0, 0.0])
        self.hole_pos = np.array([0.0, 0.0])
        self.obstacles = []
        self.water = []
        self.stroke_count = 0
        self.score = 0.0
        self.steps = 0
        self.game_over = False
        self.difficulty = 0
        self.aim_angle = 0.0
        self.aim_power = 0.5
        self.is_ball_moving = False
        self.pre_swing_ball_pos = np.array([0.0, 0.0])
        
        # Initialize state variables
        # self.reset() is called by the agent/runner, but we call it here for initialization
        # The seed will be None, so a random seed will be used by super().reset()
        self._seed_and_reset()

        # Validate implementation
        self.validate_implementation()
    
    def _seed_and_reset(self, seed=None):
        """Helper to call reset with a seed, used in both __init__ and reset."""
        super().reset(seed=seed)
        if self.game_over and self.stroke_count >= self.MAX_STROKES:
            self.difficulty = 0
        self.steps = 0
        self.score = 0
        self.stroke_count = 0
        self.game_over = False
        self.is_ball_moving = False
        self.ball_vel = np.array([0.0, 0.0])
        self._generate_hole()
        self._reset_aim()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._seed_and_reset(seed)
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        terminated = False
        
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1

        if self.is_ball_moving:
            # This state should not be entered as the simulation runs to completion
            # within the swing step. We return a small penalty.
            return self._get_observation(), -0.1, self.game_over, False, self._get_info()

        # --- AIMING PHASE ---
        reward = -0.01  # Small cost for thinking/aiming
        
        if movement == 1: self.aim_power = min(1.0, self.aim_power + 0.05)
        elif movement == 2: self.aim_power = max(0.1, self.aim_power - 0.05)
        elif movement == 3: self.aim_angle -= math.pi / 32 
        elif movement == 4: self.aim_angle += math.pi / 32
        
        if shift_press: self._reset_aim()
            
        if space_press:
            # sound: "swing.wav"
            swing_reward, swing_terminated = self._execute_swing()
            reward += swing_reward
            terminated = swing_terminated
        
        if not terminated and self.stroke_count >= self.MAX_STROKES:
            self.game_over = True
            terminated = True
            reward -= 50  # Heavy penalty for failing the hole
        
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            terminated = True
        
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _execute_swing(self):
        self.stroke_count += 1
        self.is_ball_moving = True
        self.pre_swing_ball_pos = self.ball_pos.copy()
        
        max_speed = 15.0
        self.ball_vel = np.array([
            math.cos(self.aim_angle) * self.aim_power * max_speed,
            math.sin(self.aim_angle) * self.aim_power * max_speed
        ])
        
        swing_reward = 0.0
        swing_terminated = False
        dist_before = np.linalg.norm(self.ball_pos - self.hole_pos)
        
        sim_steps = 0
        while self.is_ball_moving and sim_steps < 300:
            sim_steps += 1
            self.ball_pos += self.ball_vel
            self.ball_vel *= 0.985
            
            # Boundary collisions
            if not (5 < self.ball_pos[0] < self.WIDTH - 5):
                self.ball_pos[0] = np.clip(self.ball_pos[0], 5, self.WIDTH - 5)
                self.ball_vel[0] *= -0.7; swing_reward -= 1.0
            if not (5 < self.ball_pos[1] < self.HEIGHT - 5):
                self.ball_pos[1] = np.clip(self.ball_pos[1], 5, self.HEIGHT - 5)
                self.ball_vel[1] *= -0.7; swing_reward -= 1.0
            
            # Obstacle collisions
            ball_rect = pygame.Rect(self.ball_pos[0]-4, self.ball_pos[1]-4, 8, 8)
            for obs in self.obstacles:
                if obs.colliderect(ball_rect):
                    swing_reward -= 1.0; # sound: "thump.wav"
                    self.ball_vel[0] *= -0.7 if abs(self.ball_vel[0]) > abs(self.ball_vel[1]) else 1
                    self.ball_vel[1] *= -0.7 if abs(self.ball_vel[1]) >= abs(self.ball_vel[0]) else 1
                    self.ball_pos += self.ball_vel * 1.5
                    break

            # Water hazard check
            for w in self.water:
                if np.linalg.norm(self.ball_pos - w['pos']) < w['radius']:
                    swing_reward -= 2.0; self.stroke_count += 1
                    self.ball_pos = self.pre_swing_ball_pos.copy()
                    self.is_ball_moving = False; # sound: "splash.wav"
                    break
            if not self.is_ball_moving: break

            # Hole check
            if np.linalg.norm(self.ball_pos - self.hole_pos) < 7:
                swing_reward += 100 + max(0, (5 - self.stroke_count) * 10)
                self.game_over = True; swing_terminated = True
                self.is_ball_moving = False; self.difficulty += 1
                self.ball_pos = self.hole_pos.copy(); # sound: "sink.wav"
                break
            
            if np.linalg.norm(self.ball_vel) < 0.1: self.is_ball_moving = False
        
        if not swing_terminated and self.is_ball_moving == False:
            dist_after = np.linalg.norm(self.ball_pos - self.hole_pos)
            swing_reward += (dist_before - dist_after) * 0.2
        
        self.is_ball_moving = False
        return swing_reward, swing_terminated

    def _generate_hole(self):
        course_rect = pygame.Rect(40, 40, self.WIDTH - 80, self.HEIGHT - 80)
        self.hole_pos = np.array([
            self.np_random.uniform(course_rect.centerx, course_rect.right),
            self.np_random.uniform(course_rect.top, course_rect.bottom)
        ])
        self.ball_pos = np.array([
            self.np_random.uniform(course_rect.left, course_rect.centerx),
            self.np_random.uniform(course_rect.top, course_rect.bottom)
        ])
        
        self.obstacles, self.water = [], []
        for _ in range(min(5, self.difficulty)):
            for _ in range(10): # Max 10 attempts to place
                w = self.np_random.uniform(15, 50); h = self.np_random.uniform(15, 50)
                x = self.np_random.uniform(course_rect.left, course_rect.right - w)
                y = self.np_random.uniform(course_rect.top, course_rect.bottom - h)
                rect = pygame.Rect(x, y, w, h)
                if np.linalg.norm(self.hole_pos - rect.center) > 30 and np.linalg.norm(self.ball_pos - rect.center) > 30:
                    self.obstacles.append(rect); break
        
        for _ in range(self.np_random.integers(0, self.difficulty // 2 + 1)):
            for _ in range(10): # Max 10 attempts
                radius = self.np_random.uniform(20, 40)
                pos = self.np_random.uniform(low=[course_rect.left, course_rect.top], high=[course_rect.right, course_rect.bottom])
                if np.linalg.norm(pos - self.hole_pos) > radius + 20 and np.linalg.norm(pos - self.ball_pos) > radius + 20:
                    self.water.append({'pos': pos, 'radius': radius}); break

    def _reset_aim(self):
        delta = self.hole_pos - self.ball_pos
        self.aim_angle = math.atan2(delta[1], delta[0])
        self.aim_power = np.clip(np.linalg.norm(delta) / (self.WIDTH/2), 0.1, 1.0)

    def _get_observation(self):
        self.screen.fill(self.COLOR_GRASS_DARK)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_GRASS_LIGHT, (10, 10, self.WIDTH - 20, self.HEIGHT - 20), border_radius=10)
        
        renderables = []
        for w in self.water: renderables.append(('water', w, w['pos'][1] + w['radius']))
        for obs in self.obstacles: renderables.append(('obstacle', obs, obs.bottom))
        renderables.append(('hole', self.hole_pos, self.hole_pos[1]))
        renderables.append(('ball', self.ball_pos, self.ball_pos[1]))
        renderables.sort(key=lambda item: item[2])
        
        for r_type, r_obj, _ in renderables:
            if r_type == 'water':
                pos = (int(r_obj['pos'][0]), int(r_obj['pos'][1])); radius = int(r_obj['radius'])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_WATER)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_WATER_BORDER)
            elif r_type == 'obstacle':
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_SHADOW, r_obj.move(4, 4))
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, r_obj)
            elif r_type == 'hole':
                pos = (int(r_obj[0]), int(r_obj[1]))
                pygame.draw.line(self.screen, (150, 150, 150), (pos[0], pos[1]), (pos[0], pos[1] - 30), 2)
                pygame.draw.polygon(self.screen, self.COLOR_HOLE_FLAG, [(pos[0]+1, pos[1] - 30), (pos[0]+1, pos[1] - 20), (pos[0] + 12, pos[1] - 25)])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, (0, 0, 0))
            elif r_type == 'ball':
                pos = (int(r_obj[0]), int(r_obj[1]))
                shadow_surf = pygame.Surface((12, 12), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(shadow_surf, 6, 6, 6, self.COLOR_SHADOW)
                self.screen.blit(shadow_surf, (pos[0]-6, pos[1]-3))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.COLOR_BALL)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 5, (200, 200, 200))
        
        if not self.is_ball_moving:
            line_len = self.aim_power * 100
            end_pos = (self.ball_pos[0] + math.cos(self.aim_angle) * line_len, self.ball_pos[1] + math.sin(self.aim_angle) * line_len)
            pygame.draw.aaline(self.screen, (255, 255, 255, 150), self.ball_pos, end_pos, 2)
            pygame.gfxdraw.filled_circle(self.screen, int(end_pos[0]), int(end_pos[1]), 3, (255, 255, 255))

    def _render_ui(self):
        stroke_text = self.font_main.render(f"Stroke: {self.stroke_count}/{self.MAX_STROKES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stroke_text, (20, 20))
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 50))
        
        power_bar_bg = pygame.Rect(self.WIDTH - 170, 25, 150, 20)
        power_bar_fill = pygame.Rect(self.WIDTH - 170, 25, 150 * self.aim_power, 20)
        c_start, c_end = (255, 255, 0), (255, 0, 0)
        power_color = [int(s + (e - s) * self.aim_power) for s, e in zip(c_start, c_end)]
        pygame.draw.rect(self.screen, (50, 50, 50), power_bar_bg, border_radius=4)
        pygame.draw.rect(self.screen, power_color, power_bar_fill, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, power_bar_bg, 2, border_radius=4)
        power_text = self.font_small.render("POWER", True, self.COLOR_UI_TEXT)
        self.screen.blit(power_text, (power_bar_bg.x - 55, power_bar_bg.y + 3))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            end_str = "Hole Complete!" if self.stroke_count < self.MAX_STROKES else "Max Strokes Reached"
            end_text = self.font_main.render(end_str, True, (255, 255, 255))
            self.screen.blit(end_text, end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "strokes": self.stroke_count, "difficulty": self.difficulty}

    def close(self):
        pygame.font.quit()
        pygame.quit()
        super().close()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")