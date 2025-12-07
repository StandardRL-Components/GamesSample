
# Generated: 2025-08-27T23:00:53.537925
# Source Brief: brief_03316.md
# Brief Index: 3316

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class MovingPlatform:
    """A helper class to manage the state and behavior of a moving platform."""
    def __init__(self, x, y, w, h, move_type, move_range, speed):
        self.rect = pygame.Rect(x, y, w, h)
        self.start_pos = pygame.Vector2(x, y)
        self.move_type = move_type
        self.move_range = move_range
        
        if self.move_type == 'vertical':
            self.vel = pygame.Vector2(0, speed)
        else: # horizontal
            self.vel = pygame.Vector2(speed, 0)

    def update(self):
        """Updates the platform's position and reverses direction if at an endpoint."""
        self.rect.move_ip(self.vel)
        if self.move_type == 'vertical':
            if self.vel.y > 0 and self.rect.y >= self.start_pos.y + self.move_range:
                self.rect.y = self.start_pos.y + self.move_range
                self.vel.y *= -1
            elif self.vel.y < 0 and self.rect.y <= self.start_pos.y:
                self.rect.y = self.start_pos.y
                self.vel.y *= -1
        else: # horizontal
            if self.vel.x > 0 and self.rect.x >= self.start_pos.x + self.move_range:
                self.rect.x = self.start_pos.x + self.move_range
                self.vel.x *= -1
            elif self.vel.x < 0 and self.rect.x <= self.start_pos.x:
                self.rect.x = self.start_pos.x
                self.vel.x *= -1


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move. Press Space to jump. Press Shift to dash."
    )

    game_description = (
        "A fast-paced pixel art platformer. Race against the clock to reach the flag across three increasingly difficult stages."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and timing
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TIME_PER_STAGE = 60.0

        # Colors
        self.COLOR_BG = (20, 30, 50)
        self.COLOR_PLATFORM = (60, 180, 70)
        self.COLOR_PLAYER = (255, 150, 0)
        self.COLOR_FLAG_RED = (220, 50, 50)
        self.COLOR_FLAG_WHITE = (250, 250, 250)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_SCORE = (255, 220, 0)

        # Physics and Player constants
        self.GRAVITY = 0.5
        self.PLAYER_ACCEL = 0.8
        self.PLAYER_FRICTION = -0.15
        self.JUMP_STRENGTH = -10
        self.MAX_SPEED_X = 6
        self.DASH_SPEED = 12
        self.DASH_DURATION = 5
        self.PLAYER_SIZE = (20, 20)
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Stage definitions
        self._define_stages()
        
        # Initialize state variables
        self.reset()

    def _define_stages(self):
        self.stage_data = {
            1: {
                "player_start": (50, 300),
                "flag_pos": (580, 50),
                "platforms": [
                    # (x, y, w, h, type, move_range, speed)
                    (0, 350, 150, 50, 'static', 0, 0),
                    (200, 300, 100, 20, 'static', 0, 0),
                    (350, 250, 100, 20, 'static', 0, 0),
                    (250, 180, 80, 20, 'static', 0, 0),
                    (450, 150, 150, 50, 'static', 0, 0),
                ]
            },
            2: {
                "player_start": (50, 300),
                "flag_pos": (580, 50),
                "platforms": [
                    (0, 350, 100, 50, 'static', 0, 0),
                    (150, 300, 80, 20, 'vertical', 100, 1), # Moving
                    (300, 250, 80, 20, 'static', 0, 0),
                    (450, 200, 80, 20, 'vertical', 120, 1), # Moving
                    (550, 100, 80, 50, 'static', 0, 0),
                ]
            },
            3: {
                "player_start": (30, 100),
                "flag_pos": (580, 350),
                "platforms": [
                    (0, 150, 80, 20, 'static', 0, 0),
                    (150, 180, 60, 20, 'horizontal', 150, 2), # Moving
                    (400, 220, 60, 20, 'horizontal', -150, 2), # Moving
                    (200, 280, 60, 20, 'vertical', 80, 1), # Moving
                    (50, 350, 60, 20, 'horizontal', 200, 2), # Moving
                    (550, 380, 80, 20, 'static', 0, 0),
                ]
            }
        }
    
    def _generate_stage(self, stage_num):
        data = self.stage_data.get(stage_num, self.stage_data[1])
        
        self.player_pos = pygame.Vector2(data["player_start"])
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.dash_timer = 0
        self.last_move_dir = 1
        self.dash_trail.clear()
        
        fx, fy = data["flag_pos"]
        self.flag_rect = pygame.Rect(fx, fy, 30, 50)
        
        self.platforms = []
        self.moving_platforms = []
        for p_data in data["platforms"]:
            x, y, w, h, p_type, move_range, speed = p_data
            if p_type == 'static':
                self.platforms.append(pygame.Rect(x, y, w, h))
            else:
                self.moving_platforms.append(MovingPlatform(x, y, w, h, p_type, move_range, speed))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.stage = 1
        self.dash_trail = deque(maxlen=8)
        
        self._generate_stage(1)
        
        self.time_remaining = self.TIME_PER_STAGE
        self.space_was_held = True # Prevent jump on first frame
        self.shift_was_held = True # Prevent dash on first frame
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        self.clock.tick(self.FPS)
        self.steps += 1
        
        reward = self._update_game_state(action)
        terminated = self.game_over
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_game_state(self, action):
        self.time_remaining -= 1.0 / self.FPS
        reward = 0.0

        self._handle_input(action)
        self._update_physics()
        
        # Rewards and game events
        if self.on_ground:
            reward += 0.01 # Scaled from 0.1 to be reasonable for 30fps
        else:
            reward -= 0.02 # Scaled from -0.2

        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, *self.PLAYER_SIZE)
        
        # Check for falling off
        if player_rect.top > self.HEIGHT:
            self.game_over = True
            reward -= 10
            # Sound: Fall/death sfx
            return reward

        # Check for time out
        if self.time_remaining <= 0:
            self.game_over = True
            reward -= 10 # Penalty for timeout
            return reward

        # Check for reaching the flag
        if player_rect.colliderect(self.flag_rect):
            reward += 5
            self.score += max(0, self.time_remaining) * 0.2
            self.stage += 1
            if self.stage > 3:
                self.game_over = True
                reward += 50 # Big win bonus
                # Sound: Win fanfare
            else:
                self._generate_stage(self.stage)
                self.time_remaining = self.TIME_PER_STAGE
                # Sound: Stage complete sfx
        
        return reward

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Horizontal movement
        if movement == 3: # Left
            self.player_vel.x -= self.PLAYER_ACCEL
            self.last_move_dir = -1
        elif movement == 4: # Right
            self.player_vel.x += self.PLAYER_ACCEL
            self.last_move_dir = 1
        
        # Jump
        if space_held and not self.space_was_held and self.on_ground:
            self.player_vel.y = self.JUMP_STRENGTH
            self.on_ground = False
            # Sound: Jump sfx

        # Dash
        if shift_held and not self.shift_was_held and self.dash_timer <= 0:
            self.dash_timer = self.DASH_DURATION
            self.player_vel.y = max(self.player_vel.y, 0) * 0.2 # Dampen vertical speed slightly
            # Sound: Dash sfx
        
        self.space_was_held = space_held
        self.shift_was_held = shift_held

    def _update_physics(self):
        # Update moving platforms first
        for p in self.moving_platforms:
            p.update()
            
        # Apply friction
        if self.player_vel.x != 0:
            self.player_vel.x += self.PLAYER_FRICTION * np.sign(self.player_vel.x)
            if abs(self.player_vel.x) < 0.5: self.player_vel.x = 0

        # Apply dash
        if self.dash_timer > 0:
            self.player_vel.x = self.DASH_SPEED * self.last_move_dir
            self.dash_timer -= 1
            self.dash_trail.append(pygame.Rect(self.player_pos.x, self.player_pos.y, *self.PLAYER_SIZE))
        
        # Apply gravity
        self.player_vel.y += self.GRAVITY

        # Clamp velocity
        self.player_vel.x = max(-self.MAX_SPEED_X, min(self.MAX_SPEED_X, self.player_vel.x))
        
        # Collision detection
        self.player_pos.x += self.player_vel.x
        self._handle_collisions('horizontal')
        
        self.player_pos.y += self.player_vel.y
        self.on_ground = False
        self._handle_collisions('vertical')
        
        # Keep player on screen horizontally
        self.player_pos.x = max(0, min(self.WIDTH - self.PLAYER_SIZE[0], self.player_pos.x))

    def _handle_collisions(self, direction):
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, *self.PLAYER_SIZE)
        all_platform_rects = self.platforms + [p.rect for p in self.moving_platforms]

        for plat_rect in all_platform_rects:
            if player_rect.colliderect(plat_rect):
                if direction == 'horizontal':
                    if self.player_vel.x > 0: player_rect.right = plat_rect.left
                    elif self.player_vel.x < 0: player_rect.left = plat_rect.right
                    self.player_pos.x = player_rect.x
                    self.player_vel.x = 0
                elif direction == 'vertical':
                    if self.player_vel.y > 0:
                        player_rect.bottom = plat_rect.top
                        self.on_ground = True
                        self.player_vel.y = 0
                    elif self.player_vel.y < 0:
                        player_rect.top = plat_rect.bottom
                        self.player_vel.y = 0 # Bonk head
                    self.player_pos.y = player_rect.y

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Platforms
        for plat_rect in self.platforms: pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat_rect)
        for p in self.moving_platforms: pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p.rect)
        
        # Flag
        pygame.draw.rect(self.screen, self.COLOR_FLAG_RED, self.flag_rect)
        pygame.draw.rect(self.screen, self.COLOR_FLAG_WHITE, (self.flag_rect.x, self.flag_rect.y, 15, 25))
        pygame.draw.rect(self.screen, self.COLOR_FLAG_WHITE, (self.flag_rect.x + 15, self.flag_rect.y + 25, 15, 25))

        # Dash Trail
        if self.dash_timer > 0 or len(self.dash_trail) > 0:
            if self.dash_timer <= 0: self.dash_trail.popleft()
            for i, rect in enumerate(self.dash_trail):
                alpha = int(150 * (i + 1) / len(self.dash_trail))
                trail_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                trail_surf.fill((*self.COLOR_PLAYER, alpha))
                self.screen.blit(trail_surf, rect.topleft)

        # Player
        player_rect = pygame.Rect(int(self.player_pos.x), int(self.player_pos.y), *self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), player_rect, 2) # Outline

    def _render_ui(self):
        # Time
        time_surf = self.font_small.render(f"TIME: {max(0, int(self.time_remaining))}", True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))
        
        # Score
        score_surf = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_SCORE)
        self.screen.blit(score_surf, score_surf.get_rect(topright=(self.WIDTH - 10, 10)))

        # Stage
        stage_surf = self.font_small.render(f"STAGE: {self.stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_surf, stage_surf.get_rect(midtop=(self.WIDTH // 2, 10)))

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.stage > 3 else "GAME OVER"
            msg_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            bg_rect = msg_rect.inflate(20, 20)
            pygame.draw.rect(self.screen, self.COLOR_BG, bg_rect)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, bg_rect, 2)
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stage": self.stage}
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")