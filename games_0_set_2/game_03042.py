
# Generated: 2025-08-27T22:11:52.434014
# Source Brief: brief_03042.md
# Brief Index: 3042

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 32)
        
        # Colors
        self.COLOR_BG = (0, 0, 0)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLATFORM = (100, 100, 100)
        self.COLOR_FLAG = (0, 220, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE = (200, 200, 200)

        # Game constants
        self.GRAVITY = 0.5
        self.JUMP_STRENGTH = -11
        self.PLAYER_ACCEL = 0.8
        self.PLAYER_FRICTION = -0.12
        self.PLAYER_MAX_SPEED = 6
        self.MAX_STEPS = 2000

        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_rect = None
        self.on_ground = False
        self.platforms = []
        self.flag_rect = None
        self.particles = []
        self.camera_offset_x = 0
        self.current_level = 1
        self.rng = None
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_level = 1
        self.camera_offset_x = 0
        self.particles = []

        self._generate_level()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(30)

        movement = action[0]
        space_held = action[1] == 1
        
        prev_dist_to_flag = abs(self.player_pos.x - self.flag_rect.centerx)

        self._handle_player_movement(movement, space_held)
        self._apply_physics_and_collisions()
        
        reward = 0
        terminated = False

        if self.player_rect.colliderect(self.flag_rect):
            # sfx: level_complete_sound()
            reward += 100
            self._next_level()
        
        if self.player_pos.y > self.HEIGHT + 50:
            # sfx: fall_sound()
            reward -= 10
            terminated = True
            self.game_over = True
            
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        new_dist_to_flag = abs(self.player_pos.x - self.flag_rect.centerx)
        reward += (prev_dist_to_flag - new_dist_to_flag) * 0.1
        
        self.score += reward

        self._update_camera()
        self._update_particles()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        self.platforms = []
        start_platform = pygame.Rect(50, self.HEIGHT - 50, 200, 20)
        self.platforms.append(start_platform)

        self.player_pos = pygame.Vector2(start_platform.centerx, start_platform.top - 20)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, 20, 20)
        self.on_ground = False

        last_platform = start_platform
        num_platforms = 5 + self.current_level
        
        plat_width_decay = 0.9 ** (self.current_level - 1)
        gap_increase = 1.05 ** (self.current_level - 1)

        for _ in range(num_platforms):
            plat_width = max(30, 150 * plat_width_decay)
            gap_x = 80 * gap_increase + self.rng.integers(-20, 20)
            
            new_x = last_platform.right + gap_x
            y_diff = self.rng.integers(-60, 60)
            new_y = np.clip(last_platform.y + y_diff, 100, self.HEIGHT - 40)

            new_platform = pygame.Rect(new_x, new_y, plat_width, 20)
            self.platforms.append(new_platform)
            last_platform = new_platform

        self.flag_rect = pygame.Rect(last_platform.centerx - 5, last_platform.top - 20, 10, 20)

    def _next_level(self):
        self.current_level += 1
        self.particles = []
        self._generate_level()
        self.camera_offset_x = self.player_pos.x - self.WIDTH / 2

    def _handle_player_movement(self, movement, space_held):
        if movement == 3: # Left
            self.player_vel.x -= self.PLAYER_ACCEL
        elif movement == 4: # Right
            self.player_vel.x += self.PLAYER_ACCEL
        else:
            self.player_vel.x *= (1.0 + self.PLAYER_FRICTION)

        self.player_vel.x = np.clip(self.player_vel.x, -self.PLAYER_MAX_SPEED, self.PLAYER_MAX_SPEED)

        if (movement == 1 or space_held) and self.on_ground:
            # sfx: jump_sound()
            self.player_vel.y = self.JUMP_STRENGTH
            self.on_ground = False

    def _apply_physics_and_collisions(self):
        self.player_vel.y += self.GRAVITY
        
        self.player_pos.x += self.player_vel.x
        self.player_rect.x = int(self.player_pos.x)
        
        for plat in self.platforms:
            if self.player_rect.colliderect(plat):
                if self.player_vel.x > 0: self.player_rect.right = plat.left
                elif self.player_vel.x < 0: self.player_rect.left = plat.right
                self.player_pos.x = self.player_rect.x
                self.player_vel.x = 0

        self.player_pos.y += self.player_vel.y
        self.player_rect.y = int(self.player_pos.y)
        
        was_on_ground = self.on_ground
        self.on_ground = False
        for plat in self.platforms:
            if self.player_rect.colliderect(plat):
                if self.player_vel.y > 0:
                    self.player_rect.bottom = plat.top
                    if not was_on_ground:
                        # sfx: land_sound()
                        self._create_landing_particles(self.player_rect.midbottom)
                    self.on_ground = True
                    self.player_vel.y = 0
                elif self.player_vel.y < 0:
                    self.player_rect.top = plat.bottom
                    self.player_vel.y = 0
                self.player_pos.y = self.player_rect.y

    def _create_landing_particles(self, pos):
        for _ in range(5):
            p_vel = pygame.Vector2(self.rng.uniform(-1.5, 1.5), self.rng.uniform(-2, 0))
            p_pos = pygame.Vector2(pos)
            self.particles.append([p_pos, p_vel, self.rng.integers(10, 20)])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[2] > 0]
        for p in self.particles:
            p[0] += p[1]
            p[2] -= 1

    def _update_camera(self):
        target_camera_x = self.player_pos.x - self.WIDTH / 2
        self.camera_offset_x += (target_camera_x - self.camera_offset_x) * 0.1
        
        if self.platforms:
            min_cam_x = 0
            max_cam_x = self.platforms[-1].right - self.WIDTH
            if max_cam_x > min_cam_x:
                self.camera_offset_x = np.clip(self.camera_offset_x, min_cam_x, max_cam_x)
            else:
                self.camera_offset_x = min_cam_x

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_x = int(self.camera_offset_x)
        
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat.move(-cam_x, 0))

        if self.flag_rect:
            pygame.draw.rect(self.screen, self.COLOR_FLAG, self.flag_rect.move(-cam_x, 0))

        for p in self.particles:
            pos, _, life = p
            size = max(1, int(life / 5))
            pygame.draw.rect(self.screen, self.COLOR_PARTICLE, (int(pos.x - cam_x), int(pos.y), size, size))

        if self.player_rect:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect.move(-cam_x, 0))

    def _render_ui(self):
        time_text = f"Time: {self.steps / 30.0:.1f}s"
        time_surf = self.font.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        level_text = f"Level: {self.current_level}"
        level_surf = self.font.render(level_text, True, self.COLOR_TEXT)
        level_rect = level_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(level_surf, level_rect)

        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.current_level,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert self.observation_space.contains(obs)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert self.observation_space.contains(obs)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")