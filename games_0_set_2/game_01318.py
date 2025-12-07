
# Generated: 2025-08-27T16:45:12.309893
# Source Brief: brief_01318.md
# Brief Index: 1318

        
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
        "Controls: Press space to jump. Precise timing is key to reach the flag."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist platformer where you control a square that must jump between "
        "platforms to reach the goal. Earn points for landing on new platforms and "
        "a bonus for finishing quickly. Don't fall or run out of time!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 30
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_PLATFORM = (100, 110, 120)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_GOAL = (100, 220, 100)
        self.COLOR_UI = (230, 230, 230)
        self.COLOR_MSG_WIN = (120, 255, 120)
        self.COLOR_MSG_LOSE = (255, 80, 80)
        
        # Physics
        self.GRAVITY = 0.5
        self.JUMP_STRENGTH = -11.5
        self.PLAYER_SIZE = pygame.Vector2(20, 20)

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20)
        self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.time_left = 0.0
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.jump_request = False
        self.platforms = []
        self.flag_rect = pygame.Rect(0, 0, 0, 0)
        self.last_platform_landed = None
        self.particles = []

        self.reset()

        # self.validate_implementation() # Call this for your own testing
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.time_left = float(self.TIME_LIMIT_SECONDS)
        
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = True
        self.jump_request = False
        
        self.particles = []
        
        self._generate_level()

        self.last_platform_landed = self.platforms[0]
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = []
        start_plat = pygame.Rect(50, self.HEIGHT - 50, 100, 20)
        self.platforms.append(start_plat)
        
        self.player_pos = pygame.Vector2(
            start_plat.centerx - self.PLAYER_SIZE.x / 2,
            start_plat.top - self.PLAYER_SIZE.y
        )

        current_pos = pygame.Vector2(start_plat.x, start_plat.y)
        num_platforms = 12

        for i in range(num_platforms):
            w = self.np_random.integers(low=60, high=120)
            h = 20
            
            max_jump_height = (self.JUMP_STRENGTH**2) / (2 * self.GRAVITY) * 0.9 # 90% of max height
            dx = self.np_random.uniform(low=70, high=160)
            dy = self.np_random.uniform(low=-max_jump_height, high=40)

            new_x = current_pos.x + dx
            new_y = current_pos.y + dy

            new_x = np.clip(new_x, 0, self.WIDTH - w)
            new_y = np.clip(new_y, 50, self.HEIGHT - h)

            if abs(new_x - current_pos.x) < w:
                new_x = (current_pos.x + w + 10) % (self.WIDTH - w)

            plat_rect = pygame.Rect(int(new_x), int(new_y), w, h)
            self.platforms.append(plat_rect)
            current_pos.x, current_pos.y = new_x, new_y

        last_plat = self.platforms[-1]
        self.flag_rect = pygame.Rect(last_plat.centerx - 10, last_plat.top - 40, 20, 40)

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Handle Input ---
        space_held = action[1] == 1
        self.jump_request = space_held and self.on_ground

        # --- 2. Update State & Reward ---
        self.steps += 1
        self.time_left = max(0, self.time_left - 1.0 / self.FPS)
        reward = 0.0

        if not self.jump_request and self.last_platform_landed != self.platforms[0]:
            reward -= 0.02

        # --- 3. Physics & Collisions ---
        self._update_player_physics()
        landing_reward, new_platform = self._handle_collisions()
        reward += landing_reward
        if new_platform:
            self.last_platform_landed = new_platform

        # --- 4. Update Effects ---
        self._update_particles()

        # --- 5. Check Termination ---
        terminated = False
        player_rect = pygame.Rect(self.player_pos, self.PLAYER_SIZE)

        if player_rect.colliderect(self.flag_rect):
            reward += 10.0
            reward += math.floor(self.time_left)
            self.win = True
            terminated = True
            # SFX: Win

        elif self.player_pos.y > self.HEIGHT:
            reward = -100.0
            terminated = True
            # SFX: Fall

        elif self.time_left <= 0 or self.steps >= self.MAX_STEPS:
            terminated = True
            # SFX: Time up

        if terminated:
            self.game_over = True
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player_physics(self):
        if self.jump_request:
            self.player_vel.y = self.JUMP_STRENGTH
            self.on_ground = False
            # SFX: Jump

        if not self.on_ground:
            self.player_vel.y += self.GRAVITY
        
        self.player_pos += self.player_vel

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos, self.PLAYER_SIZE)
        landing_reward = 0.0
        new_platform_landed = None
        
        self.on_ground = False
        for plat in self.platforms:
            if player_rect.colliderect(plat) and self.player_vel.y > 0:
                prev_bottom = self.player_pos.y + self.PLAYER_SIZE.y - self.player_vel.y
                if prev_bottom <= plat.top + 1:
                    self.player_pos.y = plat.top - self.PLAYER_SIZE.y
                    self.player_vel.y = 0
                    self.on_ground = True
                    
                    self._create_landing_particles(player_rect.midbottom)
                    # SFX: Land
                    
                    if plat != self.last_platform_landed:
                        landing_reward = 0.1
                        new_platform_landed = plat
                    break
        return landing_reward, new_platform_landed

    def _create_landing_particles(self, pos):
        count = self.np_random.integers(5, 10)
        for _ in range(count):
            angle = self.np_random.uniform(math.pi * 1.1, math.pi * 1.9)
            speed = self.np_random.uniform(1, 3)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            radius = self.np_random.uniform(2, 4)
            life = self.np_random.integers(15, 25)
            self.particles.append([pygame.Vector2(pos), vel, radius, life, life])

    def _update_particles(self):
        for p in self.particles[:]:
            p[0] += p[1]
            p[3] -= 1
            if p[3] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat)

        pygame.draw.rect(self.screen, self.COLOR_GOAL, self.flag_rect)

        for pos, vel, radius, life, max_life in self.particles:
            current_radius = int(max(0, radius * (life / max_life)))
            if current_radius > 0:
                pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), current_radius, self.COLOR_PLAYER)
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), current_radius, self.COLOR_PLAYER)

        player_rect = pygame.Rect(int(self.player_pos.x), int(self.player_pos.y), *self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)

    def _render_ui(self):
        score_surf = self.font_ui.render(f"Score: {self.score:.2f}", True, self.COLOR_UI)
        self.screen.blit(score_surf, (10, 10))

        timer_surf = self.font_ui.render(f"Time: {self.time_left:.1f}", True, self.COLOR_UI)
        timer_rect = timer_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(timer_surf, timer_rect)

        if self.game_over:
            msg, color = ("YOU WIN!", self.COLOR_MSG_WIN) if self.win else ("GAME OVER", self.COLOR_MSG_LOSE)
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            # Simple shadow for readability
            shadow_surf = self.font_msg.render(msg, True, (0,0,0))
            self.screen.blit(shadow_surf, msg_rect.move(2, 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.font.quit()
        pygame.quit()