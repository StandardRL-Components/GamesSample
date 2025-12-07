
# Generated: 2025-08-27T17:34:58.579798
# Source Brief: brief_01574.md
# Brief Index: 1574

        
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
    user_guide = "Controls: ←→ to move, ↑ to jump. Reach the green flag to win!"

    # Must be a short, user-facing description of the game:
    game_description = "A fast-paced, procedurally generated platformer where risk-taking is rewarded."

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_LENGTH = 8000

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Colors
        self.COLOR_BG_TOP = (100, 149, 237)
        self.COLOR_BG_BOTTOM = (0, 0, 139)
        self.COLOR_PLAYER = (255, 69, 0)
        self.COLOR_PLATFORM = (139, 69, 19)
        self.COLOR_PIT = (10, 10, 10)
        self.COLOR_FLAG = (0, 200, 0)
        self.COLOR_FLAGPOLE = (192, 192, 192)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE = (220, 220, 220)

        # Game constants
        self.GRAVITY = 0.4
        self.PLAYER_JUMP_STRENGTH = 10
        self.PLAYER_MOVE_SPEED = 4
        self.PLAYER_SIZE = (20, 20)
        self.SCROLL_SPEED = 2.5
        self.MAX_LEVELS = 3
        self.LEVEL_TIME_SECONDS = 60
        self.RISKY_JUMP_THRESHOLD = self.PLAYER_SIZE[0] * 7

        # State variables will be initialized in reset()
        self.level = 0
        self.score = 0
        self.steps = 0
        self.game_over = True # Start in a game-over state to trigger full reset
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.game_over:
            self.score = 0
            self.level = 1
            self.steps = 0
        else:
            self.level += 1

        self.game_over = False
        self.win = False

        self.player_pos = [self.WIDTH // 4, self.HEIGHT // 2]
        self.player_vel = [0, 0]
        self.on_ground = False
        self.is_jumping = False
        self.jump_start_x = 0
        
        self.world_scroll_x = 0.0
        self.level_timer = self.LEVEL_TIME_SECONDS * 30
        self.particles = []
        
        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = []
        self.pits = []
        
        difficulty_mod = 1.0 - (self.level - 1) * 0.1
        min_plat_w = max(50, 80 * difficulty_mod)
        max_plat_w = max(100, 200 * difficulty_mod)
        min_gap = 40
        max_gap = max(60, 120 * difficulty_mod)
        pit_chance = 0.1 + (self.level - 1) * 0.1

        start_plat_w = 300
        self.platforms.append(pygame.Rect(0, self.HEIGHT - 100, start_plat_w, 100))
        
        current_x = float(start_plat_w)
        while current_x < self.WORLD_LENGTH:
            gap = self.np_random.uniform(min_gap, max_gap)
            current_x += gap
            
            if self.np_random.random() < pit_chance:
                pit_w = self.np_random.uniform(60, 150)
                self.pits.append(pygame.Rect(current_x, self.HEIGHT - 20, pit_w, 20))
                current_x += pit_w
            else:
                plat_w = self.np_random.uniform(min_plat_w, max_plat_w)
                plat_h = self.np_random.uniform(40, self.HEIGHT - 150)
                plat_y = self.HEIGHT - plat_h
                self.platforms.append(pygame.Rect(current_x, plat_y, plat_w, plat_h))
                current_x += plat_w
        
        last_entity = self.platforms[-1]
        flag_x = last_entity.right + 100
        flag_y = last_entity.y - 50 if last_entity.y > 100 else self.HEIGHT - 150
        self.flag_pos = [flag_x, flag_y]
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.level_timer -= 1
        reward = 0.0

        movement = action[0]
        
        player_moved_horizontally = False
        if movement == 3: # Left
            self.player_pos[0] -= self.PLAYER_MOVE_SPEED
            player_moved_horizontally = True
        elif movement == 4: # Right
            self.player_pos[0] += self.PLAYER_MOVE_SPEED
            reward += 0.1
            player_moved_horizontally = True
        
        if not player_moved_horizontally:
            reward -= 0.01

        if movement == 1 and self.on_ground:
            self.player_vel[1] = -self.PLAYER_JUMP_STRENGTH
            self.on_ground = False
            self.is_jumping = True
            self.jump_start_x = self.world_scroll_x + self.player_pos[0]
            # sfx: jump

        self.world_scroll_x += self.SCROLL_SPEED
        self.player_vel[1] += self.GRAVITY
        self.player_pos[1] += self.player_vel[1]

        self.player_pos[0] = max(0, min(self.player_pos[0], self.WIDTH - self.PLAYER_SIZE[0]))

        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE[0], self.PLAYER_SIZE[1])
        
        prev_on_ground = self.on_ground
        self.on_ground = False
        for plat in self.platforms:
            plat_screen_rect = plat.move(-self.world_scroll_x, 0)
            if player_rect.colliderect(plat_screen_rect):
                if self.player_vel[1] >= 0 and player_rect.bottom - self.player_vel[1] <= plat_screen_rect.top + 1:
                    self.player_pos[1] = plat_screen_rect.top - self.PLAYER_SIZE[1]
                    self.player_vel[1] = 0
                    self.on_ground = True
                    if not prev_on_ground: # Just landed
                        # sfx: land
                        if self.is_jumping:
                            jump_end_x = self.world_scroll_x + self.player_pos[0]
                            jump_dist = jump_end_x - self.jump_start_x
                            if jump_dist > self.RISKY_JUMP_THRESHOLD:
                                reward += 2.0
                            else:
                                reward -= 0.2
                            self.is_jumping = False
                    break
        
        if player_rect.top > self.HEIGHT:
            self.game_over = True
            reward -= 10
            # sfx: fall_death
        
        for pit in self.pits:
            pit_screen_rect = pit.move(-self.world_scroll_x, 0)
            if player_rect.colliderect(pit_screen_rect):
                self.game_over = True
                reward -= 10
                # sfx: fall_death
                break
        
        if self.level_timer <= 0:
            self.game_over = True
            # sfx: timeout
        
        flag_rect = pygame.Rect(self.flag_pos[0] - self.world_scroll_x, self.flag_pos[1], 40, 30)
        if player_rect.colliderect(flag_rect):
            reward += 5
            # sfx: level_complete
            if self.level == self.MAX_LEVELS:
                self.win = True
                self.game_over = True
                reward += 50
            else:
                self.reset()
        
        self._update_and_spawn_particles()
        
        self.score += reward
        terminated = self.game_over
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_and_spawn_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

        for pit in self.pits:
            pit_screen_rect = pit.move(-self.world_scroll_x, 0)
            if -pit_screen_rect.width < pit_screen_rect.x < self.WIDTH:
                 if self.np_random.random() < 0.5:
                    px = self.np_random.uniform(pit_screen_rect.left, pit_screen_rect.right)
                    pvy = self.np_random.uniform(-1.5, -0.5)
                    plife = self.np_random.integers(15, 30)
                    psize = self.np_random.uniform(1, 3)
                    self.particles.append({'pos': [px, pit_screen_rect.top], 'vel': [0, pvy], 'life': plife, 'size': psize})
    
    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_background(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = tuple(
                int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio)
                for i in range(3)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, p['pos'], p['size'])

        for pit in self.pits:
            pit_screen_rect = pit.move(-self.world_scroll_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_PIT, pit_screen_rect)
            
        for plat in self.platforms:
            plat_screen_rect = plat.move(-self.world_scroll_x, 0)
            if plat_screen_rect.right < 0 or plat_screen_rect.left > self.WIDTH:
                continue
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat_screen_rect)
            top_edge_color = (min(255, self.COLOR_PLATFORM[0] + 20),
                              min(255, self.COLOR_PLATFORM[1] + 20),
                              min(255, self.COLOR_PLATFORM[2] + 20))
            pygame.draw.rect(self.screen, top_edge_color, (plat_screen_rect.x, plat_screen_rect.y, plat_screen_rect.width, 5))

        flag_x = self.flag_pos[0] - self.world_scroll_x
        flag_y = self.flag_pos[1]
        flagpole_rect = pygame.Rect(int(flag_x - 5), int(flag_y), 5, self.HEIGHT - flag_y)
        flag_rect = pygame.Rect(int(flag_x), int(flag_y), 40, 30)
        pygame.draw.rect(self.screen, self.COLOR_FLAGPOLE, flagpole_rect)
        pygame.draw.rect(self.screen, self.COLOR_FLAG, flag_rect)

        player_rect = pygame.Rect(int(self.player_pos[0]), int(self.player_pos[1]), self.PLAYER_SIZE[0], self.PLAYER_SIZE[1])
        
        glow_rect = player_rect.inflate(6, 6)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_PLAYER, 50), glow_surface.get_rect(), border_radius=4)
        self.screen.blit(glow_surface, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)

    def _render_ui(self):
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_text = self.font_small.render(f"Time: {math.ceil(max(0, self.level_timer / 30))}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        level_text = self.font_small.render(f"Level: {self.level} / {self.MAX_LEVELS}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.WIDTH // 2 - level_text.get_width() // 2, self.HEIGHT - 30))

        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            end_text = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 20))
            
            shadow_text = self.font_large.render(message, True, color)
            shadow_rect = shadow_text.get_rect(center=(self.WIDTH // 2 + 3, self.HEIGHT // 2 - 17))
            
            self.screen.blit(shadow_text, shadow_rect)
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
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