import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game objects
class Player:
    """A helper class to store player state."""
    def __init__(self):
        self.pos = pygame.Vector2(0, 0)
        self.vel = pygame.Vector2(0, 0)
        self.size = pygame.Vector2(20, 30)
        self.on_ground = False
        self.jump_power = -9.0
        self.float_gravity_scale = 0.5
        self.current_gravity_scale = 1.0
        self.squash = 0.0  # For animation

class Platform:
    """A helper class for platform state."""
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)

class Coin:
    """A helper class for coin state."""
    def __init__(self, x, y):
        self.pos = pygame.Vector2(x, y)
        self.radius = 8
        self.animation_phase = random.uniform(0, 2 * math.pi)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to jump. Hold space for a longer jump. Hold shift to float."
    )
    game_description = (
        "Fast-paced arcade platformer. Hop between platforms, collect coins, and reach the goal."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.GRAVITY = 0.5
        self.WORLD_LENGTH = 5000

        # Colors
        self.COLOR_BG_TOP = (40, 20, 60)
        self.COLOR_BG_BOTTOM = (10, 5, 20)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_PLAYER_OUTLINE = (200, 30, 30)
        self.COLOR_PLATFORM = (60, 100, 255)
        self.COLOR_PLATFORM_TOP = (100, 140, 255)
        self.COLOR_COIN = (255, 220, 50)
        self.COLOR_COIN_OUTLINE = (200, 170, 0)
        self.COLOR_GOAL = (200, 200, 200)
        self.COLOR_FLAG = (255, 50, 50)
        self.COLOR_TEXT = (255, 255, 255)

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = Player()
        self.platforms = []
        self.coins = []
        self.goal_platform = None
        self.camera_x = 0.0
        self.last_player_x = 0.0
        self.last_jump_risky = False
        self.last_jump_safe = False
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.camera_x = 0.0
        self.last_jump_risky = False
        self.last_jump_safe = False

        self.player = Player()
        self.platforms = []
        self.coins = []
        self._generate_world()
        
        self.player.pos = pygame.Vector2(100, 200)
        self.last_player_x = self.player.pos.x

        # Initial drop to land on the first platform
        for _ in range(20): # Simulate a few frames to land
             if self.player.on_ground: break
             self._update_player()

        return self._get_observation(), self._get_info()

    def _generate_world(self):
        # Starting platform
        start_plat = Platform(50, 250, 150, 20)
        self.platforms.append(start_plat)
        
        last_plat = start_plat
        
        # Procedural generation loop
        while last_plat.rect.right < self.WORLD_LENGTH:
            difficulty_factor = min(1.0, last_plat.rect.right / self.WORLD_LENGTH)
            gap_x = self.np_random.uniform(50 + 70 * difficulty_factor, 100 + 100 * difficulty_factor)
            gap_y = self.np_random.uniform(-80, 80)
            
            new_width = self.np_random.uniform(80, 150)
            new_x = last_plat.rect.right + gap_x
            new_y = np.clip(last_plat.rect.y + gap_y, 100, self.HEIGHT - 50)
            
            new_plat = Platform(new_x, new_y, new_width, 20)
            self.platforms.append(new_plat)
            
            if self.np_random.random() < 0.7:
                coin_x = new_plat.rect.centerx
                coin_y = new_plat.rect.top - self.np_random.uniform(40, 80)
                self.coins.append(Coin(coin_x, coin_y))
            
            last_plat = new_plat
            
        self.goal_platform = Platform(last_plat.rect.x, last_plat.rect.y, last_plat.rect.width, 100)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held, shift_held)
        self._update_player()
        self._update_camera()
        
        num_coins_before = len(self.coins)
        self._update_entities_and_animations()
        newly_collected_coins = num_coins_before - len(self.coins)

        reward = self._calculate_reward(newly_collected_coins)
        self.score += newly_collected_coins * 10

        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        terminal_reward = 0
        if terminated and not truncated:
            self.game_over = True
            if self.player.pos.y > self.HEIGHT + self.player.size.y:
                terminal_reward = -10
            elif self.goal_platform and self.player.pos.x >= self.goal_platform.rect.x:
                terminal_reward = 100
        
        if truncated:
            self.game_over = True

        total_reward = reward + terminal_reward
        
        return self._get_observation(), total_reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        if self.player.on_ground and movement != 0:
            self.player.squash = 10  # Animate squash for jump
            # sfx: player_jump.wav
            
            jump_vel_x = 0
            jump_vel_y = 0
            base_jump = self.player.jump_power * (1.2 if space_held else 1.0)
            
            if movement == 1: # Up
                jump_vel_y = base_jump * 1.2; jump_vel_x = 3
            elif movement == 2: # Down (short hop forward)
                jump_vel_y = base_jump * 0.5; jump_vel_x = 4
            elif movement == 3: # Left
                jump_vel_y = base_jump * 0.8; jump_vel_x = -6
            elif movement == 4: # Right
                jump_vel_y = base_jump * 0.8; jump_vel_x = 6

            if space_held: jump_vel_x *= 1.5
                
            self.player.vel = pygame.Vector2(jump_vel_x, jump_vel_y)
            self.player.on_ground = False
        
        if shift_held and self.player.vel.y < 0:
            self.player.current_gravity_scale = self.player.float_gravity_scale
        else:
            self.player.current_gravity_scale = 1.0

    def _update_player(self):
        prev_pos = self.player.pos.copy()
        
        if not self.player.on_ground:
            self.player.vel.y += self.GRAVITY * self.player.current_gravity_scale
        else:
            # When on ground, stop horizontal sliding and vertical velocity.
            # A new jump will override this in _handle_input.
            self.player.vel.x = 0
            self.player.vel.y = 0

        self.player.vel.y = min(self.player.vel.y, 15)
        self.player.pos += self.player.vel
        
        self.player.squash *= 0.8
        
        player_rect = pygame.Rect(self.player.pos, self.player.size)
        was_on_ground = self.player.on_ground
        self.player.on_ground = False
        
        platforms_to_check = ([self.goal_platform] if self.goal_platform else []) + self.platforms
        
        for plat in platforms_to_check:
            if player_rect.colliderect(plat.rect):
                if prev_pos.y + self.player.size.y <= plat.rect.top + 1 and self.player.vel.y >= 0:
                    self.player.pos.y = plat.rect.top - self.player.size.y
                    self.player.vel.y = 0
                    self.player.on_ground = True
                    if not was_on_ground:
                        self.player.squash = -15
                        # sfx: player_land.wav
                        
                        distance_from_center = abs(player_rect.centerx - plat.rect.centerx)
                        if distance_from_center > plat.rect.width * 0.4: self.last_jump_risky = True
                        elif distance_from_center < plat.rect.width * 0.2: self.last_jump_safe = True
                    break

    def _update_camera(self):
        target_camera_x = self.player.pos.x - self.WIDTH / 3
        self.camera_x += (target_camera_x - self.camera_x) * 0.1

    def _update_entities_and_animations(self):
        player_rect = pygame.Rect(self.player.pos, self.player.size)
        collected_coins = []
        for coin in self.coins:
            coin.animation_phase += 0.2
            if player_rect.colliderect(pygame.Rect(coin.pos.x - coin.radius, coin.pos.y - coin.radius, coin.radius * 2, coin.radius * 2)):
                collected_coins.append(coin)
                # sfx: coin_collect.wav
        
        if collected_coins: self.coins = [c for c in self.coins if c not in collected_coins]
        
        self.platforms = [p for p in self.platforms if p.rect.right > self.camera_x]
        self.coins = [c for c in self.coins if c.pos.x > self.camera_x - 20]

    def _calculate_reward(self, newly_collected_coins):
        reward = 0.0
        
        dx = self.player.pos.x - self.last_player_x
        reward += 0.02 * dx if dx > 0 else 0.01 * dx
        self.last_player_x = self.player.pos.x
        
        reward += newly_collected_coins * 1.0
        
        if self.last_jump_risky: reward += 0.5; self.last_jump_risky = False
        if self.last_jump_safe: reward -= 0.2; self.last_jump_safe = False
            
        return reward
        
    def _check_termination(self):
        if self.player.pos.y > self.HEIGHT + self.player.size.y: return True
        if self.goal_platform and self.player.pos.x > self.goal_platform.rect.x:
            if pygame.Rect(self.player.pos, self.player.size).colliderect(self.goal_platform.rect):
                return True
        return False
        
    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG_BOTTOM)
        grad_rect = pygame.Rect(0, 0, self.WIDTH, self.HEIGHT)
        for y in range(grad_rect.h):
            interp = y / grad_rect.h
            color = tuple(int(c1 * (1 - interp) + c2 * interp) for c1, c2 in zip(self.COLOR_BG_TOP, self.COLOR_BG_BOTTOM))
            pygame.draw.line(self.screen, color, (grad_rect.left, y), (grad_rect.right, y))
            
    def _render_game_elements(self):
        cam_x = int(self.camera_x)
        
        if self.goal_platform:
            goal_rect = self.goal_platform.rect.move(-cam_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_GOAL, goal_rect)
            pole_rect = pygame.Rect(goal_rect.left + 10, goal_rect.top - 50, 5, 50)
            pygame.draw.rect(self.screen, self.COLOR_GOAL, pole_rect)
            flag_points = [(pole_rect.right, pole_rect.top), (pole_rect.right + 30, pole_rect.top + 15), (pole_rect.right, pole_rect.top + 30)]
            pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)

        for plat in self.platforms:
            p_rect = plat.rect.move(-cam_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p_rect)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_TOP, (p_rect.left, p_rect.top, p_rect.width, 5))
            
        for coin in self.coins:
            radius = int(coin.radius * (0.8 + 0.2 * math.sin(coin.animation_phase)))
            pos = (int(coin.pos.x - cam_x), int(coin.pos.y))
            if -radius < pos[0] < self.WIDTH + radius:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_COIN)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_COIN_OUTLINE)

        squash_y = self.player.squash * (self.player.size.y / self.player.size.x)
        player_w = int(self.player.size.x - self.player.squash)
        player_h = int(self.player.size.y + squash_y)
        player_rect = pygame.Rect(int(self.player.pos.x - cam_x), int(self.player.pos.y + (self.player.size.y - player_h)), player_w, player_h)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect.inflate(4, 4), border_radius=6)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=5)
        
    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))
        
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}
        
    def close(self):
        pygame.quit()