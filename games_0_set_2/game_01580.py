
# Generated: 2025-08-27T17:35:31.690184
# Source Brief: brief_01580.md
# Brief Index: 1580

        
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
        "Controls: → to run right, ← to run left, ↑ to jump. "
        "Collect coins and reach the flag at the end of each stage."
    )

    game_description = (
        "A fast-paced 2D platformer. Navigate procedurally generated levels, "
        "collect coins, and avoid deadly pits. Complete all three stages to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.MAX_STEPS = 2500
        self.TOTAL_STAGES = 3

        # Physics
        self.GRAVITY = 0.7
        self.PLAYER_ACCEL = 1.2
        self.PLAYER_FRICTION = -0.12
        self.PLAYER_JUMP_STRENGTH = -14
        self.PLAYER_MAX_SPEED = 6

        # Colors
        self.COLOR_BG_SKY = (200, 220, 255)
        self.COLOR_PLAYER = (0, 120, 255)
        self.COLOR_PLAYER_OUTLINE = (0, 60, 150)
        self.COLOR_GROUND = (60, 180, 80)
        self.COLOR_GROUND_OUTLINE = (40, 120, 50)
        self.COLOR_PIT = (180, 50, 50)
        self.COLOR_COIN = (255, 220, 0)
        self.COLOR_COIN_OUTLINE = (200, 160, 0)
        self.COLOR_FLAG_POLE = (150, 150, 150)
        self.COLOR_FLAG = (255, 60, 60)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 40, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.stage = 0
        self.game_over = False
        self.game_won = False
        self.player_pos = None
        self.player_vel = None
        self.player_rect = None
        self.on_ground = False
        self.level_width = 0
        self.platforms = []
        self.pits = []
        self.coins = []
        self.end_flag = None
        self.particles = []
        self.camera_x = 0.0
        self.rng = None
        self.squash_timer = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.lives = 3
        self.stage = 1
        self.game_over = False
        self.game_won = False
        
        self.player_pos = pygame.Vector2(100, self.SCREEN_HEIGHT - 100)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_rect = pygame.Rect(0, 0, 24, 32)
        
        self.on_ground = False
        self.squash_timer = 0
        self.camera_x = 0
        self.particles = []

        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = []
        self.pits = []
        self.coins = []
        
        self.level_width = self.SCREEN_WIDTH * 15
        pit_chance = self.stage * 0.1
        
        # Start platform
        current_x = 0
        platform_y = self.SCREEN_HEIGHT - 40
        platform_width = 400
        self.platforms.append(pygame.Rect(current_x, platform_y, platform_width, 80))
        current_x += platform_width

        while current_x < self.level_width:
            if self.rng.random() < pit_chance:
                # Create a pit
                pit_width = self.rng.integers(80, 200)
                self.pits.append(pygame.Rect(current_x, platform_y, pit_width, 80))
                current_x += pit_width
            else:
                # Create a platform
                platform_width = self.rng.integers(150, 500)
                height_change = self.rng.integers(-60, 61)
                platform_y = np.clip(platform_y + height_change, self.SCREEN_HEIGHT - 200, self.SCREEN_HEIGHT - 40)
                
                new_platform = pygame.Rect(current_x, platform_y, platform_width, self.SCREEN_HEIGHT - platform_y)
                self.platforms.append(new_platform)
                
                # Add coins
                num_coins = self.rng.integers(1, max(2, int(platform_width / 100)))
                for i in range(num_coins):
                    coin_x = current_x + (i + 1) * (platform_width / (num_coins + 1))
                    coin_y = platform_y - self.rng.integers(40, 100)
                    self.coins.append(pygame.Rect(coin_x, coin_y, 16, 16))

                current_x += platform_width
        
        # End flag
        last_platform = self.platforms[-1]
        self.end_flag = pygame.Rect(last_platform.right - 80, last_platform.top - 80, 10, 80)


    def step(self, action):
        if self.game_over or self.game_won:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        reward = 0

        # --- Handle Input ---
        if movement == 1 and self.on_ground: # Jump
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self.on_ground = False
            self.squash_timer = 0
            self._create_particles(self.player_rect.midbottom, 10, (150, 150, 150), life=15, speed_range=(1, 4))
            # sfx: jump_sound

        accel_x = 0
        if movement == 3: # Left
            accel_x = -self.PLAYER_ACCEL
        elif movement == 4: # Right
            accel_x = self.PLAYER_ACCEL

        # --- Physics & Collisions ---
        # Horizontal movement
        self.player_vel.x += accel_x
        self.player_vel.x += self.player_vel.x * self.PLAYER_FRICTION
        if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0
        self.player_vel.x = np.clip(self.player_vel.x, -self.PLAYER_MAX_SPEED, self.PLAYER_MAX_SPEED)
        
        self.player_pos.x += self.player_vel.x
        self.player_rect.centerx = int(self.player_pos.x)
        self._handle_horizontal_collisions()

        # Vertical movement
        self.player_vel.y += self.GRAVITY
        self.player_pos.y += self.player_vel.y
        self.player_rect.centery = int(self.player_pos.y)
        self.on_ground = False
        self._handle_vertical_collisions()
        
        # --- Rewards for movement ---
        if self.player_vel.x > 0.1:
            reward += 0.01 # Adjusted from 0.1 for better balance
        elif self.player_vel.x < -0.1:
            reward -= 0.001 # Adjusted from 0.01

        # --- Other Game Logic ---
        # Collect coins
        collected_coins = []
        for coin in self.coins:
            if self.player_rect.colliderect(coin):
                collected_coins.append(coin)
                self.score += 1
                reward += 1.0
                self._create_particles(coin.center, 15, self.COLOR_COIN, life=20, speed_range=(2, 5))
                # sfx: coin_collect_sound
        self.coins = [c for c in self.coins if c not in collected_coins]

        # Check for falling into pits / out of world
        if self.player_pos.y > self.SCREEN_HEIGHT + 50:
            reward += self._handle_death()
        
        # Check for stage completion
        if self.player_rect.colliderect(self.end_flag):
            reward += self._handle_stage_clear()

        # Update particles and animations
        self._update_particles()
        if self.squash_timer > 0: self.squash_timer -= 1
        
        # Update camera
        target_camera_x = self.player_pos.x - self.SCREEN_WIDTH / 2.5
        self.camera_x += (target_camera_x - self.camera_x) * 0.1
        self.camera_x = max(0, self.camera_x)

        # --- Termination ---
        self.steps += 1
        terminated = self.game_over or self.game_won or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not (self.game_over or self.game_won):
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_horizontal_collisions(self):
        for platform in self.platforms:
            if self.player_rect.colliderect(platform):
                if self.player_vel.x > 0: # Moving right
                    self.player_rect.right = platform.left
                elif self.player_vel.x < 0: # Moving left
                    self.player_rect.left = platform.right
                self.player_pos.x = self.player_rect.centerx
                self.player_vel.x = 0

    def _handle_vertical_collisions(self):
        for platform in self.platforms:
            if self.player_rect.colliderect(platform):
                if self.player_vel.y > 0: # Moving down
                    if self.player_rect.bottom - self.player_vel.y < platform.top + 1:
                        self.player_rect.bottom = platform.top
                        self.player_pos.y = self.player_rect.centery
                        self.player_vel.y = 0
                        if not self.on_ground: # Just landed
                            self.squash_timer = 6
                            self._create_particles(self.player_rect.midbottom, 5, (150, 150, 150), life=10, speed_range=(1,3))
                            # sfx: land_sound
                        self.on_ground = True
                elif self.player_vel.y < 0: # Moving up
                    self.player_rect.top = platform.bottom
                    self.player_pos.y = self.player_rect.centery
                    self.player_vel.y = 0

    def _handle_death(self):
        self.lives -= 1
        self._create_particles(self.player_rect.center, 50, self.COLOR_PIT, life=40, speed_range=(2, 8))
        # sfx: death_sound
        if self.lives <= 0:
            self.game_over = True
            return -10
        
        # Reset player to start of level
        self.player_pos = pygame.Vector2(100, self.SCREEN_HEIGHT - 100)
        self.player_vel = pygame.Vector2(0, 0)
        self.camera_x = 0
        return -10

    def _handle_stage_clear(self):
        # sfx: stage_clear_sound
        self.stage += 1
        if self.stage > self.TOTAL_STAGES:
            self.game_won = True
            return 100 # Final win bonus
        else:
            self._generate_level()
            self.player_pos = pygame.Vector2(100, self.SCREEN_HEIGHT - 100)
            self.player_vel = pygame.Vector2(0, 0)
            self.camera_x = 0
            return 50 # Stage clear bonus

    def _create_particles(self, pos, count, color, life, speed_range):
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.uniform(speed_range[0], speed_range[1])
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.rng.integers(life // 2, life),
                'max_life': life,
                'color': color,
                'size': self.rng.integers(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG_SKY)
        # Parallax background layers
        for i in range(3, 0, -1):
            color = (
                max(0, self.COLOR_GROUND[0] - i*15),
                max(0, self.COLOR_GROUND[1] - i*15),
                max(0, self.COLOR_GROUND[2] - i*15)
            )
            offset = -self.camera_x / (i * 2)
            bg_width = self.SCREEN_WIDTH * 2
            start_x = (offset % bg_width) - bg_width
            pygame.draw.rect(self.screen, color, (start_x, self.SCREEN_HEIGHT - 20*i - 20, bg_width, 20*i))
            pygame.draw.rect(self.screen, color, (start_x + bg_width, self.SCREEN_HEIGHT - 20*i - 20, bg_width, 20*i))

        # --- Game Objects ---
        cam_x = int(self.camera_x)

        for pit in self.pits:
            if pit.right > cam_x and pit.left < cam_x + self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_PIT, pit.move(-cam_x, 0))

        for platform in self.platforms:
            if platform.right > cam_x and platform.left < cam_x + self.SCREEN_WIDTH:
                p_rect = platform.move(-cam_x, 0)
                pygame.draw.rect(self.screen, self.COLOR_GROUND, p_rect)
                pygame.draw.rect(self.screen, self.COLOR_GROUND_OUTLINE, p_rect, 3)

        for coin in self.coins:
            if coin.right > cam_x and coin.left < cam_x + self.SCREEN_WIDTH:
                # Spinning animation
                c_rect = coin.move(-cam_x, 0)
                scale = abs(math.sin(self.steps * 0.2))
                width = int(c_rect.width * scale)
                c_rect.width = max(2, width)
                c_rect.x += (coin.width - width) // 2
                
                pygame.draw.ellipse(self.screen, self.COLOR_COIN_OUTLINE, c_rect)
                pygame.draw.ellipse(self.screen, self.COLOR_COIN, c_rect.inflate(-4, -4))

        if self.end_flag.right > cam_x and self.end_flag.left < cam_x + self.SCREEN_WIDTH:
            flag_rect = self.end_flag.move(-cam_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_FLAG_POLE, flag_rect)
            poly = [
                (flag_rect.right, flag_rect.top),
                (flag_rect.right, flag_rect.top + 25),
                (flag_rect.right + 30, flag_rect.top + 12.5)
            ]
            pygame.gfxdraw.aapolygon(self.screen, poly, self.COLOR_FLAG)
            pygame.gfxdraw.filled_polygon(self.screen, poly, self.COLOR_FLAG)
        
        # --- Player ---
        p_rect = self.player_rect.move(-cam_x, 0)
        # Squash and stretch
        if self.squash_timer > 0:
            squash_factor = (self.squash_timer / 6)
            p_rect.height = int(self.player_rect.height * (1 - 0.3 * squash_factor))
            p_rect.width = int(self.player_rect.width * (1 + 0.4 * squash_factor))
            p_rect.midbottom = (self.player_rect.centerx - cam_x, self.player_rect.bottom)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, p_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, p_rect.inflate(-4, -4), border_radius=3)

        # --- Particles ---
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            pos = (int(p['pos'].x - cam_x), int(p['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size'] * (p['life']/p['max_life'])), color)

        # --- UI ---
        # Stage
        self._draw_text(f"STAGE {self.stage}", self.font_ui, (self.SCREEN_WIDTH // 2, 20))
        # Lives
        for i in range(self.lives):
            heart_poly = [(10,15),(15,10),(20,15),(15,20)]
            heart_poly = [(x + 30 + i * 25, y+5) for x, y in heart_poly]
            pygame.gfxdraw.aapolygon(self.screen, heart_poly, self.COLOR_PIT)
            pygame.gfxdraw.filled_polygon(self.screen, heart_poly, self.COLOR_PIT)
        # Score (Coins)
        self._draw_text(f"COINS: {self.score}", self.font_ui, (self.SCREEN_WIDTH - 90, 20), align="center")

        # Game Over / Win Message
        if self.game_over:
            self._draw_text("GAME OVER", self.font_title, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20))
        elif self.game_won:
            self._draw_text("YOU WIN!", self.font_title, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20))

        # --- Final Conversion ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_text(self, text, font, pos, align="center"):
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surf = font.render(text, True, self.COLOR_TEXT)
        
        shadow_rect = shadow_surf.get_rect()
        text_rect = text_surf.get_rect()

        if align == "center":
            shadow_rect.center = (pos[0] + 2, pos[1] + 2)
            text_rect.center = pos
        elif align == "topleft":
            shadow_rect.topleft = (pos[0] + 2, pos[1] + 2)
            text_rect.topleft = pos
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
        }

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Procedural Platformer")
    clock = pygame.time.Clock()
    running = True

    total_reward = 0
    
    print(env.user_guide)

    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        # Down (2) is unused
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)
        
        if done:
            print(f"Episode finished. Total Reward: {total_reward}, Info: {info}")
            # Wait for reset
            pass

    env.close()