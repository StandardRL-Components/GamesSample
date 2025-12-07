
# Generated: 2025-08-27T20:40:28.577814
# Source Brief: brief_02538.md
# Brief Index: 2538

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper class for particles
class Particle:
    def __init__(self, x, y, color, life, size_range=(2, 5), vel_range=(-2, 2)):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(random.uniform(vel_range[0], vel_range[1]), random.uniform(vel_range[0], vel_range[1]))
        self.color = color
        self.life = life
        self.max_life = life
        self.size = random.uniform(size_range[0], size_range[1])

    def update(self):
        self.pos += self.vel
        self.life -= 1
        self.vel.y += 0.1 # Gravity on particles

    def draw(self, surface, camera_offset):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            color = self.color + (alpha,)
            temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (self.size, self.size), self.size)
            surface.blit(temp_surf, (int(self.pos.x - camera_offset.x - self.size), int(self.pos.y - camera_offset.y - self.size)))

# Helper class for the player
class Player:
    def __init__(self, x, y):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)
        self.size = pygame.Vector2(20, 20)
        self.color = (255, 255, 255)
        self.on_ground = False
        self.max_speed = 4.0
        self.acceleration = 0.5
        self.friction = 0.85
        self.gravity = 0.6
        self.jump_strength = -11
        self.just_jumped = False

    def get_rect(self):
        return pygame.Rect(self.pos.x, self.pos.y, self.size.x, self.size.y)

    def move(self, direction):
        if direction == "left":
            self.vel.x -= self.acceleration
        elif direction == "right":
            self.vel.x += self.acceleration

    def jump(self):
        if self.on_ground:
            self.vel.y = self.jump_strength
            self.on_ground = False
            self.just_jumped = True # To trigger particles

    def update(self, platforms):
        # Horizontal movement
        self.vel.x = max(-self.max_speed, min(self.max_speed, self.vel.x))
        self.pos.x += self.vel.x
        self.vel.x *= self.friction
        if abs(self.vel.x) < 0.1: self.vel.x = 0

        # Horizontal collision
        player_rect = self.get_rect()
        for platform in platforms:
            if player_rect.colliderect(platform):
                if self.vel.x > 0: # Moving right
                    player_rect.right = platform.left
                elif self.vel.x < 0: # Moving left
                    player_rect.left = platform.right
                self.pos.x = player_rect.x
                self.vel.x = 0

        # Vertical movement
        self.vel.y += self.gravity
        self.pos.y += self.vel.y
        self.on_ground = False

        # Vertical collision
        player_rect = self.get_rect()
        for platform in platforms:
            if player_rect.colliderect(platform):
                if self.vel.y > 0: # Moving down
                    player_rect.bottom = platform.top
                    self.on_ground = True
                    self.vel.y = 0
                elif self.vel.y < 0: # Moving up
                    player_rect.top = platform.bottom
                    self.vel.y = 0
                self.pos.y = player_rect.y

    def draw(self, surface, camera_offset):
        rect = self.get_rect()
        rect.topleft -= camera_offset
        pygame.draw.rect(surface, self.color, rect)
        # Simple glow effect
        glow_rect = rect.inflate(6, 6)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.color + (50,), glow_surf.get_rect(), border_radius=3)
        surface.blit(glow_surf, glow_rect.topleft)

# Helper class for enemies
class Enemy:
    def __init__(self, x, y, enemy_type, platform_rect, speed_multiplier=1.0):
        self.pos = pygame.Vector2(x, y)
        self.type = enemy_type
        self.speed_multiplier = speed_multiplier
        self.color = (255, 50, 50)
        self.initial_y = y
        self.platform_rect = platform_rect

        if self.type == 1: # Patroller
            self.size = pygame.Vector2(18, 18)
            self.vel = pygame.Vector2(1.5 * self.speed_multiplier, 0)
            self.patrol_min = platform_rect.left
            self.patrol_max = platform_rect.right - self.size.x
        elif self.type == 2: # Jumper
            self.size = pygame.Vector2(16, 16)
            self.vel = pygame.Vector2(0, -5)
            self.gravity = 0.2
        else: # Static
            self.size = pygame.Vector2(15, 15)
            self.vel = pygame.Vector2(0, 0)

    def get_rect(self):
        return pygame.Rect(self.pos.x, self.pos.y, self.size.x, self.size.y)

    def update(self, speed_multiplier):
        if self.type == 1: # Patroller
            self.vel.x = 1.5 * speed_multiplier * (1 if self.vel.x > 0 else -1)
            self.pos.x += self.vel.x
            if self.pos.x <= self.patrol_min or self.pos.x >= self.patrol_max:
                self.vel.x *= -1
                self.pos.x = max(self.patrol_min, min(self.pos.x, self.patrol_max))
        elif self.type == 2: # Jumper
            self.vel.y += self.gravity
            self.pos.y += self.vel.y
            if self.pos.y > self.initial_y:
                self.pos.y = self.initial_y
                self.vel.y = -5 * speed_multiplier

    def draw(self, surface, camera_offset):
        rect = self.get_rect()
        rect.topleft -= camera_offset
        if self.type == 1: # Square
            pygame.draw.rect(surface, self.color, rect)
        elif self.type == 2: # Circle
            pygame.gfxdraw.filled_circle(surface, int(rect.centerx), int(rect.centery), int(self.size.x / 2), self.color)
        else: # Triangle
            points = [(rect.midtop), (rect.bottomleft), (rect.bottomright)]
            pygame.draw.polygon(surface, self.color, points)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to run, ↑ or Space to jump. Collect coins and reach the flag!"
    )

    game_description = (
        "A fast-paced, procedurally generated platformer. Collect coins, avoid enemies, and race to the finish flag."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.LEVEL_WIDTH = 3200

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Colors
        self.COLOR_BG_TOP = (70, 100, 150)
        self.COLOR_BG_BOTTOM = (100, 140, 200)
        self.COLOR_PLATFORM = (50, 150, 50)
        self.COLOR_COIN = (255, 223, 0)
        self.COLOR_FLAGPOLE = (150, 150, 150)
        self.COLOR_FLAG = (255, 255, 255)

        self.rng = None
        self.player = None
        self.platforms = []
        self.coins = []
        self.enemies = []
        self.particles = []
        self.flag_pos = None
        self.flag_rect = None
        self.camera_offset = pygame.Vector2(0, 0)
        
        self.reset()
        self.validate_implementation()

    def _generate_level(self):
        self.platforms.clear()
        self.coins.clear()
        self.enemies.clear()

        # Create starting platform
        plat_y = self.HEIGHT - 50
        start_plat = pygame.Rect(0, plat_y, 200, 50)
        self.platforms.append(start_plat)

        current_x = start_plat.right
        
        # Procedurally generate platforms
        while current_x < self.LEVEL_WIDTH:
            gap = self.rng.integers(30, 90)
            width = self.rng.integers(100, 300)
            y_change = self.rng.integers(-60, 60)
            plat_y = np.clip(plat_y + y_change, self.HEIGHT / 2, self.HEIGHT - 30)

            new_plat = pygame.Rect(current_x + gap, plat_y, width, self.HEIGHT - plat_y)
            self.platforms.append(new_plat)

            # Add coins
            if self.rng.random() < 0.7:
                for i in range(self.rng.integers(1, 5)):
                    coin_pos = (new_plat.left + self.rng.integers(10, width - 10), plat_y - 20)
                    self.coins.append(pygame.Rect(coin_pos[0], coin_pos[1], 15, 15))

            # Add enemies
            if self.rng.random() < 0.4 and new_plat.width > 80:
                enemy_type = self.rng.integers(1, 4)
                enemy_x = new_plat.left + self.rng.integers(20, new_plat.width - 20)
                enemy_y = plat_y - (18 if enemy_type == 1 else 16 if enemy_type == 2 else 15)
                self.enemies.append(Enemy(enemy_x, enemy_y, enemy_type, new_plat))

            current_x = new_plat.right
        
        # Place flag on the last platform
        last_plat = self.platforms[-1]
        self.flag_pos = pygame.Vector2(last_plat.right - 40, last_plat.top)
        self.flag_rect = pygame.Rect(self.flag_pos.x, self.flag_pos.y - 50, 10, 50)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self._generate_level()
        
        self.player = Player(100, self.HEIGHT - 100)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.enemy_speed_multiplier = 1.0
        self.particles.clear()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Time penalty
        terminal_reward = 0

        movement, space_held, _ = action
        space_held = space_held == 1
        
        # 1. Handle Input
        if movement == 3: # Left
            self.player.move("left")
        elif movement == 4: # Right
            self.player.move("right")
        
        if movement == 1 or space_held: # Jump
            self.player.jump()

        # 2. Update Game State
        self.player.update(self.platforms)
        
        if self.player.just_jumped:
            # sound: JumpSfx()
            for _ in range(10):
                self.particles.append(Particle(self.player.pos.x + self.player.size.x / 2, self.player.pos.y + self.player.size.y, (200, 200, 200), 20, (1, 3), (-1.5, 1.5)))
            self.player.just_jumped = False

        if self.steps % 500 == 0 and self.steps > 0:
            self.enemy_speed_multiplier = min(2.0, self.enemy_speed_multiplier + 0.05)
        
        for enemy in self.enemies:
            enemy.update(self.enemy_speed_multiplier)

        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)

        # 3. Check Collisions & Events
        player_rect = self.player.get_rect()

        # Coins
        for coin in self.coins[:]:
            if player_rect.colliderect(coin):
                self.coins.remove(coin)
                self.score += 1
                reward += 1
                # sound: CoinSfx()
                for _ in range(15):
                    self.particles.append(Particle(coin.centerx, coin.centery, self.COLOR_COIN, 30, (2, 4), (-2.5, 2.5)))

        # Enemies
        for enemy in self.enemies:
            if player_rect.colliderect(enemy.get_rect()):
                self.game_over = True
                terminal_reward = -10
                # sound: FailSfx()
                break
        
        # Flag
        if not self.game_over and self.flag_rect.colliderect(player_rect):
            self.game_over = True
            terminal_reward = 100
            # sound: WinSfx()

        # Out of bounds
        if self.player.pos.y > self.HEIGHT:
            self.game_over = True
            terminal_reward = -10
            # sound: FailSfx()

        # 4. Finalize step
        self.steps += 1
        terminated = self.game_over or self.steps >= 2000
        
        if terminated:
            reward += terminal_reward

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _get_observation(self):
        # Update camera
        self.camera_offset.x = max(0, min(self.player.pos.x - self.WIDTH / 2, self.LEVEL_WIDTH - self.WIDTH))
        self.camera_offset.y = 0

        # Draw background gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # Draw game elements
        for platform in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, platform.move(-self.camera_offset.x, -self.camera_offset.y))

        for coin in self.coins:
            pygame.draw.rect(self.screen, self.COLOR_COIN, coin.move(-self.camera_offset.x, -self.camera_offset.y))
        
        for enemy in self.enemies:
            enemy.draw(self.screen, self.camera_offset)
        
        # Draw flag
        flagpole_rect = self.flag_rect.move(-self.camera_offset.x, -self.camera_offset.y)
        pygame.draw.rect(self.screen, self.COLOR_FLAGPOLE, flagpole_rect)
        flag_points = [
            (flagpole_rect.left + 2, flagpole_rect.top),
            (flagpole_rect.left - 30, flagpole_rect.top + 15),
            (flagpole_rect.left + 2, flagpole_rect.top + 30)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)

        self.player.draw(self.screen, self.camera_offset)

        for p in self.particles:
            p.draw(self.screen, self.camera_offset)

        # Render UI
        score_text = self.font.render(f"COINS: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": (self.player.pos.x, self.player.pos.y),
        }

    def close(self):
        pygame.quit()

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Change to 'windows' or 'dummy' as needed

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pixel Platformer")
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    clock = pygame.time.Clock()
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_UP]:
            action[0] = 1 # Can also use movement for jump

        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Run at 30 FPS

    env.close()