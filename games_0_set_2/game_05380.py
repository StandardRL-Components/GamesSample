
# Generated: 2025-08-28T04:50:35.718201
# Source Brief: brief_05380.md
# Brief Index: 5380

        
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
        "Controls: Use arrow keys to aim. Press Space to fire projectiles. Press Shift to place a defensive block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your fortress from waves of invaders. Place blocks and shoot down enemies to survive."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 4500 # 2.5 minutes at 30fps

    COLOR_BG = (10, 10, 20)
    COLOR_FORTRESS = (50, 100, 255)
    COLOR_BLOCK = (80, 120, 255)
    COLOR_INVADER = (255, 50, 50)
    COLOR_PROJECTILE = (255, 255, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_HEALTH_BG = (100, 0, 0)
    COLOR_HEALTH_FG = (0, 200, 0)
    
    FORTRESS_MAX_HEALTH = 100
    BLOCK_MAX_HEALTH = 10
    INVADER_MAX_HEALTH = 5
    
    PROJECTILE_DAMAGE = 2
    INVADER_DAMAGE = 10
    
    WAVE_DEFINITIONS = [
        {"count": 5, "speed": 0.5},
        {"count": 10, "speed": 0.7},
        {"count": 15, "speed": 0.9}
    ]
    
    INTERMISSION_TIME = 90 # 3 seconds

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans", 18)
        self.font_large = pygame.font.SysFont("sans", 48)
        
        self.fortress = None
        self.blocks = None
        self.invaders = None
        self.projectiles = None
        self.particles = None
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.current_wave_index = 0
        self.wave_state = "intermission" # intermission, spawning, active
        self.wave_timer = 0
        
        self.shoot_cooldown = 0
        self.block_cooldown = 0
        self.aim_direction = pygame.Vector2(0, -1) # Start aiming up

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.fortress = PlayerFortress(self.WIDTH / 2, self.HEIGHT / 2, self.FORTRESS_MAX_HEALTH)
        self.blocks = pygame.sprite.Group()
        self.invaders = pygame.sprite.Group()
        self.projectiles = pygame.sprite.Group()
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.current_wave_index = 0
        self.wave_state = "intermission"
        self.wave_timer = self.INTERMISSION_TIME

        self.shoot_cooldown = 0
        self.block_cooldown = 0
        self.aim_direction = pygame.Vector2(0, -1)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1
        
        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)

        # --- Update Game State ---
        self._update_wave_state()

        self.projectiles.update()
        self.invaders.update(self.fortress, self.blocks)
        self._update_particles()
        
        # --- Handle Collisions ---
        reward += self._handle_collisions()

        # --- Check Wave Completion ---
        if self.wave_state == "active" and not self.invaders:
            self.score += 50
            reward += 50
            self.current_wave_index += 1
            if self.current_wave_index >= len(self.WAVE_DEFINITIONS):
                self.win = True
                self.game_over = True
                self.score += 100
                reward += 100
            else:
                # sfx: wave complete fanfare
                self.wave_state = "intermission"
                self.wave_timer = self.INTERMISSION_TIME

        # --- Cooldowns ---
        if self.shoot_cooldown > 0: self.shoot_cooldown -= 1
        if self.block_cooldown > 0: self.block_cooldown -= 1

        # --- Termination Conditions ---
        terminated = False
        if self.fortress.health <= 0 or self.steps >= self.MAX_STEPS or self.game_over:
            terminated = True
            if self.fortress.health <= 0:
                self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _handle_input(self, movement, space_held, shift_held):
        if movement == 1: self.aim_direction = pygame.Vector2(0, -1)  # Up
        elif movement == 2: self.aim_direction = pygame.Vector2(0, 1)   # Down
        elif movement == 3: self.aim_direction = pygame.Vector2(-1, 0)  # Left
        elif movement == 4: self.aim_direction = pygame.Vector2(1, 0)   # Right

        if space_held and self.shoot_cooldown == 0:
            # sfx: shoot
            self.projectiles.add(Projectile(self.fortress.rect.center, self.aim_direction))
            self.shoot_cooldown = 10 # 3 shots per second

        if shift_held and self.block_cooldown == 0:
            self._place_block()
            self.block_cooldown = 30 # 1 block per second

    def _place_block(self):
        placement_pos = pygame.Vector2(self.fortress.rect.center) + self.aim_direction * 40
        new_block_rect = pygame.Rect(0, 0, 20, 20)
        new_block_rect.center = placement_pos

        # Prevent placing on fortress or other blocks
        if not new_block_rect.colliderect(self.fortress.rect):
            can_place = True
            for block in self.blocks:
                if new_block_rect.colliderect(block.rect):
                    can_place = False
                    break
            if can_place:
                # sfx: place block
                self.blocks.add(Block(placement_pos, self.BLOCK_MAX_HEALTH))

    def _update_wave_state(self):
        if self.wave_state == "intermission":
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.wave_state = "spawning"
        elif self.wave_state == "spawning":
            self._spawn_invaders()
            self.wave_state = "active"

    def _spawn_invaders(self):
        wave_info = self.WAVE_DEFINITIONS[self.current_wave_index]
        for _ in range(wave_info["count"]):
            side = self.np_random.integers(0, 4)
            if side == 0: # Top
                x, y = self.np_random.integers(0, self.WIDTH), -20
            elif side == 1: # Bottom
                x, y = self.np_random.integers(0, self.WIDTH), self.HEIGHT + 20
            elif side == 2: # Left
                x, y = -20, self.np_random.integers(0, self.HEIGHT)
            else: # Right
                x, y = self.WIDTH + 20, self.np_random.integers(0, self.HEIGHT)
            
            self.invaders.add(Invader((x, y), wave_info["speed"], self.INVADER_MAX_HEALTH))

    def _handle_collisions(self):
        reward = 0
        # Projectiles vs Invaders
        hits = pygame.sprite.groupcollide(self.projectiles, self.invaders, True, False)
        for proj, inv_list in hits.items():
            for inv in inv_list:
                # sfx: hit
                inv.take_damage(self.PROJECTILE_DAMAGE)
                reward += 0.1
                self._create_particles(inv.rect.center, self.COLOR_PROJECTILE, 5)
                if inv.health <= 0:
                    # sfx: explosion
                    inv.kill()
                    self.score += 1
                    reward += 1
                    self._create_particles(inv.rect.center, self.COLOR_INVADER, 15)

        # Invaders vs Blocks
        block_hits = pygame.sprite.groupcollide(self.invaders, self.blocks, False, False)
        for inv, block_list in block_hits.items():
            inv.is_attacking = True
            block_list[0].take_damage(0.5) # Invaders damage blocks slowly
            if block_list[0].health <= 0:
                # sfx: block break
                block_list[0].kill()
                self._create_particles(block_list[0].rect.center, self.COLOR_BLOCK, 10)

        # Invaders vs Fortress
        fortress_hits = pygame.sprite.spritecollide(self.fortress, self.invaders, True)
        for inv in fortress_hits:
            # sfx: fortress damage
            damage = self.INVADER_DAMAGE
            self.fortress.take_damage(damage)
            self.score -= damage
            reward -= damage
            self._create_particles(self.fortress.rect.center, self.COLOR_FORTRESS, 20, large=True)

        # Cleanup off-screen projectiles
        for proj in self.projectiles:
            if not self.screen.get_rect().colliderect(proj.rect):
                proj.kill()
        
        return reward

    def _create_particles(self, pos, color, count, large=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) if not large else self.np_random.uniform(2, 6)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append(Particle(pos, vel, lifetime, color))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Aiming reticle
        aim_start = pygame.Vector2(self.fortress.rect.center)
        aim_end = aim_start + self.aim_direction * 30
        pygame.draw.line(self.screen, (255,255,255,50), aim_start, aim_end, 2)
        pygame.draw.circle(self.screen, (255,255,255,50), aim_end, 5, 1)

        self.projectiles.draw(self.screen)
        self.blocks.draw(self.screen)
        self.fortress.draw(self.screen)
        
        for invader in self.invaders:
            invader.draw(self.screen)
        
        for particle in self.particles:
            particle.draw(self.screen)
    
    def _render_ui(self):
        # Score
        score_surf = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Wave
        wave_surf = self.font_small.render(f"WAVE: {self.current_wave_index + 1}/{len(self.WAVE_DEFINITIONS)}", True, self.COLOR_TEXT)
        self.screen.blit(wave_surf, (self.WIDTH - wave_surf.get_width() - 10, 10))

        # Fortress Health
        health_text_surf = self.font_small.render("FORTRESS HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text_surf, (10, self.HEIGHT - 30))
        health_bar_rect = pygame.Rect(10, self.HEIGHT - 18, 200, 10)
        health_ratio = max(0, self.fortress.health / self.fortress.max_health)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, health_bar_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (health_bar_rect.x, health_bar_rect.y, health_bar_rect.width * health_ratio, health_bar_rect.height))

        # Game Over / Win / Intermission Text
        if self.game_over and not self.win:
            text_surf = self.font_large.render("GAME OVER", True, self.COLOR_INVADER)
            self.screen.blit(text_surf, text_surf.get_rect(center=self.screen.get_rect().center))
        elif self.win:
            text_surf = self.font_large.render("YOU WIN!", True, self.COLOR_FORTRESS)
            self.screen.blit(text_surf, text_surf.get_rect(center=self.screen.get_rect().center))
        elif self.wave_state == "intermission":
            seconds_left = math.ceil(self.wave_timer / self.FPS)
            text_surf = self.font_large.render(f"WAVE {self.current_wave_index + 1} IN {seconds_left}", True, self.COLOR_TEXT)
            self.screen.blit(text_surf, text_surf.get_rect(center=self.screen.get_rect().center))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave_index + 1,
            "fortress_health": self.fortress.health,
            "invaders_remaining": len(self.invaders),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

# --- Helper Classes ---

class PlayerFortress(pygame.sprite.Sprite):
    def __init__(self, x, y, max_health):
        super().__init__()
        self.pos = pygame.Vector2(x, y)
        self.size = 40
        self.image = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        self.rect = self.image.get_rect(center=(x, y))
        self.max_health = max_health
        self.health = max_health
        self.damage_flash = 0

    def take_damage(self, amount):
        self.health -= amount
        self.damage_flash = 10 # frames

    def draw(self, surface):
        color = GameEnv.COLOR_FORTRESS
        if self.damage_flash > 0:
            self.damage_flash -= 1
            color = (255, 255, 255)
        
        # Glow effect
        glow_size = self.size * 1.5
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color, 60), (glow_size/2, glow_size/2), glow_size/2)
        surface.blit(glow_surf, glow_surf.get_rect(center=self.rect.center))
        
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (255,255,255), self.rect, 1)

class Block(pygame.sprite.Sprite):
    def __init__(self, pos, max_health):
        super().__init__()
        self.size = 20
        self.image = pygame.Surface((self.size, self.size))
        self.image.fill(GameEnv.COLOR_BLOCK)
        self.rect = self.image.get_rect(center=pos)
        self.max_health = max_health
        self.health = max_health

    def take_damage(self, amount):
        self.health -= amount

    def draw(self, surface):
        alpha = int(255 * (self.health / self.max_health))
        self.image.set_alpha(alpha)
        surface.blit(self.image, self.rect)

class Invader(pygame.sprite.Sprite):
    def __init__(self, pos, speed, max_health):
        super().__init__()
        self.size = 15
        self.image = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        self.rect = self.image.get_rect(center=pos)
        self.pos = pygame.Vector2(pos)
        self.speed = speed
        self.max_health = max_health
        self.health = max_health
        self.damage_flash = 0
        self.is_attacking = False

    def update(self, fortress, blocks):
        self.is_attacking = False
        target_pos = fortress.rect.center
        direction = (pygame.Vector2(target_pos) - self.pos).normalize()
        self.pos += direction * self.speed
        self.rect.center = self.pos

    def take_damage(self, amount):
        self.health -= amount
        self.damage_flash = 5

    def draw(self, surface):
        color = GameEnv.COLOR_INVADER
        if self.damage_flash > 0:
            self.damage_flash -= 1
            color = (255, 255, 255)
        
        pygame.gfxdraw.aacircle(surface, int(self.rect.centerx), int(self.rect.centery), self.size // 2, color)
        pygame.gfxdraw.filled_circle(surface, int(self.rect.centerx), int(self.rect.centery), self.size // 2, color)
        
        # Health bar
        if self.health < self.max_health:
            bar_w = self.size
            bar_h = 4
            health_ratio = self.health / self.max_health
            bg_rect = pygame.Rect(self.rect.left, self.rect.top - bar_h - 2, bar_w, bar_h)
            fg_rect = pygame.Rect(self.rect.left, self.rect.top - bar_h - 2, bar_w * health_ratio, bar_h)
            pygame.draw.rect(surface, GameEnv.COLOR_HEALTH_BG, bg_rect)
            pygame.draw.rect(surface, GameEnv.COLOR_HEALTH_FG, fg_rect)

class Projectile(pygame.sprite.Sprite):
    def __init__(self, pos, direction):
        super().__init__()
        self.size = 5
        self.image = pygame.Surface((self.size, self.size))
        self.image.fill(GameEnv.COLOR_PROJECTILE)
        self.rect = self.image.get_rect(center=pos)
        self.pos = pygame.Vector2(pos)
        self.vel = direction.normalize() * 10

    def update(self):
        self.pos += self.vel
        self.rect.center = self.pos

class Particle:
    def __init__(self, pos, vel, lifetime, color):
        self.pos = pygame.Vector2(pos)
        self.vel = vel
        self.lifetime = lifetime
        self.color = color
        self.initial_lifetime = lifetime

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1
        return self.lifetime > 0

    def draw(self, surface):
        alpha = int(255 * (self.lifetime / self.initial_lifetime))
        color = (*self.color, alpha)
        pygame.draw.circle(surface, color, self.pos, 2)