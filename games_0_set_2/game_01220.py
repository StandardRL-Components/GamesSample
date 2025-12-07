
# Generated: 2025-08-27T16:26:58.673520
# Source Brief: brief_01220.md
# Brief Index: 1220

        
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


# Helper classes for game objects
class Player:
    def __init__(self, pos, radius):
        self.pos = pygame.math.Vector2(pos)
        self.radius = radius
        self.speed = 4
        self.health = 100
        self.max_health = 100
        self.ammo = 20
        self.max_ammo = 50
        self.aim_vector = pygame.math.Vector2(0, -1)
        self.shoot_cooldown = 0
        self.damage_timer = 0
        self.color = (50, 200, 255) # Bright Cyan

class Zombie:
    def __init__(self, pos, radius):
        self.pos = pygame.math.Vector2(pos)
        self.radius = radius
        self.color = (100, 140, 80)
        self.speed = random.uniform(0.8, 1.8)
        self.health = 3
        self.damage = 10
        self.anim_offset = pygame.math.Vector2(0, 0)
        self.anim_timer = random.uniform(0, 2 * math.pi)

class Bullet:
    def __init__(self, pos, vel, radius):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.radius = radius
        self.color = (255, 255, 0)
        self.life = 60 # Expires after 2 seconds at 30fps

class AmmoBox:
    def __init__(self, pos):
        self.pos = pygame.math.Vector2(pos)
        self.size = 16
        self.color = (200, 100, 255)
        self.ammo_amount = 15
        self.rect = pygame.Rect(pos[0] - self.size/2, pos[1] - self.size/2, self.size, self.size)
        self.bob_timer = random.uniform(0, math.pi)

class Particle:
    def __init__(self, pos, vel, life, start_color, end_color, start_size, end_size=0):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.life = life
        self.max_life = life
        self.start_color = start_color
        self.end_color = end_color
        self.start_size = start_size
        self.end_size = end_size

    def update(self):
        self.pos += self.vel
        self.life -= 1
        self.vel *= 0.95

    def draw(self, surface, offset):
        if self.life > 0:
            lerp_factor = self.life / self.max_life
            current_color = (
                int(self.start_color[0] + (self.end_color[0] - self.start_color[0]) * (1 - lerp_factor)),
                int(self.start_color[1] + (self.end_color[1] - self.start_color[1]) * (1 - lerp_factor)),
                int(self.start_color[2] + (self.end_color[2] - self.start_color[2]) * (1 - lerp_factor)),
            )
            current_size = self.start_size + (self.end_size - self.start_size) * (1 - lerp_factor)
            if current_size >= 1:
                pos = self.pos + offset
                pygame.draw.circle(surface, current_color, pos, int(current_size))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move and aim. Press space to fire. Survive the horde!"
    )

    game_description = (
        "A top-down zombie survival shooter. Survive for as long as you can against ever-increasing hordes of zombies."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 3000

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        self.ui_font = pygame.font.Font(None, 24)
        self.game_over_font = pygame.font.Font(None, 64)

        # Colors
        self.COLOR_BG = (30, 35, 40)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_SHADOW = (20, 20, 25)
        self.COLOR_RED = (255, 50, 50)
        self.COLOR_GREEN = (50, 255, 50)
        self.COLOR_WHITE = (240, 240, 240)
        
        # Initialize state variables to None
        self.player = None
        self.zombies = None
        self.bullets = None
        self.ammo_boxes = None
        self.particles = None
        self.steps = 0
        self.kill_count = 0
        self.game_over = False
        self.victory = False
        self.zombie_spawn_timer = 0.0
        self.zombie_spawn_rate = 0.0
        self.ammo_spawn_timer = 0
        self.screen_shake = 0

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player = Player(pos=(self.WIDTH // 2, self.HEIGHT // 2), radius=12)
        
        self.zombies = []
        self.bullets = []
        self.ammo_boxes = []
        self.particles = []
        
        self.steps = 0
        self.kill_count = 0
        self.game_over = False
        self.victory = False
        
        self.zombie_spawn_timer = 0.0
        self.zombie_spawn_rate = 0.1 # Start with 1 zombie every 10 steps
        self.ammo_spawn_timer = 0
        self.screen_shake = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0.0
        
        if not self.game_over:
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            
            reward += self._handle_input_and_player_updates(movement, space_held)
            reward += self._update_bullets()
            reward += self._update_zombies()
            reward += self._update_ammo_boxes()
            self._update_spawners()
            self._update_particles()

        self.steps += 1
        
        terminated = self.player.health <= 0 or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.player.health <= 0:
                reward = -100.0
                self.victory = False
            else:
                reward = 100.0
                self.victory = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input_and_player_updates(self, movement, space_held):
        reward = 0.0
        
        move_vector = pygame.math.Vector2(0, 0)
        if movement == 1: move_vector.y = -1
        elif movement == 2: move_vector.y = 1
        elif movement == 3: move_vector.x = -1
        elif movement == 4: move_vector.x = 1
        
        if move_vector.length() > 0:
            move_vector.normalize_ip()
            self.player.pos += move_vector * self.player.speed
            self.player.aim_vector = move_vector.copy()

        self.player.pos.x = np.clip(self.player.pos.x, self.player.radius, self.WIDTH - self.player.radius)
        self.player.pos.y = np.clip(self.player.pos.y, self.player.radius, self.HEIGHT - self.player.radius)

        if self.player.shoot_cooldown > 0: self.player.shoot_cooldown -= 1
            
        if space_held and self.player.ammo > 0 and self.player.shoot_cooldown == 0:
            # sfx: player_shoot.wav
            self.player.ammo -= 1
            self.player.shoot_cooldown = 8
            bullet_velocity = self.player.aim_vector * 10
            new_bullet = Bullet(self.player.pos.copy(), bullet_velocity, radius=4)
            self.bullets.append(new_bullet)
            reward -= 0.01

            for _ in range(5):
                vel = self.player.aim_vector * random.uniform(2, 4) + pygame.math.Vector2(random.uniform(-2, 2), random.uniform(-2, 2))
                self.particles.append(Particle(self.player.pos + self.player.aim_vector * self.player.radius, vel, 10, (255, 255, 0), (255, 100, 0), random.randint(2, 5)))
        
        if self.player.damage_timer > 0: self.player.damage_timer -= 1
        
        return reward

    def _update_bullets(self):
        reward = 0.0
        for bullet in self.bullets[:]:
            bullet.pos += bullet.vel
            bullet.life -= 1
            
            if not (0 < bullet.pos.x < self.WIDTH and 0 < bullet.pos.y < self.HEIGHT) or bullet.life <= 0:
                if bullet in self.bullets: self.bullets.remove(bullet)
                continue

            for zombie in self.zombies[:]:
                if bullet.pos.distance_to(zombie.pos) < bullet.radius + zombie.radius:
                    # sfx: zombie_hit.wav
                    reward += 0.11 # +0.1 for hit, +0.01 to cancel miss penalty
                    zombie.health -= 1
                    
                    for _ in range(10):
                        vel = pygame.math.Vector2(random.uniform(-3, 3), random.uniform(-3, 3))
                        self.particles.append(Particle(bullet.pos, vel, 15, self.COLOR_RED, (50, 0, 0), random.randint(1, 4)))

                    if bullet in self.bullets: self.bullets.remove(bullet)

                    if zombie.health <= 0:
                        # sfx: zombie_die.wav
                        if zombie in self.zombies: self.zombies.remove(zombie)
                        reward += 1.0
                        self.kill_count += 1
                        for _ in range(30):
                            vel = pygame.math.Vector2(random.uniform(-4, 4), random.uniform(-4, 4))
                            self.particles.append(Particle(zombie.pos, vel, 20, (100, 140, 80), (40, 50, 30), random.randint(2, 5)))
                    break
        return reward

    def _update_zombies(self):
        for z in self.zombies:
            direction = (self.player.pos - z.pos)
            if direction.length() > 0:
                direction.normalize_ip()
            z.pos += direction * z.speed

            z.anim_timer += 0.2
            z.anim_offset.x = math.sin(z.anim_timer) * 2
            z.anim_offset.y = math.cos(z.anim_timer * 0.7) * 2

            if z.pos.distance_to(self.player.pos) < z.radius + self.player.radius and self.player.damage_timer == 0:
                # sfx: player_hurt.wav
                self.player.health = max(0, self.player.health - z.damage)
                self.player.damage_timer = 15
                self.screen_shake = 10
        return 0.0

    def _update_ammo_boxes(self):
        reward = 0.0
        player_rect = pygame.Rect(self.player.pos.x - self.player.radius, self.player.pos.y - self.player.radius, self.player.radius*2, self.player.radius*2)
        for box in self.ammo_boxes[:]:
            box.bob_timer += 0.1
            if player_rect.colliderect(box.rect):
                # sfx: ammo_pickup.wav
                self.player.ammo = min(self.player.max_ammo, self.player.ammo + box.ammo_amount)
                self.ammo_boxes.remove(box)
                reward += 0.5
        return reward

    def _update_spawners(self):
        if self.steps > 0 and self.steps % 10 == 0: self.zombie_spawn_rate += 0.01
        
        self.zombie_spawn_timer += self.zombie_spawn_rate
        if self.zombie_spawn_timer >= 1.0:
            num_to_spawn = int(self.zombie_spawn_timer)
            for _ in range(num_to_spawn): self._spawn_zombie()
            self.zombie_spawn_timer -= num_to_spawn

        self.ammo_spawn_timer += 1
        if self.ammo_spawn_timer > 300 and len(self.ammo_boxes) < 2:
            self._spawn_ammo_box()
            self.ammo_spawn_timer = 0

    def _spawn_zombie(self):
        edge = self.np_random.integers(0, 4)
        if edge == 0: pos = (self.np_random.integers(0, self.WIDTH), -20)
        elif edge == 1: pos = (self.np_random.integers(0, self.WIDTH), self.HEIGHT + 20)
        elif edge == 2: pos = (-20, self.np_random.integers(0, self.HEIGHT))
        else: pos = (self.WIDTH + 20, self.np_random.integers(0, self.HEIGHT))
        
        new_zombie = Zombie(pos, radius=self.np_random.integers(10, 16))
        self.zombies.append(new_zombie)

    def _spawn_ammo_box(self):
        pos = (self.np_random.integers(50, self.WIDTH - 50), self.np_random.integers(50, self.HEIGHT - 50))
        self.ammo_boxes.append(AmmoBox(pos))
        
    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.life <= 0: self.particles.remove(p)

    def _get_observation(self):
        offset = pygame.math.Vector2(0, 0)
        if self.screen_shake > 0:
            self.screen_shake -= 1
            offset.x = random.randint(-self.screen_shake, self.screen_shake)
            offset.y = random.randint(-self.screen_shake, self.screen_shake)

        self.screen.fill(self.COLOR_BG)
        self._render_background(offset)
        
        self._render_shadows(offset)
        self._render_ammo_boxes(offset)
        self._render_zombies(offset)
        self._render_player(offset)
        self._render_bullets(offset)
        self._render_particles(offset)
        
        self._render_ui()
        
        if self.game_over: self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return { "score": self.kill_count, "steps": self.steps }
        
    def _render_background(self, offset):
        for i in range(0, self.WIDTH, 50): pygame.draw.line(self.screen, self.COLOR_GRID, (i + offset.x, 0 + offset.y), (i + offset.x, self.HEIGHT + offset.y))
        for i in range(0, self.HEIGHT, 50): pygame.draw.line(self.screen, self.COLOR_GRID, (0 + offset.x, i + offset.y), (self.WIDTH + offset.x, i + offset.y))

    def _render_shadows(self, offset):
        shadow_offset = pygame.math.Vector2(3, 5)
        pygame.gfxdraw.filled_ellipse(self.screen, int(self.player.pos.x + offset.x + shadow_offset.x), int(self.player.pos.y + offset.y + shadow_offset.y), self.player.radius, self.player.radius // 2, self.COLOR_SHADOW)
        for z in self.zombies:
            pos = z.pos + z.anim_offset
            pygame.gfxdraw.filled_ellipse(self.screen, int(pos.x + offset.x + shadow_offset.x), int(pos.y + offset.y + shadow_offset.y), z.radius, z.radius // 2, self.COLOR_SHADOW)
        for box in self.ammo_boxes:
            shadow_rect = box.rect.move(offset.x + shadow_offset.x, offset.y + shadow_offset.y)
            pygame.draw.rect(self.screen, self.COLOR_SHADOW, shadow_rect, border_radius=3)
            
    def _render_ammo_boxes(self, offset):
        for box in self.ammo_boxes:
            bob = math.sin(box.bob_timer) * 3
            box_rect = box.rect.move(offset.x, offset.y - bob)
            pygame.draw.rect(self.screen, box.color, box_rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_WHITE, box_rect, width=1, border_radius=3)

    def _render_zombies(self, offset):
        for z in self.zombies:
            pos = z.pos + z.anim_offset + offset
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), z.radius, z.color)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), z.radius, z.color)
            eye_dir = (self.player.pos - z.pos).normalize() * (z.radius * 0.5) if (self.player.pos - z.pos).length() > 0 else pygame.math.Vector2(0,0)
            eye_pos = pos + eye_dir
            pygame.draw.circle(self.screen, self.COLOR_RED, eye_pos, 2)

    def _render_player(self, offset):
        pos = self.player.pos + offset
        
        if self.player.damage_timer > 0 and self.steps % 2 == 0:
            flash_color = (255, 100, 100)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), self.player.radius + 3, flash_color)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.player.radius + 3, flash_color)

        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), self.player.radius, self.player.color)
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.player.radius, self.player.color)
        
        aim_end = pos + self.player.aim_vector * (self.player.radius + 3)
        pygame.draw.line(self.screen, self.COLOR_WHITE, pos, aim_end, 3)

    def _render_bullets(self, offset):
        for b in self.bullets:
            pos = b.pos + offset
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), b.radius, b.color)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), b.radius, b.color)
            trail_pos = pos - b.vel * 0.5
            pygame.draw.line(self.screen, b.color, pos, trail_pos, b.radius * 2)

    def _render_particles(self, offset):
        for p in self.particles: p.draw(self.screen, offset)

    def _render_ui(self):
        health_pct = self.player.health / self.player.max_health
        health_bar_rect = pygame.Rect(10, 10, 150 * health_pct, 20)
        health_bar_bg_rect = pygame.Rect(10, 10, 150, 20)
        pygame.draw.rect(self.screen, self.COLOR_RED, health_bar_bg_rect)
        pygame.draw.rect(self.screen, self.COLOR_GREEN, health_bar_rect)
        pygame.draw.rect(self.screen, self.COLOR_WHITE, health_bar_bg_rect, 2)

        ammo_text = self.ui_font.render(f"AMMO: {self.player.ammo}", True, self.COLOR_WHITE)
        self.screen.blit(ammo_text, (10, 35))

        time_left = max(0, self.MAX_STEPS - self.steps) // self.FPS
        timer_text = self.ui_font.render(f"TIME: {time_left}", True, self.COLOR_WHITE)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        kill_text = self.ui_font.render(f"KILLS: {self.kill_count}", True, self.COLOR_WHITE)
        self.screen.blit(kill_text, (self.WIDTH // 2 - kill_text.get_width() // 2, self.HEIGHT - 30))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        text = "YOU DIED" if not self.victory else "YOU SURVIVED!"
        text_surf = self.game_over_font.render(text, True, self.COLOR_WHITE)
        text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        overlay.blit(text_surf, text_rect)
        self.screen.blit(overlay, (0, 0))

    def close(self):
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