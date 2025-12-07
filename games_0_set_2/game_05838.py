
# Generated: 2025-08-28T06:15:33.013608
# Source Brief: brief_05838.md
# Brief Index: 5838

        
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


# --- Helper Classes ---

class Particle:
    """A simple class for visual effect particles."""
    def __init__(self, x, y, z, color, life, dx, dy, dz, gravity=0.1):
        self.x, self.y, self.z = x, y, z
        self.color = color
        self.life = life
        self.initial_life = life
        self.dx, self.dy, self.dz = dx, dy, dz
        self.gravity = gravity

    def update(self):
        self.life -= 1
        self.x += self.dx
        self.y += self.dy
        self.z += self.dz
        self.dz -= self.gravity
        if self.z < 0:
            self.z = 0
            self.dz *= -0.5 # Bounce

class Player:
    """Represents the player character."""
    def __init__(self, x, y, env):
        self.x, self.y, self.z = x, y, 0
        self.env = env
        self.health = 100
        self.max_health = 100
        self.speed = 4
        self.size = 12
        self.jump_vel = 6
        self.z_vel = 0
        self.gravity = 0.3
        self.is_jumping = False
        self.attack_cooldown = 0
        self.attack_duration = 0
        self.attack_range = 40
        self.attack_angle = 0
        self.last_move_dir = (1, 0)
        self.invulnerability_timer = 0

    def update(self, movement, do_jump, do_attack):
        # Movement
        move_x, move_y = 0, 0
        if movement == 1: move_y = -1 # Up
        elif movement == 2: move_y = 1 # Down
        elif movement == 3: move_x = -1 # Left
        elif movement == 4: move_x = 1 # Right

        if move_x != 0 or move_y != 0:
            self.last_move_dir = (move_x, move_y)
            norm = math.sqrt(move_x**2 + move_y**2)
            self.x += self.speed * move_x / norm
            self.y += self.speed * move_y / norm

        # Clamp position to arena
        self.x = max(self.env.ARENA_BOUNDS[0], min(self.x, self.env.ARENA_BOUNDS[2]))
        self.y = max(self.env.ARENA_BOUNDS[1], min(self.y, self.env.ARENA_BOUNDS[3]))

        # Jump
        if do_jump and not self.is_jumping:
            self.is_jumping = True
            self.z_vel = self.jump_vel
            # sfx: player_jump
            self.env._add_particles(self.x, self.y, 0, self.env.COLOR_WHITE, 5, count=10, power=2)

        if self.is_jumping:
            self.z += self.z_vel
            self.z_vel -= self.gravity
            if self.z <= 0:
                self.z = 0
                self.is_jumping = False
                self.z_vel = 0
                # sfx: player_land
                self.env._add_particles(self.x, self.y, 0, self.env.COLOR_WHITE, 5, count=5, power=1)

        # Attack
        if self.attack_cooldown > 0: self.attack_cooldown -= 1
        if self.attack_duration > 0: self.attack_duration -= 1

        if do_attack and self.attack_cooldown == 0:
            self.attack_cooldown = 15 # 0.5s cooldown at 30fps
            self.attack_duration = 5 # 0.16s attack time
            self.attack_angle = math.atan2(self.last_move_dir[1], self.last_move_dir[0])
            # sfx: player_attack_slash

    def take_damage(self, amount):
        if self.invulnerability_timer == 0:
            self.health -= amount
            self.invulnerability_timer = 30 # 1s invulnerability
            # sfx: player_hit
            self.env._add_particles(self.x, self.y, self.z + 15, self.env.COLOR_RED, 15, count=20, power=4)
            return True # Damage was taken
        return False

    def draw(self, screen):
        # Shadow
        shadow_pos = self.env._iso_to_screen(self.x, self.y, 0)
        shadow_size = int(self.size * (1 - min(1, self.z / 100) * 0.5))
        shadow_alpha = 100 * (1 - min(1, self.z / 100) * 0.7)
        pygame.gfxdraw.filled_ellipse(screen, shadow_pos[0], shadow_pos[1], shadow_size, shadow_size // 2, (*self.env.COLOR_BLACK, shadow_alpha))

        # Player Body
        pos = self.env._iso_to_screen(self.x, self.y, self.z)
        color = self.env.COLOR_PLAYER_ALT if self.invulnerability_timer % 4 < 2 else self.env.COLOR_PLAYER
        pygame.draw.circle(screen, color, pos, self.size)
        pygame.draw.circle(screen, self.env.COLOR_WHITE, pos, self.size, 2)

        # Attack Arc
        if self.attack_duration > 0:
            start_angle = self.attack_angle - math.pi / 4
            end_angle = self.attack_angle + math.pi / 4
            points = [pos]
            for i in range(8):
                angle = start_angle + (end_angle - start_angle) * i / 7
                px = pos[0] + self.attack_range * math.cos(angle)
                py = pos[1] + self.attack_range * math.sin(angle)
                points.append((int(px), int(py)))
            
            alpha = 255 * (self.attack_duration / 5)
            pygame.gfxdraw.aapolygon(screen, points, (*self.env.COLOR_ATTACK, alpha))
            pygame.gfxdraw.filled_polygon(screen, points, (*self.env.COLOR_ATTACK, int(alpha * 0.5)))


class Monster:
    """Represents a monster enemy."""
    def __init__(self, x, y, m_type, health, speed, env):
        self.x, self.y, self.z = x, y, 0
        self.m_type = m_type
        self.health = health
        self.max_health = health
        self.speed = speed
        self.env = env
        self.size = 10
        self.state = 'idle'
        self.state_timer = self.env.np_random.integers(30, 90)
        self.attack_cooldown = 0
        self.target_pos = None

        if self.m_type == 1: # Melee
            self.color = self.env.COLOR_MONSTER_1
            self.sight_range = 200
            self.attack_range = 30
            self.attack_damage = 10
        elif self.m_type == 2: # Ranged
            self.color = self.env.COLOR_MONSTER_2
            self.size = 12
            self.sight_range = 300
            self.attack_range = 280
            self.attack_damage = 15
        else: # Fast
            self.color = self.env.COLOR_MONSTER_3
            self.size = 8
            self.sight_range = 250
            self.attack_range = 25
            self.attack_damage = 5

    def update(self, player):
        if self.attack_cooldown > 0: self.attack_cooldown -= 1
        self.state_timer -= 1
        
        dist_to_player = math.hypot(self.x - player.x, self.y - player.y)

        # FSM
        if self.state == 'idle':
            if dist_to_player < self.sight_range:
                self.state = 'chasing'
            elif self.state_timer <= 0:
                if self.m_type == 3: # Fast monsters wander randomly
                    self.target_pos = (
                        self.env.np_random.uniform(self.env.ARENA_BOUNDS[0], self.env.ARENA_BOUNDS[2]),
                        self.env.np_random.uniform(self.env.ARENA_BOUNDS[1], self.env.ARENA_BOUNDS[3])
                    )
                    self.state = 'wandering'
                    self.state_timer = self.env.np_random.integers(60, 120)

        elif self.state == 'wandering':
            if dist_to_player < self.sight_range:
                self.state = 'chasing'
                self.target_pos = None
            elif self.state_timer <= 0 or (self.target_pos and math.hypot(self.x - self.target_pos[0], self.y - self.target_pos[1]) < 10):
                self.state = 'idle'
                self.state_timer = self.env.np_random.integers(30, 90)
            else:
                self._move_towards(self.target_pos)

        elif self.state == 'chasing':
            if dist_to_player > self.sight_range * 1.2:
                self.state = 'idle'
                self.state_timer = self.env.np_random.integers(30, 60)
            elif dist_to_player < self.attack_range and self.attack_cooldown == 0:
                self.state = 'attacking'
                self.state_timer = 10 # Attack wind-up
            else:
                self._move_towards((player.x, player.y))

        elif self.state == 'attacking':
            if self.state_timer <= 0:
                self._perform_attack(player)
                self.state = 'cooldown'
                self.state_timer = self.env.np_random.integers(45, 75)

        elif self.state == 'cooldown':
            if self.state_timer <= 0:
                self.state = 'idle'
                self.state_timer = self.env.np_random.integers(15, 30)

    def _move_towards(self, target):
        if not target: return
        dx, dy = target[0] - self.x, target[1] - self.y
        dist = math.hypot(dx, dy)
        if dist > 1:
            self.x += self.speed * dx / dist
            self.y += self.speed * dy / dist

    def _perform_attack(self, player):
        # sfx: monster_attack
        if self.m_type == 2: # Ranged
            self.env.projectiles.append(Projectile(self.x, self.y, player.x, player.y, self.attack_damage, self.env))
        else: # Melee
            self.env.monster_attacks.append({
                'x': self.x, 'y': self.y, 'range': self.attack_range, 
                'damage': self.attack_damage, 'life': 5
            })

    def take_damage(self, amount):
        self.health -= amount
        # sfx: monster_hit
        self.env._add_particles(self.x, self.y, self.z + 10, self.env.COLOR_ATTACK, 10, count=10, power=2)
        return self.health <= 0

    def draw(self, screen):
        # Shadow
        shadow_pos = self.env._iso_to_screen(self.x, self.y, 0)
        pygame.gfxdraw.filled_ellipse(screen, shadow_pos[0], shadow_pos[1], self.size, self.size // 2, (*self.env.COLOR_BLACK, 100))

        # Body
        pos = self.env._iso_to_screen(self.x, self.y, self.z)
        pygame.draw.circle(screen, self.color, pos, self.size)
        pygame.draw.circle(screen, self.env.COLOR_BLACK, pos, self.size, 2)
        
        # Health bar
        if self.health < self.max_health:
            bar_width = 20
            bar_height = 4
            health_pct = self.health / self.max_health
            bar_pos_x = pos[0] - bar_width // 2
            bar_pos_y = pos[1] - self.size - 8
            pygame.draw.rect(screen, self.env.COLOR_RED, (bar_pos_x, bar_pos_y, bar_width, bar_height))
            pygame.draw.rect(screen, self.env.COLOR_GREEN, (bar_pos_x, bar_pos_y, int(bar_width * health_pct), bar_height))

class Projectile:
    """Represents a ranged attack projectile."""
    def __init__(self, x, y, target_x, target_y, damage, env):
        self.x, self.y = x, y
        self.z = 20
        self.damage = damage
        self.env = env
        self.speed = 6
        dx, dy = target_x - x, target_y - y
        dist = math.hypot(dx, dy)
        self.dx = self.speed * dx / dist if dist > 0 else 0
        self.dy = self.speed * dy / dist if dist > 0 else 0
        self.life = 120 # 4 seconds

    def update(self):
        self.life -= 1
        self.x += self.dx
        self.y += self.dy
        return self.life <= 0 or not (self.env.ARENA_BOUNDS[0] < self.x < self.env.ARENA_BOUNDS[2] and self.env.ARENA_BOUNDS[1] < self.y < self.env.ARENA_BOUNDS[3])

    def draw(self, screen):
        pos = self.env._iso_to_screen(self.x, self.y, self.z)
        pygame.draw.circle(screen, self.env.COLOR_MONSTER_2, pos, 5)
        pygame.draw.circle(screen, self.env.COLOR_WHITE, pos, 5, 1)

# --- Main Environment Class ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    user_guide = "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    game_description = "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 16)
        self.font_l = pygame.font.SysFont("monospace", 32, bold=True)

        # Colors
        self.COLOR_BG = (30, 30, 40)
        self.COLOR_GRID = (50, 50, 60)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_ALT = (128, 255, 200)
        self.COLOR_MONSTER_1 = (255, 80, 80)
        self.COLOR_MONSTER_2 = (200, 80, 255)
        self.COLOR_MONSTER_3 = (255, 160, 0)
        self.COLOR_ATTACK = (255, 255, 100)
        self.COLOR_RED = (200, 0, 0)
        self.COLOR_GREEN = (0, 200, 0)
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_BLACK = (0, 0, 0)

        # Game constants
        self.MAX_STEPS = 1000
        self.ARENA_WIDTH, self.ARENA_HEIGHT = 500, 300
        self.ARENA_BOUNDS = [-self.ARENA_WIDTH/2, -self.ARENA_HEIGHT/2, self.ARENA_WIDTH/2, self.ARENA_HEIGHT/2]
        self.TILE_W, self.TILE_H = 32, 16
        self.ORIGIN_X, self.ORIGIN_Y = self.WIDTH // 2, self.HEIGHT // 2 - 50
        
        self.reset()
        
    def _iso_to_screen(self, x, y, z=0):
        screen_x = self.ORIGIN_X + (x - y) * 0.7
        screen_y = self.ORIGIN_Y + (x + y) * 0.35 - z
        return int(screen_x), int(screen_y)

    def _add_particles(self, x, y, z, color, life, count, power):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, power)
            dz = self.np_random.uniform(1, power) if z >= 0 else 0
            self.particles.append(Particle(x, y, z, color, life, math.cos(angle) * speed, math.sin(angle) * speed, dz))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 1
        
        self.player = Player(0, 0, self)
        self.monsters = []
        self.particles = []
        self.projectiles = []
        self.monster_attacks = []
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def _spawn_wave(self):
        self.monsters.clear()
        base_health = 20 + (self.wave_number - 1) * 2
        base_speed = 1.5 + (self.wave_number - 1) * 0.05
        
        for i in range(15):
            angle = (2 * math.pi / 15) * i
            dist = self.np_random.uniform(self.ARENA_WIDTH * 0.3, self.ARENA_WIDTH * 0.45)
            x = math.cos(angle) * dist
            y = math.sin(angle) * dist
            m_type = self.np_random.integers(1, 4)
            
            speed = base_speed
            health = base_health
            if m_type == 2: speed *= 0.5 # Ranged are slower
            if m_type == 3: speed *= 1.5 # Fast are faster
            
            self.monsters.append(Monster(x, y, m_type, health, speed, self))

    def step(self, action):
        reward = 0.1 # Survival reward

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if self.player.invulnerability_timer > 0: self.player.invulnerability_timer -= 1
        self.player.update(movement, space_held, shift_held)

        for monster in self.monsters:
            monster.update(self.player)

        self.projectiles[:] = [p for p in self.projectiles if not p.update()]
        self.monster_attacks[:] = [a for a in self.monster_attacks if a['life'] > 0]
        for attack in self.monster_attacks: attack['life'] -= 1

        # Player attack -> Monsters
        if self.player.attack_duration > 0:
            for monster in self.monsters:
                dist = math.hypot(self.player.x - monster.x, self.player.y - monster.y)
                if dist < self.player.attack_range + monster.size:
                    angle_to_monster = math.atan2(monster.y - self.player.y, monster.x - self.player.x)
                    angle_diff = abs((self.player.attack_angle - angle_to_monster + math.pi) % (2 * math.pi) - math.pi)
                    
                    if angle_diff < math.pi / 3:
                        damage = 15 if self.player.z > monster.z else 10
                        if monster.take_damage(damage):
                            reward += 2.0
                            self.score += 100
                            self._add_particles(monster.x, monster.y, 10, monster.color, 20, count=30, power=5)
                        else:
                            reward += 1.0
                            self.score += 10
                        self.player.attack_duration = 0
                        break

        # Monster attacks -> Player
        player_hit_this_frame = False
        for attack in self.monster_attacks:
            dist = math.hypot(attack['x'] - self.player.x, attack['y'] - self.player.y)
            if dist < attack['range'] and self.player.z < 5:
                if self.player.take_damage(attack['damage']):
                    player_hit_this_frame = True
            elif not player_hit_this_frame and dist < attack['range'] + 30 and self.player.z < 5:
                reward += 1.0 # Near miss reward

        for p in self.projectiles:
            dist = math.hypot(p.x - self.player.x, p.y - self.player.y)
            if dist < self.player.size + 5 and abs(p.z - self.player.z) < self.player.size:
                if self.player.take_damage(p.damage):
                    player_hit_this_frame = True
                p.life = 0
            elif not player_hit_this_frame and dist < self.player.size + 40 and abs(p.z - self.player.z) < self.player.size + 20:
                 reward += 1.0 # Near miss reward
        
        self.particles[:] = [p for p in self.particles if p.life > 0]
        for p in self.particles: p.update()

        self.monsters[:] = [m for m in self.monsters if m.health > 0]

        if not self.monsters:
            self.wave_number += 1
            reward += 50.0
            self.score += 1000
            self._spawn_wave()
            # sfx: wave_complete

        self.steps += 1
        terminated = self.player.health <= 0 or self.steps >= self.MAX_STEPS
        
        if self.player.health <= 0:
            reward = -100.0
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid
        for i in range(-10, 11):
            p1 = self._iso_to_screen(self.ARENA_BOUNDS[0], i * self.TILE_H * 2)
            p2 = self._iso_to_screen(self.ARENA_BOUNDS[2], i * self.TILE_H * 2)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
            p1 = self._iso_to_screen(i * self.TILE_W, self.ARENA_BOUNDS[1])
            p2 = self._iso_to_screen(i * self.TILE_W, self.ARENA_BOUNDS[3])
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)

        renderables = self.monsters + [self.player]
        renderables.sort(key=lambda e: e.x + e.y)

        for entity in renderables:
            entity.draw(self.screen)
        
        for p in self.projectiles:
            p.draw(self.screen)
        
        for p in self.particles:
            pos = self._iso_to_screen(p.x, p.y, p.z)
            alpha = p.color[3] if len(p.color) == 4 else 255
            final_color = (*p.color[:3], int(alpha * (p.life / p.initial_life)))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, final_color)

    def _render_ui(self):
        health_pct = max(0, self.player.health / self.player.max_health)
        pygame.draw.rect(self.screen, self.COLOR_RED, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_GREEN, (10, 10, int(200 * health_pct), 20))
        pygame.draw.rect(self.screen, self.COLOR_WHITE, (10, 10, 200, 20), 2)
        
        score_text = self.font_s.render(f"SCORE: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        wave_text = self.font_s.render(f"WAVE: {self.wave_number}", True, self.COLOR_WHITE)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 30))

        if self.game_over:
            text = self.font_l.render("GAME OVER", True, self.COLOR_RED)
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number, "health": self.player.health}

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

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    env.auto_advance = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Arena Fighter")
    clock = pygame.time.Clock()
    
    total_reward = 0

    while not done:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()