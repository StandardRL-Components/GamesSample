
# Generated: 2025-08-27T23:02:06.162571
# Source Brief: brief_03324.md
# Brief Index: 3324

        
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
    """
    A top-down tower defense game where the player must defend their base from
    waves of enemies by strategically placing defensive towers. The game is
    implemented as a Gymnasium environment.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrows to select a build location. Press Shift to cycle tower types. Press Space to build tower or start the next wave."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of geometric enemies by strategically placing various towers. Earn money from kills to build a stronger defense."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_PATH = (40, 40, 50)
    COLOR_BASE = (50, 200, 50)
    COLOR_ENEMY_NORMAL = (220, 50, 50)
    COLOR_ENEMY_FAST = (255, 100, 100)
    COLOR_ENEMY_TANK = (180, 40, 40)
    COLOR_TOWER_SPOT = (60, 60, 80)
    COLOR_TOWER_SPOT_SELECTED = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_WAVE_NOTICE = (255, 200, 0)
    
    # Game Parameters
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 5000
    INITIAL_BASE_HEALTH = 100
    INITIAL_MONEY = 150
    MAX_WAVES = 10

    # Tower Specs
    TOWER_SPECS = {
        'MACHINE_GUN': {'cost': 100, 'range': 80, 'damage': 5, 'fire_rate': 5, 'color': (50, 150, 255), 'unlock_wave': 1, 'projectile_speed': 10},
        'CANNON': {'cost': 250, 'range': 120, 'damage': 50, 'fire_rate': 1, 'color': (200, 150, 50), 'unlock_wave': 3, 'projectile_speed': 7},
        'SLOW_TOWER': {'cost': 150, 'range': 100, 'damage': 1, 'fire_rate': 2, 'color': (150, 50, 255), 'unlock_wave': 5, 'projectile_speed': 8},
    }

    # Enemy Specs
    ENEMY_SPECS = {
        'NORMAL': {'health': 100, 'speed': 1.0, 'damage': 5, 'value': 10, 'color': COLOR_ENEMY_NORMAL},
        'FAST': {'health': 50, 'speed': 2.0, 'damage': 2, 'value': 8, 'color': COLOR_ENEMY_FAST},
        'TANK': {'health': 500, 'speed': 0.7, 'damage': 20, 'value': 25, 'color': COLOR_ENEMY_TANK},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self._define_path_and_spots()
        self.reset()
        
        # self.validate_implementation() # Uncomment to run validation checks

    def _define_path_and_spots(self):
        self.path = [
            (-20, 80), (100, 80), (100, 250), (450, 250), (450, 150), (self.SCREEN_WIDTH + 20, 150)
        ]
        self.base_pos = (self.SCREEN_WIDTH - 40, 150)
        self.base_rect = pygame.Rect(self.base_pos[0] - 20, self.base_pos[1] - 20, 40, 40)
        
        self.tower_spots = [
            (180, 150), (300, 150), (400, 320), (150, 320), (530, 200)
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.money = self.INITIAL_MONEY
        self.current_wave = 0
        
        self.wave_state = "BETWEEN_WAVES" # "BETWEEN_WAVES", "WAVE_IN_PROGRESS"
        self.wave_timer = 0
        self.wave_enemies_to_spawn = []

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.selected_spot_idx = 0
        self.selected_tower_type_idx = 0
        self.unlocked_towers = []
        self._update_unlocked_towers()
        
        self.last_action_feedback = ""
        self.last_action_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01 # Small penalty for each step
        self.steps += 1
        
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input ---
        if self.last_action_timer > 0:
            self.last_action_timer -= 1

        if shift_pressed:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.unlocked_towers)
        
        if movement in [2, 3]: # Down or Left
            self.selected_spot_idx = (self.selected_spot_idx - 1 + len(self.tower_spots)) % len(self.tower_spots)
        elif movement in [1, 4]: # Up or Right
            self.selected_spot_idx = (self.selected_spot_idx + 1) % len(self.tower_spots)
        
        if space_pressed:
            if self.wave_state == "BETWEEN_WAVES":
                self._start_next_wave()
            else:
                reward += self._place_tower()

        # --- Game Logic ---
        if self.wave_state == "WAVE_IN_PROGRESS":
            self.wave_timer -= 1
            if self.wave_timer <= 0 and self.wave_enemies_to_spawn:
                enemy_type = self.wave_enemies_to_spawn.pop(0)
                self.enemies.append(self._create_enemy(enemy_type))
                self.wave_timer = 30

        for tower in self.towers:
            new_projectile, _ = tower.update(self.enemies)
            if new_projectile:
                self.projectiles.append(new_projectile)
                # Play sound: new_projectile.sound

        reward += self._update_projectiles()
        destroyed_rewards, damage_to_base = self._update_enemies()
        reward += destroyed_rewards
        self.base_health -= damage_to_base

        self.particles = [p for p in self.particles if p.update()]

        if self.wave_state == "WAVE_IN_PROGRESS" and not self.enemies and not self.wave_enemies_to_spawn:
            self.wave_state = "BETWEEN_WAVES"
            reward += 5
            self._update_unlocked_towers()
            if self.current_wave >= self.MAX_WAVES:
                self.game_over = True
                reward += 100

        terminated = self.base_health <= 0 or self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            reward -= 100
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            hit = proj.update()
            if hit:
                # Play sound: proj.hit_sound
                self.projectiles.remove(proj)
                if proj.target.health > 0:
                    proj.target.health -= proj.damage
                    reward += 0.1
                    self._create_hit_particles(proj.pos, proj.color)
        return reward
    
    def _update_enemies(self):
        reward = 0
        damage_to_base = 0
        for enemy in self.enemies[:]:
            reached_base = enemy.update(self.path)
            if reached_base:
                damage_to_base += enemy.damage
                self.enemies.remove(enemy)
                self._create_hit_particles(self.base_pos, self.COLOR_ENEMY_NORMAL, 20)
                # Play sound: base_hit
            elif enemy.health <= 0:
                reward += 1
                self.score += enemy.value
                self.money += enemy.value
                self.enemies.remove(enemy)
                self._create_explosion(enemy.pos, enemy.spec['color'])
                # Play sound: enemy_destroyed
        return reward, damage_to_base

    def _start_next_wave(self):
        if self.current_wave < self.MAX_WAVES:
            self.current_wave += 1
            self.wave_state = "WAVE_IN_PROGRESS"
            self.wave_enemies_to_spawn = self._generate_wave_enemies(self.current_wave)
            self.wave_timer = 0
            # Play sound: wave_start
    
    def _generate_wave_enemies(self, wave_num):
        enemies = []
        if wave_num <= 2: enemies.extend(['NORMAL'] * (wave_num * 5))
        elif wave_num <= 4:
            enemies.extend(['NORMAL'] * (wave_num * 4))
            enemies.extend(['FAST'] * (wave_num * 2))
        elif wave_num <= 7:
            enemies.extend(['NORMAL'] * (wave_num * 3))
            enemies.extend(['FAST'] * (wave_num * 3))
            enemies.extend(['TANK'] * (wave_num - 4))
        else:
            enemies.extend(['NORMAL'] * (wave_num * 2))
            enemies.extend(['FAST'] * (wave_num * 4))
            enemies.extend(['TANK'] * (wave_num - 3))
        random.shuffle(enemies)
        return enemies

    def _create_enemy(self, enemy_type):
        spec = self.ENEMY_SPECS[enemy_type]
        speed_multiplier = 1 + (self.current_wave * 0.05)
        return Enemy(spec, spec['speed'] * speed_multiplier)

    def _place_tower(self):
        spot_pos = self.tower_spots[self.selected_spot_idx]
        if any(t.pos == spot_pos for t in self.towers):
            self._set_action_feedback("Location occupied!")
            return 0
        tower_name = self.unlocked_towers[self.selected_tower_type_idx]
        spec = self.TOWER_SPECS[tower_name]
        if self.money >= spec['cost']:
            self.money -= spec['cost']
            self.towers.append(Tower(spot_pos, spec))
            self._set_action_feedback(f"Placed {tower_name}!")
            # Play sound: place_tower
            return 0
        else:
            self._set_action_feedback("Not enough money!")
            return -0.1

    def _update_unlocked_towers(self):
        self.unlocked_towers = [name for name, spec in self.TOWER_SPECS.items() if self.current_wave + 1 >= spec['unlock_wave']]
        if self.selected_tower_type_idx >= len(self.unlocked_towers): self.selected_tower_type_idx = 0

    def _set_action_feedback(self, text):
        self.last_action_feedback = text
        self.last_action_timer = 60

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 30)
        pygame.draw.rect(self.screen, self.COLOR_BASE, self.base_rect)
        pygame.gfxdraw.rectangle(self.screen, self.base_rect, (200, 255, 200))

        for i, pos in enumerate(self.tower_spots):
            color = self.COLOR_TOWER_SPOT if not any(t.pos == pos for t in self.towers) else (30,30,40)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 15, color)
        
        sel_pos = self.tower_spots[self.selected_spot_idx]
        pygame.gfxdraw.aacircle(self.screen, int(sel_pos[0]), int(sel_pos[1]), 18, self.COLOR_TOWER_SPOT_SELECTED)
        pygame.gfxdraw.aacircle(self.screen, int(sel_pos[0]), int(sel_pos[1]), 19, self.COLOR_TOWER_SPOT_SELECTED)
        
        for entity in self.towers + self.projectiles + self.enemies + self.particles:
            entity.render(self.screen)

    def _render_ui(self):
        pygame.draw.rect(self.screen, (10,10,15), (0, 0, self.SCREEN_WIDTH, 30))
        self.screen.blit(self.font_small.render(f"Base HP: {int(self.base_health)}/{self.INITIAL_BASE_HEALTH}", True, self.COLOR_UI_TEXT), (10, 7))
        self.screen.blit(self.font_small.render(f"Money: ${self.money}", True, self.COLOR_UI_TEXT), (200, 7))
        self.screen.blit(self.font_small.render(f"Wave: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT), (350, 7))
        self.screen.blit(self.font_small.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT), (500, 7))

        if self.unlocked_towers:
            tower_name = self.unlocked_towers[self.selected_tower_type_idx]
            spec = self.TOWER_SPECS[tower_name]
            self.screen.blit(self.font_small.render(f"Build: {tower_name} (${spec['cost']})", True, spec['color']), (10, self.SCREEN_HEIGHT - 25))

        if self.last_action_timer > 0:
            self.screen.blit(self.font_small.render(self.last_action_feedback, True, self.COLOR_TOWER_SPOT_SELECTED), (200, self.SCREEN_HEIGHT - 25))

        if self.wave_state == "BETWEEN_WAVES" and not self.game_over:
            text = "PRESS SPACE TO START WAVE" if self.current_wave < self.MAX_WAVES else "YOU WIN!"
            notice = self.font_large.render(text, True, self.COLOR_UI_WAVE_NOTICE)
            self.screen.blit(notice, notice.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2)))
        
        if self.game_over and self.base_health <= 0:
            notice = self.font_large.render("GAME OVER", True, self.COLOR_ENEMY_NORMAL)
            self.screen.blit(notice, notice.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2)))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave, "money": self.money, "base_health": self.base_health}
    
    def _create_explosion(self, pos, color):
        for _ in range(30): self.particles.append(Particle(pos, color, size=random.uniform(1, 4), speed=random.uniform(1, 5), duration=20))
            
    def _create_hit_particles(self, pos, color, count=5):
        for _ in range(count): self.particles.append(Particle(pos, color, size=random.uniform(1, 2), speed=random.uniform(0.5, 2), duration=10))

    def validate_implementation(self):
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2]
        obs = self._get_observation()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3) and obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3) and isinstance(info, dict)
        action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3) and isinstance(reward, (int, float)) and isinstance(term, bool) and not trunc and isinstance(info, dict)
        print("✓ Implementation validated successfully")

class Enemy:
    def __init__(self, spec, speed):
        self.spec, self.health, self.max_health, self.speed, self.damage, self.value = spec, spec['health'], spec['health'], speed, spec['damage'], spec['value']
        self.pos, self.path_idx, self.size, self.slow_effect_timer = np.array([-20.0, 80.0]), 0, 12, 0
    
    def update(self, path):
        if self.path_idx >= len(path) - 1: return True
        target_pos, direction = np.array(path[self.path_idx + 1]), np.array(path[self.path_idx + 1]) - self.pos
        dist = np.linalg.norm(direction)
        current_speed = self.speed * (0.5 if self.slow_effect_timer > 0 else 1.0)
        if self.slow_effect_timer > 0: self.slow_effect_timer -= 1
        if dist < current_speed: self.pos, self.path_idx = target_pos, self.path_idx + 1
        else: self.pos += (direction / dist) * current_speed
        return False

    def render(self, screen):
        x, y = int(self.pos[0]), int(self.pos[1])
        color = (100, 100, 255) if self.slow_effect_timer > 0 else self.spec['color']
        pygame.draw.rect(screen, color, (x - self.size/2, y - self.size/2, self.size, self.size))
        bar_x, bar_width = x - self.size*0.6, self.size * 1.2
        pygame.draw.rect(screen, (80,0,0), (bar_x, y - self.size, bar_width, 4))
        pygame.draw.rect(screen, (0,200,0), (bar_x, y - self.size, bar_width * (self.health/self.max_health), 4))

class Tower:
    def __init__(self, pos, spec):
        self.pos, self.spec, self.range, self.damage, self.fire_rate = pos, spec, spec['range'], spec['damage'], spec['fire_rate']
        self.cooldown, self.target, self.angle = 0, None, 0
    
    def update(self, enemies):
        self.cooldown = max(0, self.cooldown - 1)
        if self.target and (self.target.health <= 0 or np.linalg.norm(self.target.pos - np.array(self.pos)) > self.range): self.target = None
        if not self.target:
            in_range = [e for e in enemies if np.linalg.norm(e.pos - np.array(self.pos)) <= self.range]
            if in_range: self.target = min(in_range, key=lambda e: np.linalg.norm(e.pos - np.array(self.pos)))
        if self.target and self.cooldown == 0:
            self.cooldown = 60 / self.fire_rate
            self.angle = math.atan2(self.target.pos[1] - self.pos[1], self.target.pos[0] - self.pos[0])
            return Projectile(self.pos, self.target, self.spec), "shoot"
        return None, None

    def render(self, screen):
        x, y = int(self.pos[0]), int(self.pos[1])
        pygame.gfxdraw.filled_circle(screen, x, y, 12, (40,40,50))
        pygame.gfxdraw.aacircle(screen, x, y, 12, self.spec['color'])
        pygame.draw.line(screen, self.spec['color'], (x,y), (x + 15 * math.cos(self.angle), y + 15 * math.sin(self.angle)), 4)

class Projectile:
    def __init__(self, start_pos, target, tower_spec):
        self.pos, self.target, self.speed, self.damage = np.array(start_pos, dtype=float), target, tower_spec['projectile_speed'], tower_spec['damage']
        self.color = (255,255,100)
        self.is_slow_proj = tower_spec.get('color') == GameEnv.TOWER_SPECS['SLOW_TOWER']['color']

    def update(self):
        if self.target.health <= 0: return True
        direction, dist = self.target.pos - self.pos, np.linalg.norm(self.target.pos - self.pos)
        if dist < self.speed:
            if self.is_slow_proj: self.target.slow_effect_timer = 120
            return True
        self.pos += (direction / dist) * self.speed
        return False

    def render(self, screen):
        x, y = int(self.pos[0]), int(self.pos[1])
        pygame.gfxdraw.filled_circle(screen, x, y, 3, self.color)
        pygame.gfxdraw.aacircle(screen, x, y, 3, self.color)

class Particle:
    def __init__(self, pos, color, size, speed, duration):
        self.pos, self.color, self.size, self.duration, self.life = np.array(pos, dtype=float), color, size, duration, duration
        angle = random.uniform(0, 2 * math.pi)
        self.vel = np.array([math.cos(angle), math.sin(angle)]) * speed

    def update(self):
        self.pos += self.vel; self.vel *= 0.95; self.life -= 1
        return self.life > 0

    def render(self, screen):
        alpha = int(255 * (self.life / self.duration))
        temp_surf = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*self.color, alpha), (self.size, self.size), self.size)
        screen.blit(temp_surf, self.pos - self.size)

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    pygame.display.set_caption("Tower Defense Gym Environment")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    running = True
    while running:
        action = [0, 0, 0]
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000)
            obs, info = env.reset()
        
        env.clock.tick(30)
    pygame.quit()