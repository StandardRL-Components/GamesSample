import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set Pygame to run in a headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()
pygame.font.init()


# Helper classes for game objects
class Turret:
    def __init__(self, pos, turret_spec):
        self.pos = pos
        self.spec = turret_spec
        self.cooldown = 0
        self.angle = -math.pi / 2
        self.target = None

class Enemy:
    def __init__(self, pos, health, speed, value):
        self.pos = np.array(pos, dtype=float)
        self.health = health
        self.max_health = health
        self.speed = speed
        self.value = value
        self.path_index = 0
        self.hit_timer = 0
        self.slow_timer = 0

class Projectile:
    def __init__(self, pos, target, spec):
        self.pos = np.array(pos, dtype=float)
        self.target = target
        self.spec = spec
        self.vel = np.zeros(2, dtype=float)

class Particle:
    def __init__(self, pos, vel, radius, color, lifespan):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.radius = radius
        self.color = color
        self.lifespan = lifespan
        self.life = lifespan

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to place the selected turret. "
        "Hold Shift to cycle through turret types."
    )

    game_description = (
        "A top-down tower defense game. Place turrets to defend your base from waves of enemies. "
        "Earn resources by defeating enemies and use them to build more powerful defenses. Survive all 10 waves to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and rendering setup
        self.width, self.height = 640, 400
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PATH = (30, 30, 45)
        self.COLOR_BASE = (0, 150, 200)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_ENEMY_HIT = (255, 255, 255)
        self.COLOR_UI_TEXT = (200, 200, 220)
        self.COLOR_UI_BG = (50, 50, 70)
        self.COLOR_CURSOR_VALID = (20, 200, 20)
        self.COLOR_CURSOR_INVALID = (200, 20, 20)

        # Fonts
        self.font_s = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_m = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 32, bold=True)

        # Game constants
        self.MAX_STEPS = 30 * 180 # 3 minutes at 30fps
        self.TOTAL_WAVES = 10
        self.BASE_POS = (self.width - 40, self.height // 2)
        self.BASE_SIZE = 30
        
        self.TURRET_SPECS = {
            0: {"name": "Gatling", "cost": 50, "range": 80, "damage": 4, "fire_rate": 5, "proj_speed": 8, "color": (255, 180, 0), "splash": 0},
            1: {"name": "Cannon", "cost": 120, "range": 120, "damage": 25, "fire_rate": 25, "proj_speed": 6, "color": (0, 180, 255), "splash": 25},
        }

        # Initialize state variables
        self.path_waypoints = []
        self.turrets = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.prev_space_held = False
        self.prev_shift_held = False

    def _generate_path(self):
        self.path_waypoints = []
        margin = 50
        self.path_waypoints.append((0, margin))
        self.path_waypoints.append((self.width - margin, margin))
        self.path_waypoints.append((self.width - margin, self.height // 2))
        self.path_waypoints.append((margin, self.height // 2))
        self.path_waypoints.append((margin, self.height - margin))
        self.path_waypoints.append((self.width - 2 * margin, self.height - margin))
        self.path_waypoints.append((self.width - 2 * margin, self.BASE_POS[1]))
        self.path_waypoints.append(self.BASE_POS)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._generate_path()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_max_health = 200
        self.base_health = self.base_max_health
        self.resources = 150
        
        self.wave_number = 0
        self.wave_timer = 150 # Time before first wave
        self.enemies_in_wave = 0
        self.enemies_spawned = 0

        self.turrets = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = np.array([self.width // 4, self.height // 2], dtype=int)
        self.selected_turret_type = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.step_rewards = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.step_rewards = []
        self.game_over = self.game_won = False

        # Time-based penalty
        self.step_rewards.append(-0.001)

        self._handle_input(action)
        self._update_game_state()

        self.steps += 1
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if truncated and not terminated:
             self.step_rewards.append(-50) # Penalty for timeout
        
        reward = sum(self.step_rewards)
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Cursor Movement ---
        cursor_speed = 8
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        elif movement == 2: self.cursor_pos[1] += cursor_speed
        elif movement == 3: self.cursor_pos[0] -= cursor_speed
        elif movement == 4: self.cursor_pos[0] += cursor_speed
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.width)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.height)

        # --- Cycle Turret Type (on key press) ---
        if shift_held and not self.prev_shift_held:
            self.selected_turret_type = (self.selected_turret_type + 1) % len(self.TURRET_SPECS)

        # --- Place Turret (on key press) ---
        spec = self.TURRET_SPECS[self.selected_turret_type]
        can_place = self.resources >= spec["cost"]
        if space_held and not self.prev_space_held and can_place:
            self.turrets.append(Turret(self.cursor_pos.copy(), spec))
            self.resources -= spec["cost"]
            self.step_rewards.append(-spec["cost"] / 20) # Small penalty for spending
            self._create_particles(self.cursor_pos, 10, spec["color"])

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_game_state(self):
        self._update_waves()
        self._update_turrets()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()

    def _update_waves(self):
        if self.wave_number > self.TOTAL_WAVES:
            return

        is_wave_active = self.enemies_spawned < self.enemies_in_wave
        if not self.enemies and not is_wave_active and self.wave_number <= self.TOTAL_WAVES:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._start_next_wave()
        
        if is_wave_active and self.steps % 15 == 0: # Spawn interval
            self._spawn_enemy()

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.TOTAL_WAVES:
            return
        
        self.enemies_in_wave = 5 + self.wave_number * 5
        self.enemies_spawned = 0
        self.wave_timer = 150 # Cooldown between waves
        
    def _spawn_enemy(self):
        health = 20 * (1.1 ** self.wave_number)
        speed = 1.0 + self.wave_number * 0.05
        value = 5 + self.wave_number
        self.enemies.append(Enemy(self.path_waypoints[0], health, speed, value))
        self.enemies_spawned += 1

    def _update_turrets(self):
        for turret in self.turrets:
            if turret.cooldown > 0:
                turret.cooldown -= 1
            
            # Find target
            if turret.target and (turret.target not in self.enemies or np.linalg.norm(turret.pos - turret.target.pos) > turret.spec["range"]):
                turret.target = None

            if not turret.target:
                for enemy in self.enemies:
                    dist = np.linalg.norm(turret.pos - enemy.pos)
                    if dist <= turret.spec["range"]:
                        turret.target = enemy
                        break
            
            # Aim and shoot
            if turret.target:
                target_angle = math.atan2(turret.target.pos[1] - turret.pos[1], turret.target.pos[0] - turret.pos[0])
                # Smooth angle interpolation
                turret.angle += min( (target_angle - turret.angle + math.pi*3) % (math.pi*2) - math.pi, 0.2)

                if turret.cooldown == 0:
                    self.projectiles.append(Projectile(turret.pos, turret.target, turret.spec))
                    turret.cooldown = turret.spec["fire_rate"]
                    # Muzzle flash
                    flash_pos = turret.pos + np.array([math.cos(turret.angle), math.sin(turret.angle)]) * 12
                    self._create_particles(flash_pos, 3, (255,255,150), life=3)


    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj.target not in self.enemies:
                self.projectiles.remove(proj)
                continue

            direction = proj.target.pos - proj.pos
            dist = np.linalg.norm(direction)
            
            if dist < proj.spec["proj_speed"]:
                self._handle_projectile_hit(proj)
                self.projectiles.remove(proj)
                continue

            proj.vel = (direction / dist) * proj.spec["proj_speed"]
            proj.pos += proj.vel

    def _handle_projectile_hit(self, proj):
        hit_pos = proj.target.pos
        if proj.spec["splash"] > 0:
            # Splash damage
            self._create_particles(hit_pos, 20, proj.spec["color"])
            for enemy in self.enemies[:]:
                if np.linalg.norm(enemy.pos - hit_pos) <= proj.spec["splash"]:
                    self._damage_enemy(enemy, proj.spec["damage"])
        else:
            # Single target damage
            self._create_particles(hit_pos, 5, proj.spec["color"])
            self._damage_enemy(proj.target, proj.spec["damage"])

    def _damage_enemy(self, enemy, damage):
        enemy.health -= damage
        enemy.hit_timer = 5
        self.step_rewards.append(0.01) # Small reward for any hit
        
        if enemy.health <= 0:
            self.resources += enemy.value
            self.step_rewards.append(1.0 + enemy.value / 10.0)
            if enemy in self.enemies:
                self.enemies.remove(enemy)
            self._create_particles(enemy.pos, 15, self.COLOR_ENEMY, life=20)


    def _update_enemies(self):
        for enemy in self.enemies[:]:
            if enemy.hit_timer > 0: enemy.hit_timer -= 1

            if enemy.path_index < len(self.path_waypoints) -1:
                target_pos = self.path_waypoints[enemy.path_index + 1]
                direction = target_pos - enemy.pos
                dist = np.linalg.norm(direction)

                if dist < enemy.speed:
                    enemy.path_index += 1
                else:
                    enemy.pos += (direction / dist) * enemy.speed
            else: # Reached base
                self.base_health -= 10
                self.step_rewards.append(-10)
                self.enemies.remove(enemy)
                self._create_particles(self.BASE_POS, 30, (255,100,0), life=30)


    def _update_particles(self):
        for p in self.particles[:]:
            p.pos += p.vel
            p.life -= 1
            p.radius *= 0.95
            if p.life <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            self.step_rewards.append(-100)
            return True
        elif self.wave_number > self.TOTAL_WAVES and not self.enemies:
            self.game_over = True
            self.game_won = True
            self.step_rewards.append(100)
            return True
        
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "base_health": self.base_health,
        }

    def _render_game(self):
        self._render_path()
        self._render_base()
        for turret in self.turrets: self._render_turret(turret)
        for proj in self.projectiles: self._render_projectile(proj)
        for enemy in self.enemies: self._render_enemy(enemy)
        for p in self.particles: self._render_particle(p)
        self._render_cursor()

    def _render_path(self):
        if len(self.path_waypoints) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints, 30)

    def _render_base(self):
        base_rect = pygame.Rect(0, 0, self.BASE_SIZE, self.BASE_SIZE)
        base_rect.center = self.BASE_POS
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        pygame.draw.rect(self.screen, (150, 220, 255), base_rect, 3)

    def _render_turret(self, turret):
        pos = (int(turret.pos[0]), int(turret.pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, turret.spec["color"])
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, tuple(c//2 for c in turret.spec["color"]))
        
        # Barrel
        end_x = pos[0] + 15 * math.cos(turret.angle)
        end_y = pos[1] + 15 * math.sin(turret.angle)
        pygame.draw.line(self.screen, (100, 100, 110), pos, (int(end_x), int(end_y)), 5)

    def _render_projectile(self, proj):
        pos = (int(proj.pos[0]), int(proj.pos[1]))
        end_pos = proj.pos - proj.vel * 0.5
        end_pos_int = (int(end_pos[0]), int(end_pos[1]))
        pygame.draw.line(self.screen, proj.spec["color"], pos, end_pos_int, 2)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, proj.spec["color"])

    def _render_enemy(self, enemy):
        pos = (int(enemy.pos[0]), int(enemy.pos[1]))
        color = self.COLOR_ENEMY_HIT if enemy.hit_timer > 0 else self.COLOR_ENEMY
        
        size = 8
        rect = pygame.Rect(pos[0] - size, pos[1] - size, size * 2, size * 2)
        pygame.draw.rect(self.screen, color, rect)

        # Health bar
        if enemy.health < enemy.max_health:
            bar_w = 16
            bar_h = 3
            bar_x = pos[0] - bar_w // 2
            bar_y = pos[1] - size - 6
            health_ratio = enemy.health / enemy.max_health
            pygame.draw.rect(self.screen, (80,0,0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (bar_x, bar_y, int(bar_w * health_ratio), bar_h))

    def _render_particle(self, p):
        if p.life > 0:
            alpha = int(255 * (p.life / p.lifespan))
            color = (*p.color, alpha)
            s = pygame.Surface((int(p.radius*2), int(p.radius*2)), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (int(p.radius), int(p.radius)), int(p.radius))
            self.screen.blit(s, (int(p.pos[0] - p.radius), int(p.pos[1] - p.radius)))

    def _render_cursor(self):
        pos = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        spec = self.TURRET_SPECS[self.selected_turret_type]
        can_place = self.resources >= spec["cost"]
        color = self.COLOR_CURSOR_VALID if can_place else self.COLOR_CURSOR_INVALID

        # Range indicator
        range_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(range_surf, pos[0], pos[1], spec["range"], (*color, 30))
        pygame.gfxdraw.aacircle(range_surf, pos[0], pos[1], spec["range"], (*color, 80))
        self.screen.blit(range_surf, (0,0))

        # Cursor crosshair
        pygame.draw.line(self.screen, color, (pos[0] - 10, pos[1]), (pos[0] + 10, pos[1]), 2)
        pygame.draw.line(self.screen, color, (pos[0], pos[1] - 10), (pos[0], pos[1] + 10), 2)

    def _render_ui(self):
        # Base Health Bar
        bar_width = 200
        bar_height = 15
        health_ratio = max(0, self.base_health / self.base_max_health)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (10, 10, int(bar_width * health_ratio), bar_height))
        health_text = self.font_s.render(f"BASE: {int(self.base_health)}/{self.base_max_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 11))

        # Resources
        res_text = self.font_m.render(f"$ {self.resources}", True, (255, 220, 100))
        self.screen.blit(res_text, (220, 8))
        
        # Wave Info
        wave_str = f"WAVE {self.wave_number}/{self.TOTAL_WAVES}" if self.wave_number <= self.TOTAL_WAVES else "ALL WAVES CLEARED"
        wave_text = self.font_m.render(wave_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.width - wave_text.get_width() - 10, 8))
        
        # Selected Turret Info
        spec = self.TURRET_SPECS[self.selected_turret_type]
        turret_info_surf = pygame.Surface((180, 70), pygame.SRCALPHA)
        turret_info_surf.fill((*self.COLOR_UI_BG, 180))
        
        name_text = self.font_m.render(spec["name"], True, spec["color"])
        cost_text = self.font_s.render(f"Cost: ${spec['cost']}", True, self.COLOR_UI_TEXT)
        damage_text = self.font_s.render(f"Damage: {spec['damage']}", True, self.COLOR_UI_TEXT)
        range_text = self.font_s.render(f"Range: {spec['range']}", True, self.COLOR_UI_TEXT)

        turret_info_surf.blit(name_text, (10, 5))
        turret_info_surf.blit(cost_text, (10, 28))
        turret_info_surf.blit(damage_text, (10, 42))
        turret_info_surf.blit(range_text, (10, 56))
        self.screen.blit(turret_info_surf, (10, self.height - 80))

    def _render_game_over(self):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        msg = "VICTORY!" if self.game_won else "GAME OVER"
        color = (100, 255, 100) if self.game_won else (255, 100, 100)
        
        text = self.font_l.render(msg, True, color)
        text_rect = text.get_rect(center=(self.width / 2, self.height / 2 - 20))
        overlay.blit(text, text_rect)
        
        score_text = self.font_m.render(f"Final Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(center=(self.width / 2, self.height / 2 + 20))
        overlay.blit(score_text, score_rect)
        
        self.screen.blit(overlay, (0, 0))

    def _create_particles(self, pos, count, color, life=15, speed_range=(1, 4)):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(*speed_range)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(2, 5)
            self.particles.append(Particle(pos, vel, radius, color, life))
            
    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    done = False
    
    # Use a different screen for display to keep the environment headless
    pygame.display.init()
    display_screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Tower Defense")
    
    running = True
    while running:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # --- Display ---
        # The observation is (H, W, C), but pygame blits (W, H) surfaces
        # So we need to transpose it back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS

    env.close()