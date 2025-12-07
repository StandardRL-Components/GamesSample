
# Generated: 2025-08-28T03:58:50.744537
# Source Brief: brief_05104.md
# Brief Index: 5104

        
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
        "Controls: ←→ to rotate the turret. Hold Shift for faster rotation. Press Space to fire."
    )

    game_description = (
        "Defend your base from waves of incoming enemies by strategically rotating and firing a turret."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 3000 # Increased from 1000 to allow for 5 waves
    FPS = 30

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_TURRET = (50, 200, 50)
    COLOR_TURRET_GLOW = (150, 255, 150)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_ENEMY_BORDER = (150, 30, 30)
    COLOR_PROJECTILE = (255, 255, 100)
    COLOR_PROJECTILE_GLOW = (255, 255, 200)
    COLOR_HEALTH_GREEN = (0, 255, 0)
    COLOR_HEALTH_RED = (255, 0, 0)
    COLOR_UI_TEXT = (220, 220, 220)

    # Turret
    TURRET_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50)
    TURRET_RADIUS = 25
    TURRET_BARREL_LENGTH = 40
    TURRET_BARREL_WIDTH = 8
    TURRET_MAX_HEALTH = 100
    TURRET_ROTATION_SPEED = 0.05
    TURRET_ROTATION_BOOST = 2.0
    PROJECTILE_SPEED = 10
    PROJECTILE_RADIUS = 4
    FIRE_COOLDOWN = 5  # in steps/frames

    # Enemies
    ENEMY_SIZE = 20
    ENEMY_MAX_HEALTH = 10
    ENEMY_BASE_SPEED = 0.75
    ENEMY_DAMAGE = 10
    ENEMIES_PER_WAVE = 20
    ENEMY_SPAWN_COOLDOWN = 30 # steps between spawns in a wave

    # Waves
    MAX_WAVES = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_wave = pygame.font.SysFont("monospace", 48, bold=True)

        self.turret_health = 0
        self.turret_angle = 0.0
        self.projectiles = []
        self.enemies = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.wave_number = 0
        self.enemies_spawned_this_wave = 0
        self.last_fire_step = 0
        self.last_enemy_spawn_step = 0
        self.muzzle_flash_timer = 0
        self.wave_clear_timer = 0
        self.game_over_timer = 0
        self.game_won = False
        self.game_over = False
        self.rng = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.turret_health = self.TURRET_MAX_HEALTH
        self.turret_angle = -math.pi / 2
        self.projectiles = []
        self.enemies = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.wave_number = 0
        self.enemies_spawned_this_wave = 0
        self.last_fire_step = 0
        self.last_enemy_spawn_step = 0
        self.muzzle_flash_timer = 0
        self.wave_clear_timer = 0
        self.game_over_timer = 0
        self.game_won = False
        self.game_over = False
        
        self._start_new_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if self.game_over:
            # Allow game over/win screen to persist
            self.game_over_timer -= 1
            if self.game_over_timer <= 0:
                self.game_over = True # True termination
            
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self._handle_input(movement, space_held, shift_held)
        
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        self._update_spawner()

        collision_reward = self._handle_collisions()
        reward += collision_reward

        wave_reward, win_reward = self._check_wave_completion()
        reward += wave_reward + win_reward
        
        if win_reward > 0:
            self.game_won = True

        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_won:
            reward -= 100 # Loss penalty

        if terminated and not self.game_over:
            self.game_over = True
            self.game_over_timer = self.FPS * 2 # 2 second display

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, movement, space_held, shift_held):
        rotation_speed = self.TURRET_ROTATION_SPEED
        if shift_held:
            rotation_speed *= self.TURRET_ROTATION_BOOST

        if movement == 3:  # Left
            self.turret_angle -= rotation_speed
        elif movement == 4:  # Right
            self.turret_angle += rotation_speed

        if space_held and (self.steps - self.last_fire_step) >= self.FIRE_COOLDOWN:
            self.last_fire_step = self.steps
            self.muzzle_flash_timer = 2 # frames
            
            start_pos_x = self.TURRET_POS[0] + self.TURRET_BARREL_LENGTH * math.cos(self.turret_angle)
            start_pos_y = self.TURRET_POS[1] + self.TURRET_BARREL_LENGTH * math.sin(self.turret_angle)
            
            velocity_x = self.PROJECTILE_SPEED * math.cos(self.turret_angle)
            velocity_y = self.PROJECTILE_SPEED * math.sin(self.turret_angle)
            
            self.projectiles.append({
                "pos": [start_pos_x, start_pos_y],
                "vel": [velocity_x, velocity_y]
            })
            # sfx: player_shoot.wav

    def _update_projectiles(self):
        for p in self.projectiles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
        
        # Remove off-screen projectiles
        self.projectiles = [p for p in self.projectiles if 0 <= p["pos"][0] < self.SCREEN_WIDTH and 0 <= p["pos"][1] < self.SCREEN_HEIGHT]

    def _update_enemies(self):
        enemy_speed = self.ENEMY_BASE_SPEED + (self.wave_number - 1) * 0.1
        for e in self.enemies:
            angle_to_turret = math.atan2(self.TURRET_POS[1] - e["pos"][1], self.TURRET_POS[0] - e["pos"][0])
            e["pos"][0] += enemy_speed * math.cos(angle_to_turret)
            e["pos"][1] += enemy_speed * math.sin(angle_to_turret)

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _update_spawner(self):
        if self.wave_clear_timer > 0:
            self.wave_clear_timer -= 1
            if self.wave_clear_timer == 0:
                self._start_new_wave()
            return
            
        if self.enemies_spawned_this_wave < self.ENEMIES_PER_WAVE and \
           (self.steps - self.last_enemy_spawn_step) >= self.ENEMY_SPAWN_COOLDOWN:
            self._spawn_enemy()
            self.last_enemy_spawn_step = self.steps
            self.enemies_spawned_this_wave += 1

    def _spawn_enemy(self):
        side = self.rng.integers(4)
        if side == 0: # Top
            pos = [self.rng.uniform(0, self.SCREEN_WIDTH), -self.ENEMY_SIZE]
        elif side == 1: # Right
            pos = [self.SCREEN_WIDTH + self.ENEMY_SIZE, self.rng.uniform(0, self.SCREEN_HEIGHT * 0.8)]
        elif side == 2: # Bottom
            pos = [self.rng.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.ENEMY_SIZE]
        else: # Left
            pos = [-self.ENEMY_SIZE, self.rng.uniform(0, self.SCREEN_HEIGHT * 0.8)]
        
        self.enemies.append({
            "pos": pos,
            "health": self.ENEMY_MAX_HEALTH,
            "max_health": self.ENEMY_MAX_HEALTH
        })

    def _handle_collisions(self):
        reward = 0
        
        # Projectile-Enemy collisions
        for p in self.projectiles[:]:
            for e in self.enemies[:]:
                dist = math.hypot(p["pos"][0] - e["pos"][0], p["pos"][1] - e["pos"][1])
                if dist < self.ENEMY_SIZE / 2 + self.PROJECTILE_RADIUS:
                    if p in self.projectiles: self.projectiles.remove(p)
                    e["health"] -= 10 # Projectile damage is 10
                    reward += 0.1 # Hit reward
                    # sfx: enemy_hit.wav
                    
                    if e["health"] <= 0:
                        self.score += 10
                        reward += 1 # Kill reward
                        self._create_explosion(e["pos"], 20, self.COLOR_ENEMY)
                        if e in self.enemies: self.enemies.remove(e)
                        # sfx: enemy_explode.wav
                    break # Projectile can only hit one enemy

        # Enemy-Turret collisions
        for e in self.enemies[:]:
            dist = math.hypot(e["pos"][0] - self.TURRET_POS[0], e["pos"][1] - self.TURRET_POS[1])
            if dist < self.TURRET_RADIUS + self.ENEMY_SIZE / 2:
                self.turret_health -= self.ENEMY_DAMAGE
                self.turret_health = max(0, self.turret_health)
                self._create_explosion(e["pos"], 10, self.COLOR_UI_TEXT)
                if e in self.enemies: self.enemies.remove(e)
                # sfx: player_damage.wav
        
        return reward

    def _check_wave_completion(self):
        wave_reward = 0
        win_reward = 0
        
        if self.wave_clear_timer == 0 and not self.enemies and self.enemies_spawned_this_wave == self.ENEMIES_PER_WAVE:
            if self.wave_number < self.MAX_WAVES:
                self.score += 100
                wave_reward = 100
                self.wave_clear_timer = self.FPS * 3 # 3 second pause
                # sfx: wave_clear.wav
            elif self.wave_number == self.MAX_WAVES:
                if not self.game_won: # Grant reward only once
                    self.score += 500
                    win_reward = 500
                    self.game_won = True
                    # sfx: game_win.wav

        return wave_reward, win_reward

    def _start_new_wave(self):
        self.wave_number += 1
        self.enemies_spawned_this_wave = 0
        self.last_enemy_spawn_step = self.steps

    def _check_termination(self):
        if self.turret_health <= 0:
            return True
        if self.game_won:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [speed * math.cos(angle), speed * math.sin(angle)],
                "lifespan": self.rng.integers(10, 25),
                "color": color,
                "radius": self.rng.uniform(1, 3)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 25))
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), color)

        # Projectiles
        for p in self.projectiles:
            px, py = int(p["pos"][0]), int(p["pos"][1])
            pygame.gfxdraw.filled_circle(self.screen, px, py, self.PROJECTILE_RADIUS + 2, (*self.COLOR_PROJECTILE_GLOW, 100))
            pygame.gfxdraw.filled_circle(self.screen, px, py, self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, px, py, self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)
            
        # Turret Base
        pygame.gfxdraw.filled_circle(self.screen, self.TURRET_POS[0], self.TURRET_POS[1], self.TURRET_RADIUS, self.COLOR_TURRET)
        pygame.gfxdraw.aacircle(self.screen, self.TURRET_POS[0], self.TURRET_POS[1], self.TURRET_RADIUS, self.COLOR_TURRET_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, self.TURRET_POS[0], self.TURRET_POS[1], self.TURRET_RADIUS-2, (*self.COLOR_TURRET_GLOW, 50))


        # Turret Barrel
        barrel_end_x = self.TURRET_POS[0] + self.TURRET_BARREL_LENGTH * math.cos(self.turret_angle)
        barrel_end_y = self.TURRET_POS[1] + self.TURRET_BARREL_LENGTH * math.sin(self.turret_angle)
        
        dx = barrel_end_x - self.TURRET_POS[0]
        dy = barrel_end_y - self.TURRET_POS[1]
        
        points = [
            (self.TURRET_POS[0] - dy * 0.15, self.TURRET_POS[1] + dx * 0.15),
            (self.TURRET_POS[0] + dy * 0.15, self.TURRET_POS[1] - dx * 0.15),
            (barrel_end_x + dy * 0.15, barrel_end_y - dx * 0.15),
            (barrel_end_x - dy * 0.15, barrel_end_y + dx * 0.15),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_TURRET_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_TURRET)
        
        # Muzzle Flash
        if self.muzzle_flash_timer > 0:
            self.muzzle_flash_timer -= 1
            flash_size = 15
            flash_points = []
            for i in range(8):
                angle = self.turret_angle + i * math.pi / 4
                radius = flash_size if i % 2 == 0 else flash_size / 2
                flash_points.append(
                    (barrel_end_x + radius * math.cos(angle), barrel_end_y + radius * math.sin(angle))
                )
            pygame.gfxdraw.filled_polygon(self.screen, flash_points, self.COLOR_PROJECTILE_GLOW)

        # Enemies
        for e in self.enemies:
            ex, ey = int(e["pos"][0]), int(e["pos"][1])
            size = self.ENEMY_SIZE
            rect = pygame.Rect(ex - size / 2, ey - size / 2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_BORDER, rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect.inflate(-4, -4), border_radius=3)
            
            # Enemy Health Bar
            self._render_health_bar(self.screen, (ex, ey - size), (size, 4), e["health"], e["max_health"])

    def _render_ui(self):
        # Health Bar
        health_bar_width = 200
        health_bar_pos = (self.SCREEN_WIDTH // 2 - health_bar_width // 2, self.SCREEN_HEIGHT - 30)
        self._render_health_bar(self.screen, health_bar_pos, (health_bar_width, 15), self.turret_health, self.TURRET_MAX_HEALTH, True)

        # Score and Wave Text
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, self.SCREEN_HEIGHT - 35))
        
        wave_text_str = f"WAVE: {self.wave_number}/{self.MAX_WAVES}"
        wave_text = self.font_ui.render(wave_text_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, self.SCREEN_HEIGHT - 35))

        # Wave Clear / Game Over Text
        if self.wave_clear_timer > 0:
            text = self.font_wave.render(f"WAVE {self.wave_number} CLEARED", True, self.COLOR_TURRET_GLOW)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 50))
            self.screen.blit(text, text_rect)
        elif self.game_over:
            if self.game_won:
                text = self.font_wave.render("VICTORY!", True, self.COLOR_TURRET_GLOW)
            else:
                text = self.font_wave.render("GAME OVER", True, self.COLOR_ENEMY)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 50))
            self.screen.blit(text, text_rect)

    def _render_health_bar(self, surface, pos, size, current_hp, max_hp, centered=False):
        x, y = pos
        w, h = size
        if centered:
            x -= w / 2
        
        ratio = max(0, min(1, current_hp / max_hp))
        
        bg_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surface, (80, 80, 80), bg_rect, border_radius=2)
        
        # Interpolate color from green to red
        color = (
            int(self.COLOR_HEALTH_RED[0] * (1 - ratio) + self.COLOR_HEALTH_GREEN[0] * ratio),
            int(self.COLOR_HEALTH_RED[1] * (1 - ratio) + self.COLOR_HEALTH_GREEN[1] * ratio),
            int(self.COLOR_HEALTH_RED[2] * (1 - ratio) + self.COLOR_HEALTH_GREEN[2] * ratio)
        )
        
        fg_rect = pygame.Rect(x, y, w * ratio, h)
        pygame.draw.rect(surface, color, fg_rect, border_radius=2)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "turret_health": self.turret_health
        }

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")