
# Generated: 2025-08-27T17:55:20.753476
# Source Brief: brief_01680.md
# Brief Index: 1680

        
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
class Tower:
    def __init__(self, pos, tower_type='basic'):
        self.pos = pygame.math.Vector2(pos)
        self.type = tower_type
        self.range = 100
        self.damage = 1
        self.fire_rate = 0.8  # shots per second
        self.cooldown = 0
        self.target = None
        self.fire_animation = 0  # for visual effect

    def update(self, dt, enemies):
        self.cooldown = max(0, self.cooldown - dt)
        self.fire_animation = max(0, self.fire_animation - dt * 5)
        
        # Find new target if needed
        if self.target is None or self.target.health <= 0 or self.pos.distance_to(self.target.pos) > self.range:
            self.target = None
            in_range_enemies = [e for e in enemies if self.pos.distance_to(e.pos) <= self.range]
            if in_range_enemies:
                self.target = min(in_range_enemies, key=lambda e: e.path_dist) # Target enemy furthest along path

    def can_fire(self):
        return self.cooldown == 0 and self.target is not None

    def fire(self):
        self.cooldown = 1 / self.fire_rate
        self.fire_animation = 1
        # SFX: Tower shoot (e.g., laser pew)
        return Projectile(self.pos, self.target, self.damage)

class Enemy:
    def __init__(self, path, speed, health):
        self.path = path
        self.path_index = 0
        self.pos = pygame.math.Vector2(path[0])
        self.speed = speed
        self.max_health = health
        self.health = health
        self.size = 12
        self.path_dist = 0 # Total distance traveled

    def update(self, dt):
        if self.path_index >= len(self.path) - 1:
            return True  # Reached the end

        target_pos = pygame.math.Vector2(self.path[self.path_index + 1])
        direction = (target_pos - self.pos).normalize()
        distance_to_target = self.pos.distance_to(target_pos)
        
        move_dist = self.speed * dt
        
        if move_dist >= distance_to_target:
            self.pos = target_pos
            self.path_index += 1
            self.path_dist += distance_to_target
        else:
            self.pos += direction * move_dist
            self.path_dist += move_dist
            
        return False

class Projectile:
    def __init__(self, start_pos, target, damage):
        self.pos = pygame.math.Vector2(start_pos)
        self.target = target
        self.damage = damage
        self.speed = 400

    def update(self, dt):
        if self.target.health <= 0:
            return True, False # Expired, no hit

        direction = (self.target.pos - self.pos).normalize()
        self.pos += direction * self.speed * dt
        
        if self.pos.distance_to(self.target.pos) < 10:
            self.target.health -= self.damage
            # SFX: Projectile hit
            return True, True # Hit target
        return False, False # In flight

class Particle:
    def __init__(self, pos, color, size, lifetime, vel):
        self.pos = pygame.math.Vector2(pos)
        self.color = color
        self.size = size
        self.lifetime = lifetime
        self.life = lifetime
        self.vel = pygame.math.Vector2(vel)

    def update(self, dt):
        self.life -= dt
        self.pos += self.vel * dt
        self.vel *= 0.95 # Damping

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.lifetime))
            s = max(0, int(self.size * (self.life / self.lifetime)))
            if s > 0:
                pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), s, (*self.color, alpha))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to select a tower spot or the 'START WAVE' button. Press space to build or start."
    )
    game_description = (
        "A minimalist tower defense game. Place towers to defend your base from waves of enemies."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Colors
        self.COLOR_BG = (44, 62, 80)
        self.COLOR_PATH = (52, 73, 94)
        self.COLOR_TOWER_SPOT = (127, 140, 141)
        self.COLOR_TOWER_SPOT_SELECTED = (241, 196, 15)
        self.COLOR_TOWER = (46, 204, 113)
        self.COLOR_ENEMY = (231, 76, 60)
        self.COLOR_PROJECTILE = (52, 152, 219)
        self.COLOR_UI_TEXT = (236, 240, 241)
        self.COLOR_GOLD = (241, 196, 15)
        self.COLOR_HEALTH_BAR_BG = (192, 57, 43)
        self.COLOR_HEALTH_BAR_FG = (39, 174, 96)
        
        self.UI_FONT_MD = pygame.font.Font(None, 24)
        self.UI_FONT_LG = pygame.font.Font(None, 48)

        # Game constants
        self.MAX_STEPS = 20000
        self.MAX_WAVES = 10
        self.INITIAL_BASE_HEALTH = 100
        self.INITIAL_GOLD = 80
        self.TOWER_COST = 30
        
        # Game path and tower spots
        self.path = [(0, 200), (100, 200), (100, 100), (540, 100), (540, 300), (self.WIDTH, 300)]
        self.tower_spots = [(100, 150), (220, 150), (380, 150), (540, 200), (320, 50)]
        
        self.start_wave_button_rect = pygame.Rect(self.WIDTH // 2 - 75, self.HEIGHT - 45, 150, 35)
        self.selectable_items = self.tower_spots + [self.start_wave_button_rect]
        
        self.state_vars_initialized = False
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.gold = self.INITIAL_GOLD
        self.current_wave = 0
        self.game_phase = "PRE_WAVE"
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.enemies_to_spawn = []
        self.spawn_timer = 0
        
        self.selected_index = 0
        self.last_action_state = {'movement': 0, 'space': False}
        
        self.state_vars_initialized = True
        return self._get_observation(), self._get_info()

    def step(self, action):
        if not self.state_vars_initialized:
            self.reset()

        dt = self.clock.tick(self.FPS) / 1000.0
        reward = 0

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        movement_pressed = movement != 0 and movement != self.last_action_state['movement']
        space_pressed = space_held and not self.last_action_state['space']
        self.last_action_state = {'movement': movement, 'space': space_held}

        if self.game_phase == "PRE_WAVE":
            if movement_pressed:
                if movement in [1, 4]:  # Up, Right
                    self.selected_index = (self.selected_index + 1) % len(self.selectable_items)
                elif movement in [2, 3]:  # Down, Left
                    self.selected_index = (self.selected_index - 1 + len(self.selectable_items)) % len(self.selectable_items)
            
            if space_pressed:
                if self.selected_index < len(self.tower_spots): # It's a tower spot
                    spot = self.tower_spots[self.selected_index]
                    is_occupied = any(t.pos.x == spot[0] and t.pos.y == spot[1] for t in self.towers)
                    if not is_occupied and self.gold >= self.TOWER_COST:
                        self.gold -= self.TOWER_COST
                        self.towers.append(Tower(spot))
                        # SFX: Build tower
                else: # It's the "start wave" button
                    self.game_phase = "WAVE_ACTIVE"
                    self.current_wave += 1
                    self._prepare_wave()
                    # SFX: Wave start horn
        
        elif self.game_phase == "WAVE_ACTIVE":
            r, h_loss = self._update_game_logic(dt)
            reward += r
            self.score += r
            if h_loss > 0:
                reward -= 5 * h_loss # Penalty for losing health
                self.base_health -= 10 * h_loss

            if not self.enemies and not self.enemies_to_spawn:
                self.game_phase = "PRE_WAVE"
                reward += 10 # Wave clear bonus
                if self.current_wave >= self.MAX_WAVES:
                    self.win = True
                    reward += 100 # Win game bonus
        
        self._update_particles(dt)
        self.steps += 1
        terminated = self._check_termination()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _prepare_wave(self):
        num_enemies = 3 + (self.current_wave - 1)
        enemy_speed = 40 + (self.current_wave - 1) * 5
        enemy_health = 3 + (self.current_wave - 1) * 1
        self.enemies_to_spawn = [
            {'speed': enemy_speed, 'health': enemy_health} for _ in range(num_enemies)
        ]
        self.spawn_timer = 0

    def _update_game_logic(self, dt):
        reward = 0
        health_lost_count = 0

        # Spawn enemies
        self.spawn_timer -= dt
        if self.spawn_timer <= 0 and self.enemies_to_spawn:
            enemy_data = self.enemies_to_spawn.pop(0)
            self.enemies.append(Enemy(self.path, enemy_data['speed'], enemy_data['health']))
            self.spawn_timer = 0.6

        # Update towers and fire projectiles
        for tower in self.towers:
            tower.update(dt, self.enemies)
            if tower.can_fire():
                self.projectiles.append(tower.fire())

        # Update projectiles
        new_projectiles = []
        for p in self.projectiles:
            expired, hit = p.update(dt)
            if hit:
                reward += 0.1
                self._create_particles(p.pos, self.COLOR_PROJECTILE, 5, 0.2, 3)
            if not expired:
                new_projectiles.append(p)
        self.projectiles = new_projectiles
        
        # Update enemies
        new_enemies = []
        for enemy in self.enemies:
            if enemy.update(dt): # Reached end
                health_lost_count += 1
                self._create_particles(self.path[-1], self.COLOR_ENEMY, 20, 1.0, 15)
                # SFX: Base damage
            elif enemy.health <= 0:
                reward += 1.0
                self.gold += 5
                self._create_particles(enemy.pos, self.COLOR_ENEMY, 15, 0.5, 10)
                # SFX: Enemy destroyed
            else:
                new_enemies.append(enemy)
        self.enemies = new_enemies

        return reward, health_lost_count

    def _update_particles(self, dt):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update(dt)

    def _create_particles(self, pos, color, count, lifetime, size):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(20, 80)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append(Particle(pos, color, random.uniform(size*0.5, size), lifetime, vel))

    def _check_termination(self):
        if self.win:
            self.game_over = True
            return True
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_path()
        self._render_tower_spots()
        for p in self.particles: p.draw(self.screen)
        for t in self.towers: self._render_tower(t)
        for e in self.enemies: self._render_enemy(e)
        for p in self.projectiles: self._render_projectile(p)
        self._render_ui()

        if self.game_over:
            self._render_end_screen()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "wave": self.current_wave,
            "base_health": self.base_health,
        }

    def _render_path(self):
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 20)

    def _render_tower_spots(self):
        for i, spot in enumerate(self.tower_spots):
            color = self.COLOR_TOWER_SPOT_SELECTED if i == self.selected_index else self.COLOR_TOWER_SPOT
            is_occupied = any(t.pos.x == spot[0] and t.pos.y == spot[1] for t in self.towers)
            if is_occupied:
                pygame.gfxdraw.filled_circle(self.screen, spot[0], spot[1], 15, (*self.COLOR_TOWER_SPOT, 50))
            else:
                pygame.gfxdraw.aacircle(self.screen, spot[0], spot[1], 15, color)
                pygame.gfxdraw.aacircle(self.screen, spot[0], spot[1], 14, color)

    def _render_tower(self, tower):
        pos = (int(tower.pos.x), int(tower.pos.y))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, self.COLOR_TOWER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 12, self.COLOR_TOWER)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.COLOR_BG)
        if tower.fire_animation > 0:
            alpha = int(100 * tower.fire_animation)
            radius = int(15 + 20 * (1 - tower.fire_animation))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*self.COLOR_TOWER, alpha))

    def _render_enemy(self, enemy):
        pos = (int(enemy.pos.x), int(enemy.pos.y))
        size = int(enemy.size)
        rect = pygame.Rect(pos[0] - size//2, pos[1] - size//2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect, border_radius=2)
        
        # Health bar
        health_pct = enemy.health / enemy.max_health
        bar_w = int(size * 1.5)
        bar_h = 4
        bar_x = pos[0] - bar_w // 2
        bar_y = pos[1] - size - 5
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=1)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (bar_x, bar_y, int(bar_w * health_pct), bar_h), border_radius=1)
        
    def _render_projectile(self, projectile):
        pos = (int(projectile.pos.x), int(projectile.pos.y))
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

    def _render_ui(self):
        # Health Bar
        health_pct = self.base_health / self.INITIAL_BASE_HEALTH
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, int(200 * health_pct), 20))
        self._draw_text(f"BASE", (110, 20), self.UI_FONT_MD, self.COLOR_UI_TEXT, centered=True)

        # Info text
        self._draw_text(f"GOLD: {self.gold}", (self.WIDTH - 10, 15), self.UI_FONT_MD, self.COLOR_GOLD, align='right')
        self._draw_text(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", (self.WIDTH - 10, 40), self.UI_FONT_MD, self.COLOR_UI_TEXT, align='right')
        self._draw_text(f"SCORE: {int(self.score)}", (self.WIDTH - 10, 65), self.UI_FONT_MD, self.COLOR_UI_TEXT, align='right')

        # Start Wave Button
        if self.game_phase == "PRE_WAVE":
            color = self.COLOR_TOWER_SPOT_SELECTED if self.selected_index == len(self.selectable_items) - 1 else self.COLOR_UI_TEXT
            pygame.draw.rect(self.screen, color, self.start_wave_button_rect, 2, border_radius=5)
            self._draw_text("START WAVE", self.start_wave_button_rect.center, self.UI_FONT_MD, color, centered=True)

    def _render_end_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        msg = "YOU WIN!" if self.win else "GAME OVER"
        color = self.COLOR_TOWER if self.win else self.COLOR_ENEMY
        self._draw_text(msg, (self.WIDTH // 2, self.HEIGHT // 2 - 20), self.UI_FONT_LG, color, centered=True)
        final_score_text = f"Final Score: {int(self.score)}"
        self._draw_text(final_score_text, (self.WIDTH // 2, self.HEIGHT // 2 + 20), self.UI_FONT_MD, self.COLOR_UI_TEXT, centered=True)

    def _draw_text(self, text, pos, font, color, centered=False, align='left'):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if centered:
            text_rect.center = pos
        elif align == 'right':
            text_rect.topright = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)
    
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")