import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T19:04:51.550666
# Source Brief: brief_03121.md
# Brief Index: 3121
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

def draw_aa_polygon(surface, points, color, outline_color=None, width=1):
    """Helper function to draw anti-aliased filled polygons with optional outlines."""
    pygame.gfxdraw.aapolygon(surface, points, color)
    pygame.gfxdraw.filled_polygon(surface, points, color)
    if outline_color:
        pygame.draw.polygon(surface, outline_color, points, width)

def draw_glow(surface, center, radius, color, num_layers=5, max_alpha=100):
    """Helper function to draw a soft glow effect around a point."""
    radius = int(radius)
    center = (int(center[0]), int(center[1]))
    if radius <= 0: return
    
    temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    for i in range(num_layers, 0, -1):
        alpha = int(max_alpha * (i / num_layers)**2)
        glow_color = (*color, alpha)
        layer_radius = int(radius * (i / num_layers))
        if layer_radius > 0:
            pygame.draw.circle(temp_surf, glow_color, (radius, radius), layer_radius)
    surface.blit(temp_surf, (center[0] - radius, center[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A top-down shooter where you pilot a ship, battle enemies, collect energy shards, and reshape the terrain with your weapons."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move your ship. Press space to fire and shift to cycle weapons."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000
    WIN_SCORE = 1000

    # Colors
    COLOR_BG = (10, 20, 30)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_SHARD_STD = (0, 255, 150)
    COLOR_PROJECTILE = (255, 255, 255)
    COLOR_TERRAIN = (40, 30, 60)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR = (0, 200, 100)
    COLOR_HEALTH_BAR_BG = (80, 80, 80)

    # Player
    PLAYER_SIZE = 12
    PLAYER_SPEED = 5
    PLAYER_MAX_HEALTH = 100

    # Enemy
    ENEMY_SIZE = 15
    ENEMY_BASE_HEALTH = 50
    ENEMY_BASE_FIRE_RATE = 0.02 # prob per step
    ENEMY_DIFFICULTY_INTERVAL = 200

    # Shards
    SHARD_SIZE = 6
    MAX_SHARDS = 10
    SHARD_SPAWN_INTERVAL = 50

    # --- WEAPON DEFINITIONS ---
    WEAPONS = {
        0: {"name": "Basic", "cost": 0, "damage": 10, "speed": 10, "terrain_impact": -5, "icon_points": [(-1, -1), (1, -1), (0, 1)]},
        1: {"name": "Heavy", "cost": 2, "damage": 25, "speed": 7, "terrain_impact": -15, "icon_points": [(-1, 0), (0, -1), (1, 0), (0, 1)]},
        2: {"name": "Terraformer", "cost": 1, "damage": 5, "speed": 12, "terrain_impact": 20, "icon_points": [(-1, 1), (0, -1), (1, 1)]},
        3: {"name": "Laser", "cost": 5, "damage": 15, "speed": 20, "terrain_impact": 0, "icon_points": [(-0.2, -1), (0.2, -1), (0.2, 1), (-0.2, 1)]},
    }
    WEAPON_UNLOCK_SCORES = {1: 200, 2: 500, 3: 800}

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
        self.font = pygame.font.SysFont("Consolas", 18, bold=True)

        self.player = None
        self.enemies = []
        self.shards = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.terrain_heightmap = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        player_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        self.player = {'pos': player_pos, 'health': self.PLAYER_MAX_HEALTH, 'size': self.PLAYER_SIZE, 'shards': 0, 'aim_angle': 0, 'unlocked_weapons': [0], 'current_weapon_idx': 0}

        self.enemies = []
        self._spawn_enemies(3)

        self.shards = []
        for _ in range(self.MAX_SHARDS):
            self._spawn_shard()

        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []

        self.terrain_heightmap = [self.HEIGHT - 50] * self.WIDTH

        self.last_space_held = False
        self.last_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        self._handle_input(action)
        self._update_player()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        self._update_shards()
        self._update_difficulty()
        reward += self._check_collisions()
        
        terminated = False
        truncated = False
        if self.player['health'] <= 0:
            reward -= 100
            terminated = True
            self._create_explosion(self.player['pos'], self.COLOR_PLAYER, 100)
        elif self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
        
        self.game_over = terminated or truncated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            self.player['pos'] += move_vec.normalize() * self.PLAYER_SPEED
            self.player['aim_angle'] = move_vec.as_polar()[1]

        self.player['pos'].x = np.clip(self.player['pos'].x, self.player['size'], self.WIDTH - self.player['size'])
        terrain_y = self.terrain_heightmap[int(self.player['pos'].x)]
        self.player['pos'].y = np.clip(self.player['pos'].y, self.player['size'], terrain_y - self.player['size'])

        if space_held and not self.last_space_held:
            self._fire_weapon()
        self.last_space_held = space_held

        if shift_held and not self.last_shift_held:
            self._cycle_weapon()
        self.last_shift_held = shift_held

    def _update_player(self):
        for weapon_id, unlock_score in self.WEAPON_UNLOCK_SCORES.items():
            if self.score >= unlock_score and weapon_id not in self.player['unlocked_weapons']:
                self.player['unlocked_weapons'].append(weapon_id)

    def _update_projectiles(self):
        for proj in self.player_projectiles[:]:
            proj['pos'] += proj['vel']
            proj['dist_traveled'] += proj['vel'].length()
            if proj['dist_traveled'] > self.WIDTH * 1.5 or not (0 < proj['pos'].x < self.WIDTH):
                self.player_projectiles.remove(proj)
            elif proj['pos'].y >= self.terrain_heightmap[int(proj['pos'].x)]:
                self._deform_terrain(proj['pos'].x, proj['terrain_impact'])
                self._create_explosion(proj['pos'], self.COLOR_TERRAIN, 10, 5)
                self.player_projectiles.remove(proj)

        for proj in self.enemy_projectiles[:]:
            proj['pos'] += proj['vel']
            if not self.screen.get_rect().contains(pygame.Rect(proj['pos'].x, proj['pos'].y, 1, 1)):
                self.enemy_projectiles.remove(proj)

    def _update_enemies(self):
        for enemy in self.enemies:
            if self.np_random.random() < enemy['fire_rate']:
                direction = (self.player['pos'] - enemy['pos']).normalize()
                self.enemy_projectiles.append({'pos': pygame.Vector2(enemy['pos']), 'vel': direction * 6, 'color': self.COLOR_ENEMY, 'size': 4})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _update_shards(self):
        if self.steps % self.SHARD_SPAWN_INTERVAL == 0 and len(self.shards) == 0:
            self._spawn_shard()
            
    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.ENEMY_DIFFICULTY_INTERVAL == 0:
            for enemy in self.enemies:
                enemy['health'] = max(self.ENEMY_BASE_HEALTH, enemy['health'] * 1.10)
                enemy['fire_rate'] = min(0.1, enemy['fire_rate'] + 0.01)

    def _check_collisions(self):
        reward = 0
        
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                if proj['pos'].distance_to(enemy['pos']) < self.ENEMY_SIZE:
                    enemy['health'] -= proj['damage']
                    reward += 1
                    self._create_explosion(proj['pos'], self.COLOR_ENEMY, 20)
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)

                    if enemy['health'] <= 0:
                        reward += 10
                        self.score += 50
                        self._create_explosion(enemy['pos'], self.COLOR_ENEMY, 50, 20)
                        self.enemies.remove(enemy)
                        self._spawn_enemies(1)
                    break

        for proj in self.enemy_projectiles[:]:
            if proj['pos'].distance_to(self.player['pos']) < self.player['size']:
                self.player['health'] -= 10
                self.player['health'] = max(0, self.player['health'])
                self._create_explosion(self.player['pos'], self.COLOR_PLAYER, 30, 5)
                self.enemy_projectiles.remove(proj)

        for shard in self.shards[:]:
            if self.player['pos'].distance_to(shard['pos']) < self.player['size'] + self.SHARD_SIZE:
                self.player['shards'] += 1
                self.score += 1
                reward += 0.1
                self.shards.remove(shard)
                self._spawn_shard()
        
        return reward

    def _fire_weapon(self):
        weapon_id = self.player['unlocked_weapons'][self.player['current_weapon_idx']]
        weapon = self.WEAPONS[weapon_id]
        
        if self.player['shards'] >= weapon['cost']:
            self.player['shards'] -= weapon['cost']
            direction = pygame.Vector2(); direction.from_polar((1, self.player['aim_angle']))
            start_pos = self.player['pos'] + direction * (self.player['size'] + 5)
            self.player_projectiles.append({'pos': start_pos, 'vel': direction * weapon['speed'], 'damage': weapon['damage'], 'terrain_impact': weapon['terrain_impact'], 'dist_traveled': 0})
    
    def _cycle_weapon(self):
        num_unlocked = len(self.player['unlocked_weapons'])
        if num_unlocked > 1:
            self.player['current_weapon_idx'] = (self.player['current_weapon_idx'] + 1) % num_unlocked

    def _spawn_enemies(self, num_to_spawn):
        base_health = self.ENEMY_BASE_HEALTH * (1 + (self.steps // self.ENEMY_DIFFICULTY_INTERVAL) * 0.1)
        fire_rate = self.ENEMY_BASE_FIRE_RATE + (self.steps // self.ENEMY_DIFFICULTY_INTERVAL) * 0.01

        for _ in range(num_to_spawn):
            if len(self.enemies) >= 3: return
            pos = pygame.Vector2(self.np_random.uniform(self.ENEMY_SIZE, self.WIDTH - self.ENEMY_SIZE), self.np_random.uniform(self.ENEMY_SIZE, self.HEIGHT / 2))
            self.enemies.append({'pos': pos, 'health': base_health, 'fire_rate': fire_rate})

    def _spawn_shard(self):
        pos = pygame.Vector2(self.np_random.uniform(self.SHARD_SIZE, self.WIDTH - self.SHARD_SIZE), self.np_random.uniform(self.SHARD_SIZE, self.HEIGHT - 100))
        self.shards.append({'pos': pos, 'type': 'std'})

    def _deform_terrain(self, x, amount):
        radius = 30
        x_int = int(x)
        for i in range(max(0, x_int - radius), min(self.WIDTH, x_int + radius)):
            dist = abs(i - x_int)
            effect = (math.cos(dist / radius * math.pi / 2)) * amount
            self.terrain_heightmap[i] += effect
            self.terrain_heightmap[i] = np.clip(self.terrain_heightmap[i], self.HEIGHT/2, self.HEIGHT - 10)

    def _create_explosion(self, pos, color, num_particles, speed_mult=1):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(1, 4) * speed_mult
            vel = pygame.Vector2(); vel.from_polar((speed, angle))
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'lifespan': self.np_random.integers(10, 20), 'color': color, 'size': self.np_random.integers(1, 4)})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.player['health'], "shards": self.player['shards']}

    def _render_game(self):
        terrain_points = [(i, h) for i, h in enumerate(self.terrain_heightmap)]
        terrain_points.extend([(self.WIDTH, self.HEIGHT), (0, self.HEIGHT)])
        pygame.gfxdraw.filled_polygon(self.screen, terrain_points, self.COLOR_TERRAIN)
        
        for shard in self.shards:
            pos = (int(shard['pos'].x), int(shard['pos'].y))
            draw_glow(self.screen, pos, 10, self.COLOR_SHARD_STD, max_alpha=120)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.SHARD_SIZE, self.COLOR_SHARD_STD)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.SHARD_SIZE, self.COLOR_SHARD_STD)

        for proj in self.enemy_projectiles:
            pos = (int(proj['pos'].x), int(proj['pos'].y))
            draw_glow(self.screen, pos, proj['size'] * 2, proj['color'], max_alpha=150)
            pygame.draw.circle(self.screen, proj['color'], pos, proj['size'])

        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            points = [(pos[0], pos[1] - self.ENEMY_SIZE), (pos[0] - self.ENEMY_SIZE * 0.866, pos[1] + self.ENEMY_SIZE * 0.5), (pos[0] + self.ENEMY_SIZE * 0.866, pos[1] + self.ENEMY_SIZE * 0.5)]
            draw_glow(self.screen, pos, self.ENEMY_SIZE * 1.5, self.COLOR_ENEMY, max_alpha=150)
            draw_aa_polygon(self.screen, points, self.COLOR_ENEMY)

        if self.player['health'] > 0:
            p_pos = (int(self.player['pos'].x), int(self.player['pos'].y))
            health_alpha = 50 + int(150 * (self.player['health'] / self.PLAYER_MAX_HEALTH))
            draw_glow(self.screen, p_pos, self.player['size'] * 2.5, self.COLOR_PLAYER, max_alpha=health_alpha)
            
            angle_rad = math.radians(self.player['aim_angle'])
            p1 = (p_pos[0] + self.player['size'] * math.cos(angle_rad), p_pos[1] + self.player['size'] * math.sin(angle_rad))
            p2 = (p_pos[0] + self.player['size'] * math.cos(angle_rad + 2.356), p_pos[1] + self.player['size'] * math.sin(angle_rad + 2.356))
            p3 = (p_pos[0] + self.player['size'] * math.cos(angle_rad - 2.356), p_pos[1] + self.player['size'] * math.sin(angle_rad - 2.356))
            draw_aa_polygon(self.screen, [p1,p2,p3], self.COLOR_PLAYER)

            reticle_end = self.player['pos'] + pygame.Vector2(1, 0).rotate(self.player['aim_angle']) * 40
            pygame.draw.line(self.screen, self.COLOR_PLAYER, p_pos, (int(reticle_end.x), int(reticle_end.y)), 1)
        
        for proj in self.player_projectiles:
            pos = (int(proj['pos'].x), int(proj['pos'].y))
            draw_glow(self.screen, pos, 8, self.COLOR_PROJECTILE, max_alpha=200)
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, pos, 3)

        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 20.0))
            color = (*p['color'], alpha)
            p_size = int(p['size'])
            if p_size <= 0: continue
            temp_surf = pygame.Surface((p_size*2, p_size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p_size, p_size), p_size)
            self.screen.blit(temp_surf, (int(p['pos'].x - p_size), int(p['pos'].y - p_size)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        shard_text = self.font.render(f"SHARDS: {self.player['shards']}", True, self.COLOR_SHARD_STD)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(shard_text, (10, 30))

        health_ratio = max(0, self.player['health'] / self.PLAYER_MAX_HEALTH)
        bar_width, bar_height = 200, 15
        bar_x, bar_y = (self.WIDTH - bar_width) // 2, self.HEIGHT - 30
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, int(bar_width * health_ratio), bar_height))

        weapon_id = self.player['unlocked_weapons'][self.player['current_weapon_idx']]
        weapon = self.WEAPONS[weapon_id]
        icon_pos = (self.player['pos'].x, self.player['pos'].y - self.player['size'] - 15)
        scale = 5
        scaled_points = [(icon_pos[0] + p[0]*scale, icon_pos[1] + p[1]*scale) for p in weapon['icon_points']]
        if len(scaled_points) > 1:
            pygame.draw.lines(self.screen, self.COLOR_UI_TEXT, True, scaled_points, 2)
        
        if self.game_over:
            end_font = pygame.font.SysFont("Consolas", 50, bold=True)
            msg = "VICTORY!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = (0, 255, 0) if self.score >= self.WIN_SCORE else (255, 0, 0)
            end_text = end_font.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This part is for human play and is not used by the tests
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    movement, space_held, shift_held = 0, 0, 0
    
    pygame.display.set_caption("Prismatic Brawl")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()

    print("Controls: Arrow keys to move, Space to fire, Left Shift to cycle weapon.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_LSHIFT:
                shift_held = 1
            if event.type == pygame.KEYUP and event.key == pygame.K_LSHIFT:
                shift_held = 0

        keys = pygame.key.get_pressed()
        
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
    env.close()