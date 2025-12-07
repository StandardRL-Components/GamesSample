import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import os
import pygame


# Set Pygame to run in headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Battle through icy caverns, flipping gravity to navigate and evade foes. "
        "Fire ice shards to defeat the Frost Queen and her minions."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to fire an ice shard and shift to flip gravity."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_PLAYER = (100, 200, 255)
    COLOR_PLAYER_GLOW = (200, 230, 255)
    COLOR_ENEMY = (255, 80, 80)
    COLOR_ENEMY_GLOW = (255, 150, 150)
    COLOR_BOSS = (200, 100, 255)
    COLOR_BOSS_GLOW = (230, 180, 255)
    COLOR_ICE_SHARD = (220, 240, 255)
    COLOR_PLATFORM = (60, 80, 120)
    COLOR_PLATFORM_DECO = (80, 110, 160)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_HEALTH = (50, 220, 50)
    COLOR_UI_HEALTH_BG = (100, 0, 0)
    COLOR_UI_BOSS_HEALTH = (180, 50, 220)

    # Game Parameters
    PLAYER_SIZE = 12
    PLAYER_ACCEL = 0.8
    PLAYER_DAMPING = 0.92
    GRAVITY_STRENGTH = 0.6
    MAX_VEL = 8
    PLAYER_MAX_HEALTH = 100
    ENEMY_MAX_HEALTH = 20
    BOSS_MAX_HEALTH = 300
    SHARD_SPEED = 12
    SHARD_COOLDOWN = 10  # frames
    GRAVITY_FLIP_COOLDOWN = 60 # frames
    ENEMY_SIGHT_RADIUS = 200
    ENEMY_ATTACK_RADIUS = 150

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.player_pos = np.zeros(2, dtype=float)
        self.player_vel = np.zeros(2, dtype=float)
        self.player_health = 0
        self.enemies = []
        self.boss = {}
        self.projectiles = []
        self.particles = []
        self.platforms = []
        self.decorations = []
        self.gravity_dir = np.array([0.0, 1.0])
        self.gravity_angle_target = 180
        self.gravity_angle_current = 180
        
        self.steps = 0
        self.score = 0
        self.game_over_message = ""
        
        self.prev_space_held = 0
        self.prev_shift_held = 0
        self.gravity_flip_cooldown_timer = 0
        self.shoot_cooldown_timer = 0
        self.screen_shake = 0

    def _generate_level(self):
        self.platforms = []
        self.decorations = []
        # Floor and ceiling
        self.platforms.append(pygame.Rect(0, self.HEIGHT - 20, self.WIDTH, 20))
        self.platforms.append(pygame.Rect(0, 0, self.WIDTH, 20))
        # Side walls
        self.platforms.append(pygame.Rect(0, 0, 20, self.HEIGHT))
        self.platforms.append(pygame.Rect(self.WIDTH - 20, 0, 20, self.HEIGHT))

        # Central platforms
        for _ in range(self.np_random.integers(5, 8)):
            w = self.np_random.integers(80, 200)
            h = self.np_random.integers(15, 30)
            x = self.np_random.integers(40, self.WIDTH - 40 - w)
            y = self.np_random.integers(40, self.HEIGHT - 40 - h)
            self.platforms.append(pygame.Rect(x, y, w, h))

        # Boss platform
        self.platforms.append(pygame.Rect(self.WIDTH // 2 - 75, 60, 150, 20))
        
        # Generate decorations
        for p in self.platforms:
            for i in range(int(p.width / 20)):
                if self.np_random.random() < 0.5:
                    pos = (p.left + i * 20 + self.np_random.integers(-5, 6), p.top)
                    self.decorations.append({'pos': pos, 'base': 'top'})
                if self.np_random.random() < 0.5:
                    pos = (p.left + i * 20 + self.np_random.integers(-5, 6), p.bottom)
                    self.decorations.append({'pos': pos, 'base': 'bottom'})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_level()

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT - 50], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        self.player_health = self.PLAYER_MAX_HEALTH
        
        self.gravity_dir = np.array([0.0, 1.0])
        self.gravity_angle_target = 180
        self.gravity_angle_current = 180

        self.enemies = []
        num_enemies = self.np_random.integers(3, 6)
        for _ in range(num_enemies):
            self._spawn_enemy()

        self.boss = {
            'pos': np.array([self.WIDTH / 2, 85], dtype=float),
            'vel': np.array([0.0, 0.0], dtype=float),
            'health': self.BOSS_MAX_HEALTH,
            'max_health': self.BOSS_MAX_HEALTH,
            'cooldown': self.np_random.integers(60, 120),
            'size': 20
        }

        self.projectiles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over_message = ""
        
        self.prev_space_held = 0
        self.prev_shift_held = 0
        self.gravity_flip_cooldown_timer = 0
        self.shoot_cooldown_timer = 0
        self.screen_shake = 0
        
        return self._get_observation(), self._get_info()

    def _spawn_enemy(self):
        while True:
            pos = np.array([
                self.np_random.uniform(50, self.WIDTH - 50),
                self.np_random.uniform(50, self.HEIGHT - 50)
            ], dtype=float)
            if not any(p.collidepoint(pos) for p in self.platforms):
                break
        
        self.enemies.append({
            'pos': pos,
            'vel': np.array([0.0, 0.0], dtype=float),
            'health': self.ENEMY_MAX_HEALTH,
            'max_health': self.ENEMY_MAX_HEALTH,
            'size': 8,
            'state': 'patrol',
            'patrol_target': None
        })

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False
        truncated = False

        self.shoot_cooldown_timer = max(0, self.shoot_cooldown_timer - 1)
        self.gravity_flip_cooldown_timer = max(0, self.gravity_flip_cooldown_timer - 1)
        self.screen_shake = max(0, self.screen_shake - 1)
        
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        move_force = np.array([0.0, 0.0])
        if movement == 1: move_force[1] = -1.0
        elif movement == 2: move_force[1] = 1.0
        elif movement == 3: move_force[0] = -1.0
        elif movement == 4: move_force[0] = 1.0
        self.player_vel += move_force * self.PLAYER_ACCEL

        if shift_pressed and self.gravity_flip_cooldown_timer == 0:
            self.gravity_dir *= -1
            self.gravity_angle_target = (self.gravity_angle_target + 180) % 360
            self.gravity_flip_cooldown_timer = self.GRAVITY_FLIP_COOLDOWN
            self.screen_shake = 10
            self._create_particles(self.player_pos, 20, self.COLOR_PLAYER_GLOW, 3, 20)

        if space_pressed and self.shoot_cooldown_timer == 0:
            last_move_dir = move_force if np.any(move_force) else -self.gravity_dir
            proj_vel = (last_move_dir / np.linalg.norm(last_move_dir)) * self.SHARD_SPEED if np.linalg.norm(last_move_dir) > 0 else np.array([1.0, 0.0]) * self.SHARD_SPEED
            
            self.projectiles.append({'pos': self.player_pos.copy(), 'vel': proj_vel, 'owner': 'player', 'size': 4})
            self.shoot_cooldown_timer = self.SHARD_COOLDOWN
            self._create_particles(self.player_pos, 5, self.COLOR_ICE_SHARD, 2, 10, initial_vel=proj_vel*0.2)

        self._update_player()
        self._update_enemies()
        self._update_boss()
        self._update_projectiles()
        self._update_particles()
        
        new_projectiles = []
        for p in self.projectiles:
            hit = False
            if p['owner'] == 'player':
                for enemy in self.enemies:
                    if np.linalg.norm(p['pos'] - enemy['pos']) < enemy['size'] + p['size']:
                        enemy['health'] -= 10
                        reward += 0.1
                        hit = True
                        self._create_particles(p['pos'], 10, self.COLOR_ENEMY, 2, 15)
                        break
                if not hit and np.linalg.norm(p['pos'] - self.boss['pos']) < self.boss['size'] + p['size']:
                    self.boss['health'] -= 10
                    reward += 0.1
                    hit = True
                    self._create_particles(p['pos'], 15, self.COLOR_BOSS, 3, 20)
            elif p['owner'] == 'boss':
                if np.linalg.norm(p['pos'] - self.player_pos) < self.PLAYER_SIZE + p['size']:
                    self.player_health -= 15
                    reward -= 0.1
                    hit = True
                    self.screen_shake = 5
                    self._create_particles(p['pos'], 10, self.COLOR_PLAYER, 2, 15)
            
            if not hit:
                for plat in self.platforms:
                    if plat.collidepoint(p['pos']):
                        hit = True
                        self._create_particles(p['pos'], 5, self.COLOR_PLATFORM_DECO, 1, 10)
                        break
            if not hit:
                new_projectiles.append(p)
        self.projectiles = new_projectiles

        for enemy in self.enemies:
            if np.linalg.norm(self.player_pos - enemy['pos']) < self.PLAYER_SIZE + enemy['size']:
                self.player_health -= 0.5
                reward -= 0.01
                knockback_dir = self.player_pos - enemy['pos']
                if np.linalg.norm(knockback_dir) > 0:
                    knockback_dir /= np.linalg.norm(knockback_dir)
                self.player_vel += knockback_dir * 2
                enemy['vel'] -= knockback_dir * 2

        alive_enemies = []
        for enemy in self.enemies:
            if enemy['health'] > 0:
                alive_enemies.append(enemy)
            else:
                reward += 1.0
                self.score += 100
                self._create_particles(enemy['pos'], 30, self.COLOR_ENEMY_GLOW, 3, 30)
        self.enemies = alive_enemies
        
        if self.steps > 0 and self.steps % 500 == 0:
            self._spawn_enemy()
        
        self.steps += 1
        self.score += reward
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        if self.player_health <= 0:
            terminated = True
            self.game_over_message = "YOU DIED"
        elif self.boss['health'] <= 0:
            terminated = True
            reward += 50.0
            self.score += 5000
            self.game_over_message = "VICTORY!"
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over_message = "TIME'S UP"

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_entity(self, entity):
        entity['vel'] += self.gravity_dir * self.GRAVITY_STRENGTH
        entity['vel'] *= self.PLAYER_DAMPING
        vel_norm = np.linalg.norm(entity['vel'])
        if vel_norm > self.MAX_VEL:
            entity['vel'] = entity['vel'] / vel_norm * self.MAX_VEL
        entity['pos'] += entity['vel']
        self._resolve_platform_collisions(entity)

    def _update_player(self):
        self._update_entity({'pos': self.player_pos, 'vel': self.player_vel, 'size': self.PLAYER_SIZE})

    def _update_enemies(self):
        for e in self.enemies:
            dist_to_player = np.linalg.norm(e['pos'] - self.player_pos)
            e['state'] = 'attack' if dist_to_player < self.ENEMY_ATTACK_RADIUS else 'patrol'
            
            if e['state'] == 'attack':
                direction = self.player_pos - e['pos']
                if np.linalg.norm(direction) > 0:
                    direction /= np.linalg.norm(direction)
                e['vel'] += direction * (self.PLAYER_ACCEL * 0.5)
            self._update_entity(e)

    def _update_boss(self):
        if self.boss['health'] <= 0: return
        self.boss['cooldown'] = max(0, self.boss['cooldown'] - 1)
        if self.boss['cooldown'] == 0:
            direction = self.player_pos - self.boss['pos']
            if np.linalg.norm(direction) > 0:
                direction /= np.linalg.norm(direction)
            
            for i in range(-1, 2):
                angle = math.atan2(direction[1], direction[0]) + math.radians(i * 15)
                proj_vel = np.array([math.cos(angle), math.sin(angle)]) * (self.SHARD_SPEED * 0.7)
                self.projectiles.append({'pos': self.boss['pos'].copy(), 'vel': proj_vel, 'owner': 'boss', 'size': 6})
            self.boss['cooldown'] = self.np_random.integers(90, 150)
        self._update_entity(self.boss)
    
    def _update_projectiles(self):
        self.projectiles = [p for p in self.projectiles if 0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT]
        for p in self.projectiles:
            p['pos'] += p['vel']

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _resolve_platform_collisions(self, entity):
        size = entity.get('size', self.PLAYER_SIZE)
        for _ in range(2):
            entity_rect = pygame.Rect(entity['pos'][0] - size, entity['pos'][1] - size, size*2, size*2)
            for plat in self.platforms:
                if entity_rect.colliderect(plat):
                    if entity['vel'][0] > 0: entity['pos'][0], entity['vel'][0] = plat.left - size, 0
                    elif entity['vel'][0] < 0: entity['pos'][0], entity['vel'][0] = plat.right + size, 0
                    entity_rect.x = entity['pos'][0] - size
            
            for plat in self.platforms:
                if entity_rect.colliderect(plat):
                    if entity['vel'][1] > 0: entity['pos'][1], entity['vel'][1] = plat.top - size, 0
                    elif entity['vel'][1] < 0: entity['pos'][1], entity['vel'][1] = plat.bottom + size, 0
    
    def _create_particles(self, pos, count, color, size_max, life_max, initial_vel=np.array([0.0, 0.0])):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed + initial_vel
            self.particles.append({
                'pos': pos.copy() + self.np_random.uniform(-2, 2, 2),
                'vel': vel,
                'life': self.np_random.integers(life_max // 2, life_max),
                'max_life': life_max,
                'color': color,
                'size': self.np_random.integers(1, size_max) if size_max > 1 else 1,
            })

    def _get_observation(self):
        offset = (self.np_random.integers(-5, 6), self.np_random.integers(-5, 6)) if self.screen_shake > 0 else (0, 0)
        self.screen.fill(self.COLOR_BG)
        self._render_game(offset)
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self, offset):
        for p in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p.move(offset))
        
        gravity_up = self.gravity_dir[1] < 0
        for d in self.decorations:
            p1 = (d['pos'][0] + offset[0], d['pos'][1] + offset[1])
            p2, p3 = ((p1[0] - 5, p1[1] + 10 * self.gravity_dir[1]), (p1[0] + 5, p1[1] + 10 * self.gravity_dir[1])) if (d['base'] == 'bottom' and not gravity_up) or (d['base'] == 'top' and gravity_up) else ((p1[0] - 5, p1[1] - 10 * self.gravity_dir[1]), (p1[0] + 5, p1[1] - 10 * self.gravity_dir[1]))
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLATFORM_DECO)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLATFORM_DECO)

        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*p['color'], alpha), (p['size'], p['size']), p['size'])
            pos = (int(p['pos'][0] + offset[0] - p['size']), int(p['pos'][1] + offset[1] - p['size']))
            self.screen.blit(temp_surf, pos)

        for p in self.projectiles:
            pos = (int(p['pos'][0] + offset[0]), int(p['pos'][1] + offset[1]))
            color = self.COLOR_BOSS_GLOW if p['owner'] == 'boss' else self.COLOR_ICE_SHARD
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], p['size'] + 2, (*color, 100))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], p['size'] + 2, (*color, 100))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], p['size'], color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], p['size'], color)

        for e in self.enemies:
            pos = (int(e['pos'][0] + offset[0]), int(e['pos'][1] + offset[1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], e['size'] + 4, (*self.COLOR_ENEMY_GLOW, 100))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], e['size'] + 4, (*self.COLOR_ENEMY_GLOW, 100))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], e['size'], self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], e['size'], self.COLOR_ENEMY)
            self._render_health_bar(e['pos'] + offset, e['health'], e['max_health'], e['size'])

        if self.boss.get('health', 0) > 0:
            pos = (int(self.boss['pos'][0] + offset[0]), int(self.boss['pos'][1] + offset[1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.boss['size'] + 6, (*self.COLOR_BOSS_GLOW, 150))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.boss['size'] + 6, (*self.COLOR_BOSS_GLOW, 150))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.boss['size'], self.COLOR_BOSS)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.boss['size'], self.COLOR_BOSS)

        if self.player_health > 0:
            pos = (int(self.player_pos[0] + offset[0]), int(self.player_pos[1] + offset[1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_SIZE + 5, (*self.COLOR_PLAYER_GLOW, 150))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_SIZE + 5, (*self.COLOR_PLAYER_GLOW, 150))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_SIZE, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_SIZE, self.COLOR_PLAYER)
            self._render_health_bar(self.player_pos + offset, self.player_health, self.PLAYER_MAX_HEALTH, self.PLAYER_SIZE)

    def _render_health_bar(self, pos, current, maximum, entity_size):
        bar_width = 30
        bar_height = 5
        draw_pos = (pos[0] - bar_width / 2, pos[1] - entity_size - 15)
        health_ratio = max(0, current / maximum)
        bg_rect = pygame.Rect(draw_pos[0], draw_pos[1], bar_width, bar_height)
        health_rect = pygame.Rect(draw_pos[0], draw_pos[1], bar_width * health_ratio, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_BG, bg_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH, health_rect)

    def _render_ui(self):
        if self.boss.get('health', 0) > 0:
            bar_width, bar_height, draw_pos = self.WIDTH / 2, 15, (self.WIDTH / 4, 15)
            health_ratio = max(0, self.boss['health'] / self.boss['max_health'])
            bg_rect = pygame.Rect(draw_pos[0], draw_pos[1], bar_width, bar_height)
            health_rect = pygame.Rect(draw_pos[0], draw_pos[1], bar_width * health_ratio, bar_height)
            pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_BG, bg_rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_UI_BOSS_HEALTH, health_rect, border_radius=3)
            boss_text = self.font_small.render("FROST QUEEN", True, self.COLOR_UI_TEXT)
            self.screen.blit(boss_text, (self.WIDTH/2 - boss_text.get_width()/2, 16))

        indicator_pos = (self.WIDTH - 40, 40)
        angle_diff = (self.gravity_angle_target - self.gravity_angle_current + 180) % 360 - 180
        self.gravity_angle_current += angle_diff * 0.1
        rad = math.radians(self.gravity_angle_current)
        p1, p2, p3 = (indicator_pos[0], indicator_pos[1] - 15), (indicator_pos[0] - 8, indicator_pos[1] + 8), (indicator_pos[0] + 8, indicator_pos[1] + 8)
        
        def rotate(p, angle, origin):
            x, y = p[0] - origin[0], p[1] - origin[1]
            return (x * math.cos(angle) - y * math.sin(angle) + origin[0], x * math.sin(angle) + y * math.cos(angle) + origin[1])

        p1_r, p2_r, p3_r = rotate(p1, rad, indicator_pos), rotate(p2, rad, indicator_pos), rotate(p3, rad, indicator_pos)
        pygame.gfxdraw.aapolygon(self.screen, [p1_r, p2_r, p3_r], self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, [p1_r, p2_r, p3_r], self.COLOR_PLAYER_GLOW)

        if self.game_over_message:
            text = self.font_large.render(self.game_over_message, True, self.COLOR_UI_TEXT)
            self.screen.blit(text, text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "boss_health": self.boss.get('health', 0),
            "enemies_left": len(self.enemies),
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not run in headless mode
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Ice Spirit Caverns")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated, truncated = False, False

        if not terminated and not truncated:
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        if terminated or truncated:
            font = pygame.font.SysFont("monospace", 24)
            restart_text = font.render("Press 'R' to restart", True, (255, 255, 255))
            screen.blit(restart_text, (env.WIDTH/2 - restart_text.get_width()/2, env.HEIGHT/2 + 40))

        pygame.display.flip()
        clock.tick(GameEnv.FPS)

    env.close()