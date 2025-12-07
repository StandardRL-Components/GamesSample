import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:18:19.447527
# Source Brief: brief_00327.md
# Brief Index: 327
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Pilot a ship and its clones against waves of enemies and powerful bosses, "
        "using a central black hole to your advantage."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to shoot at the nearest enemy. "
        "Hold shift to magnetize enemies towards the black hole."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_CLONE = (0, 180, 120)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_BOSS = (255, 100, 100)
    COLOR_PLAYER_PROJ = (255, 255, 0)
    COLOR_ENEMY_PROJ = (255, 150, 0)
    COLOR_MAGNET_BEAM = (100, 100, 255)
    COLOR_BLACK_HOLE = (20, 0, 40)
    COLOR_ACCRETION_DISK = (150, 50, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH = (50, 200, 50)
    COLOR_AMMO = (50, 150, 200)

    # Player
    PLAYER_SPEED = 5
    PLAYER_SIZE = 12
    PLAYER_HEALTH_START = 100
    PLAYER_AMMO_START = 50
    PLAYER_SHOOT_COOLDOWN = 5  # steps

    # Cloning
    CLONE_INTERVAL = 200

    # Enemies
    ENEMY_SIZE = 10
    ENEMY_SPEED_START = 1.0
    ENEMY_HEALTH_START = 20
    ENEMY_SHOOT_COOLDOWN = 60
    ENEMY_SPAWN_INTERVAL_START = 100
    ENEMIES_TO_BOSS = 10

    # Boss
    BOSS_SIZE = 30
    BOSS_HEALTH_START = 300
    BOSS_SPEED = 1.5

    # Projectiles
    PROJ_SPEED = 8
    PROJ_SIZE = 3
    PROJ_DAMAGE = 10

    # Black Hole
    BH_POS = pygame.Vector2(WIDTH // 2, HEIGHT // 2)
    BH_RADIUS = 25
    BH_PULL_STRENGTH = 0.005

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # State variables are initialized in reset()
        self.players = []
        self.enemies = []
        self.boss = None
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.stars = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0

        # Player state
        self.players = [self._create_player(is_clone=False)]

        # Enemy state
        self.enemies = []
        self.enemy_spawn_timer = self.ENEMY_SPAWN_INTERVAL_START
        self.enemy_spawn_interval = self.ENEMY_SPAWN_INTERVAL_START
        self.enemy_speed = self.ENEMY_SPEED_START
        self.enemies_defeated_count = 0
        
        # Boss state
        self.boss = None
        self.boss_level = 1

        # Projectiles and effects
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []

        # Background
        self.stars = [
            (
                random.randint(0, self.WIDTH),
                random.randint(0, self.HEIGHT),
                random.choice([(100, 100, 100), (150, 150, 150), (200, 200, 200)]),
                random.randint(1, 2)
            )
            for _ in range(150)
        ]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._update_difficulty()
        self._handle_player_actions(movement, space_held, shift_held)
        self._update_enemies()
        self._update_boss()
        self._update_projectiles()
        self._handle_collisions()
        self._update_black_hole_pull()
        self._update_particles()
        self._spawn_enemies()
        self._spawn_boss()

        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Entity Creation ---
    def _create_player(self, is_clone=True):
        spawn_pos = pygame.Vector2(self.WIDTH // 4, self.HEIGHT // 2)
        if is_clone and self.players:
            # Spawn near the original player
            original_pos = self.players[0]['pos']
            angle = random.uniform(0, 2 * math.pi)
            spawn_pos = original_pos + pygame.Vector2(math.cos(angle), math.sin(angle)) * 50
        
        return {
            'pos': spawn_pos,
            'health': self.PLAYER_HEALTH_START,
            'ammo': self.PLAYER_AMMO_START,
            'is_clone': is_clone,
            'shoot_cooldown': 0,
            'magnet_target': None
        }

    def _create_enemy(self):
        side = random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top': pos = pygame.Vector2(random.randint(0, self.WIDTH), -self.ENEMY_SIZE)
        elif side == 'bottom': pos = pygame.Vector2(random.randint(0, self.WIDTH), self.HEIGHT + self.ENEMY_SIZE)
        elif side == 'left': pos = pygame.Vector2(-self.ENEMY_SIZE, random.randint(0, self.HEIGHT))
        else: pos = pygame.Vector2(self.WIDTH + self.ENEMY_SIZE, random.randint(0, self.HEIGHT))
        
        return {
            'pos': pos,
            'health': self.ENEMY_HEALTH_START,
            'shoot_cooldown': random.randint(0, self.ENEMY_SHOOT_COOLDOWN),
            'patrol_offset': random.uniform(0, 2 * math.pi)
        }

    # --- Update Logic ---
    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.enemy_speed = min(3.0, self.enemy_speed + 0.05)
        if self.steps > 0 and self.steps % 100 == 0:
            self.enemy_spawn_interval = max(20, self.enemy_spawn_interval * 0.99)
        if self.steps > 0 and self.steps % self.CLONE_INTERVAL == 0 and len(self.players) < 5:
            self.players.append(self._create_player(is_clone=True))

    def _handle_player_actions(self, movement, space_held, shift_held):
        if not self.players: return
        dist_before = self._get_closest_enemy_dist(self.players[0]['pos']) if self.enemies else float('inf')

        for p in self.players:
            vel = pygame.Vector2(0, 0)
            if movement == 1: vel.y = -1
            elif movement == 2: vel.y = 1
            elif movement == 3: vel.x = -1
            elif movement == 4: vel.x = 1
            if vel.length() > 0:
                vel.scale_to_length(self.PLAYER_SPEED)
            p['pos'] += vel

            p['pos'].x = np.clip(p['pos'].x, self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
            p['pos'].y = np.clip(p['pos'].y, self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)
            
            p['shoot_cooldown'] = max(0, p['shoot_cooldown'] - 1)
            if space_held and p['ammo'] > 0 and p['shoot_cooldown'] == 0:
                self._fire_projectile(p, self.player_projectiles, is_player=True)
                p['ammo'] -= 1
                p['shoot_cooldown'] = self.PLAYER_SHOOT_COOLDOWN

            p['magnet_target'] = None
            if shift_held:
                target = self._get_closest_enemy(p['pos'])
                if target:
                    p['magnet_target'] = target
                    if (self.BH_POS - target['pos']).length_squared() > 0:
                        pull_vec = (self.BH_POS - target['pos']).normalize() * 0.5
                        target['pos'] += pull_vec
        
        if self.players:
            dist_after = self._get_closest_enemy_dist(self.players[0]['pos']) if self.enemies else float('inf')
            if dist_after < dist_before:
                self.reward_this_step += 0.01

    def _fire_projectile(self, owner, projectile_list, is_player):
        pos = pygame.math.Vector2(owner['pos'])
        
        target_obj = None
        if is_player:
            target_obj = self._get_closest_enemy(pos) or self.boss
        else: # Enemy shooting
            if self.players:
                target_obj = self.players[0]

        if not target_obj: return
        
        target_pos = target_obj['pos']
        if (target_pos - pos).length_squared() > 0:
            direction = (target_pos - pos).normalize()
            projectile_list.append({'pos': pos, 'vel': direction * self.PROJ_SPEED})

    def _update_enemies(self):
        for e in self.enemies:
            e['patrol_offset'] += 0.05
            patrol_vec = pygame.Vector2(math.sin(e['patrol_offset']), math.cos(e['patrol_offset'] * 0.5)) * 0.5
            
            track_vec = pygame.Vector2(0,0)
            if self.players:
                closest_player = min(self.players, key=lambda p: p['pos'].distance_to(e['pos']))
                if (closest_player['pos'] - e['pos']).length_squared() > 0:
                    track_vec = (closest_player['pos'] - e['pos']).normalize()

            if (track_vec + patrol_vec).length_squared() > 0:
                e['pos'] += (track_vec + patrol_vec).normalize() * self.enemy_speed
            
            e['shoot_cooldown'] = max(0, e['shoot_cooldown'] - 1)
            if e['shoot_cooldown'] == 0 and self.players:
                self._fire_projectile(e, self.enemy_projectiles, is_player=False)
                e['shoot_cooldown'] = self.ENEMY_SHOOT_COOLDOWN

    def _update_boss(self):
        if not self.boss: return
        if self.players and (self.players[0]['pos'] - self.boss['pos']).length_squared() > 0:
            direction = (self.players[0]['pos'] - self.boss['pos']).normalize()
            self.boss['pos'] += direction * self.BOSS_SPEED
        
        self.boss['shoot_cooldown'] = max(0, self.boss['shoot_cooldown'] - 1)
        if self.boss['shoot_cooldown'] == 0 and self.players:
            for i in range(-2, 3):
                boss_pos_copy = pygame.math.Vector2(self.boss['pos'])
                if (self.players[0]['pos'] - boss_pos_copy).length_squared() > 0:
                    direction = (self.players[0]['pos'] - boss_pos_copy).normalize().rotate(i * 15)
                    self.enemy_projectiles.append({'pos': boss_pos_copy, 'vel': direction * self.PROJ_SPEED})
            self.boss['shoot_cooldown'] = self.ENEMY_SHOOT_COOLDOWN

    def _update_projectiles(self):
        for proj_list in [self.player_projectiles, self.enemy_projectiles]:
            for p in proj_list[:]:
                p['pos'] += p['vel']
                if not self.screen.get_rect().collidepoint(p['pos']):
                    proj_list.remove(p)

    def _update_black_hole_pull(self):
        for entity_list in [self.players, self.enemies, [self.boss] if self.boss else []]:
            for entity in entity_list[:]:
                if not entity: continue
                dist_vec = self.BH_POS - entity['pos']
                dist_sq = dist_vec.length_squared()
                if dist_sq > 1:
                    pull_force = dist_vec.normalize() * (self.WIDTH * self.BH_PULL_STRENGTH / dist_sq)
                    entity['pos'] += pull_force
                
                if dist_vec.length() < self.BH_RADIUS:
                    self._create_particles(entity['pos'], 30, (200, 100, 255))
                    if entity in self.players:
                        if not entity['is_clone']: self.game_over = True
                        entity_list.remove(entity)
                    elif entity in self.enemies:
                        self.score += 1
                        self.reward_this_step += 1
                        self.enemies_defeated_count += 1
                        entity_list.remove(entity)
                    elif self.boss and entity == self.boss:
                        self.boss['health'] -= 5
                        if self.boss['health'] <= 0: self._on_boss_defeat()

    def _handle_collisions(self):
        for proj in self.player_projectiles[:]:
            processed_proj = False
            for enemy in self.enemies[:]:
                if proj['pos'].distance_to(enemy['pos']) < self.ENEMY_SIZE:
                    enemy['health'] -= self.PROJ_DAMAGE
                    self.reward_this_step += 0.1
                    self._create_particles(proj['pos'], 5, self.COLOR_ENEMY)
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    processed_proj = True
                    if enemy['health'] <= 0:
                        self.score += 1
                        self.reward_this_step += 1
                        self._create_particles(enemy['pos'], 20, self.COLOR_ENEMY)
                        if enemy in self.enemies: self.enemies.remove(enemy)
                        self.enemies_defeated_count += 1
                    break
            if processed_proj: continue
            
            if self.boss and proj['pos'].distance_to(self.boss['pos']) < self.BOSS_SIZE:
                self.boss['health'] -= self.PROJ_DAMAGE
                self.reward_this_step += 10
                self._create_particles(proj['pos'], 10, self.COLOR_BOSS)
                if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                if self.boss['health'] <= 0: self._on_boss_defeat()

        for proj in self.enemy_projectiles[:]:
            for player in self.players[:]:
                if proj['pos'].distance_to(player['pos']) < self.PLAYER_SIZE:
                    player['health'] -= self.PROJ_DAMAGE
                    self.reward_this_step -= 0.1
                    self._create_particles(proj['pos'], 5, self.COLOR_PLAYER)
                    if proj in self.enemy_projectiles: self.enemy_projectiles.remove(proj)
                    if player['health'] <= 0:
                        self._create_particles(player['pos'], 30, self.COLOR_PLAYER)
                        if not player['is_clone']: self.game_over = True
                        if player in self.players: self.players.remove(player)
                    break

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _spawn_enemies(self):
        if self.boss: return
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            self.enemies.append(self._create_enemy())
            self.enemy_spawn_timer = self.enemy_spawn_interval

    def _spawn_boss(self):
        if not self.boss and self.enemies_defeated_count >= self.ENEMIES_TO_BOSS:
            self.enemies.clear()
            self.boss = {
                'pos': pygame.Vector2(self.WIDTH - self.BOSS_SIZE, self.HEIGHT / 2),
                'health': self.BOSS_HEALTH_START * self.boss_level,
                'max_health': self.BOSS_HEALTH_START * self.boss_level,
                'shoot_cooldown': 0
            }

    def _on_boss_defeat(self):
        self.score += 50
        self.reward_this_step += 100
        if self.boss: self._create_particles(self.boss['pos'], 100, self.COLOR_BOSS)
        self.boss = None
        self.boss_level += 1.5
        self.enemies_defeated_count = 0
        self.game_over = True

    def _check_termination(self):
        main_player_exists = any(not p['is_clone'] for p in self.players)
        if not main_player_exists:
            self.game_over = True
        
        if self.game_over and not main_player_exists:
            self.reward_this_step -= 100
        
        return self.game_over

    def _get_closest_enemy(self, pos):
        if not self.enemies: return None
        return min(self.enemies, key=lambda e: pos.distance_squared_to(e['pos']))

    def _get_closest_enemy_dist(self, pos):
        if not self.enemies: return float('inf')
        closest = self._get_closest_enemy(pos)
        return pos.distance_to(closest['pos'])

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            pos_copy = pygame.math.Vector2(pos)
            self.particles.append({'pos': pos_copy, 'vel': vel, 'lifetime': random.randint(10, 25), 'color': color})

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        
        for x, y, color, size in self.stars:
            pygame.draw.rect(self.screen, color, (x, y, size, size))
            
        for i in range(self.BH_RADIUS, 0, -2):
            alpha = int(150 * (1 - i / self.BH_RADIUS))
            pygame.gfxdraw.filled_circle(self.screen, int(self.BH_POS.x), int(self.BH_POS.y), i, (*self.COLOR_ACCRETION_DISK, alpha))
        pygame.gfxdraw.filled_circle(self.screen, int(self.BH_POS.x), int(self.BH_POS.y), self.BH_RADIUS // 2, self.COLOR_BLACK_HOLE)
        
        for p in self.players:
            if p['magnet_target'] and p['magnet_target'] in self.enemies:
                pygame.draw.line(self.screen, self.COLOR_MAGNET_BEAM, p['pos'], p['magnet_target']['pos'], 2)
        
        for p in self.player_projectiles: self._draw_glow_circle(p['pos'], self.COLOR_PLAYER_PROJ, self.PROJ_SIZE)
        for p in self.enemy_projectiles: self._draw_glow_circle(p['pos'], self.COLOR_ENEMY_PROJ, self.PROJ_SIZE)
        for e in self.enemies: self._draw_ship(e['pos'], self.ENEMY_SIZE, self.COLOR_ENEMY, self.BH_POS); self._draw_health_bar(e['pos'], e['health'], self.ENEMY_HEALTH_START, self.ENEMY_SIZE)
        if self.boss: self._draw_ship(self.boss['pos'], self.BOSS_SIZE, self.COLOR_BOSS, self.BH_POS); self._draw_health_bar(self.boss['pos'], self.boss['health'], self.boss['max_health'], self.BOSS_SIZE)
        for p in self.players: self._draw_ship(p['pos'], self.PLAYER_SIZE, self.COLOR_PLAYER if not p['is_clone'] else self.COLOR_CLONE, self.BH_POS)
        for part in self.particles:
            alpha = max(0, int(255 * (part['lifetime'] / 25)))
            s = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.circle(s, (*part['color'], alpha), (2,2), 2)
            self.screen.blit(s, (part['pos'].x-2, part['pos'].y-2))

    def _render_ui(self):
        main_player = next((p for p in self.players if not p['is_clone']), None)
        if not main_player: return

        health_pct = max(0, main_player['health'] / self.PLAYER_HEALTH_START)
        pygame.draw.rect(self.screen, (50, 0, 0), (10, 10, 150, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (10, 10, 150 * health_pct, 20))
        
        ammo_pct = max(0, main_player['ammo'] / self.PLAYER_AMMO_START)
        pygame.draw.rect(self.screen, (0, 0, 50), (10, 35, 150, 20))
        pygame.draw.rect(self.screen, self.COLOR_AMMO, (10, 35, 150 * ammo_pct, 20))
        
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

    def _draw_ship(self, pos, size, color, target_pos):
        angle = 0
        if (target_pos - pos).length_squared() > 0:
            angle = (target_pos - pos).angle_to(pygame.Vector2(1, 0))
        points = [
            pygame.Vector2(size, 0).rotate(angle) + pos,
            pygame.Vector2(-size/2, -size/2).rotate(angle) + pos,
            pygame.Vector2(-size/2, size/2).rotate(angle) + pos,
        ]
        int_points = [(int(p.x), int(p.y)) for p in points]
        pygame.draw.polygon(self.screen, color, int_points)
        pygame.gfxdraw.aapolygon(self.screen, int_points, color)

    def _draw_health_bar(self, pos, current, maximum, owner_size):
        if current < maximum:
            bar_width = 30
            bar_height = 5
            pct = max(0, current / maximum)
            y_offset = owner_size + 5
            pygame.draw.rect(self.screen, (100, 0, 0), (pos.x - bar_width/2, pos.y - y_offset, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH, (pos.x - bar_width/2, pos.y - y_offset, bar_width * pct, bar_height))

    def _draw_glow_circle(self, pos, color, radius):
        int_pos = (int(pos.x), int(pos.y))
        for i in range(radius, 0, -1):
            alpha = int(100 * (1 - i / radius))
            pygame.gfxdraw.filled_circle(self.screen, int_pos[0], int_pos[1], radius + i, (*color, alpha))
        pygame.draw.circle(self.screen, color, int_pos, radius)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Example Usage ---
    env = GameEnv()
    
    # This human render mode is not compatible with the headless setup required by the tests
    # and is provided for local debugging only.
    try:
        pygame.display.init()
        human_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Clone Magnetar")
        
        obs, info = env.reset()
        done = False
        
        print("\n--- Controls ---")
        print("Arrows: Move")
        print("Space: Shoot")
        print("Shift: Magnetize Enemy")
        print("R: Reset")
        print("Q: Quit")
        print("----------------\n")
        
        while not done:
            # Human controls
            keys = pygame.key.get_pressed()
            mov = 0 # none
            if keys[pygame.K_UP]: mov = 1
            elif keys[pygame.K_DOWN]: mov = 2
            elif keys[pygame.K_LEFT]: mov = 3
            elif keys[pygame.K_RIGHT]: mov = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [mov, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render to human screen
            human_screen.blit(env.screen, (0, 0))
            pygame.display.flip()
            env.clock.tick(env.FPS)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}")
                done = True

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("--- Resetting Game ---")
                    obs, info = env.reset()

    finally:
        env.close()