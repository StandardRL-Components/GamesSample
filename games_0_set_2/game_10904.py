import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:08:21.611139
# Source Brief: brief_00904.md
# Brief Index: 904
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A rhythmic top-down shooter where you survive waves of enemies by creating clones and defensive portals."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to create a clone at your position and shift to create a defensive portal."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 40
    GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE
    
    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_GRID = (20, 15, 40)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 128, 128)
    COLOR_CLONE = (0, 180, 180)
    COLOR_ENEMY_BASIC = (255, 100, 0)
    COLOR_ENEMY_SHOOTER = (255, 0, 100)
    COLOR_PORTAL = (160, 32, 240)
    COLOR_PROJECTILE_PLAYER = (200, 255, 255)
    COLOR_PROJECTILE_ENEMY = (255, 150, 150)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR = (0, 255, 128)
    COLOR_HEALTH_BAR_BG = (128, 0, 64)
    
    # Game Parameters
    MAX_STEPS = 5000
    PLAYER_MAX_HEALTH = 100
    MAX_CLONES = 5
    PORTAL_DURATION = 30 # steps/beats
    COMBO_TIMEOUT = 60 # steps/beats
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Initialize all state variables to prevent AttributeError
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_health = 0
        self.player_last_move_dir = pygame.math.Vector2(0, -1)
        self.clones = []
        self.enemies = []
        self.portals = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.combo_multiplier = 1
        self.combo_timer = 0
        self.base_spawn_rate = 0.1
        self.base_enemy_speed = 0.5
        self.beat_pulse = 0.0

        self.np_random = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.math.Vector2(self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_last_move_dir = pygame.math.Vector2(0, -1) # Default up
        
        self.clones = []
        self.enemies = []
        self.portals = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        
        self.combo_multiplier = 1
        self.combo_timer = 0
        
        self.base_spawn_rate = 0.1
        self.base_enemy_speed = 0.5
        
        self.beat_pulse = 1.0

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1
        self.beat_pulse = 1.0 # Visual beat indicator
        
        # --- Update Timers & Counters ---
        reward += 0.1 # Survival reward
        self._update_combo()
        self._update_portals()
        self._update_difficulty()

        # --- Handle Player Actions ---
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        player_moved = self._handle_player_movement(movement)
        
        if space_pressed:
            self._create_clone()
        
        if shift_pressed:
            self._create_portal()
            
        # --- Attack Phase ---
        # Player and clones attack on every beat
        self._create_player_projectiles()
        # SFX: player_shoot.wav
        
        # --- Enemy Phase ---
        self._spawn_enemies()
        self._update_enemies(reward) # Pass reward to modify it inside
        
        # --- Physics & Collision Phase ---
        self._update_projectiles()
        reward += self._handle_collisions()
        
        # --- Cleanup ---
        self._cleanup_entities()
        
        # --- Termination Check ---
        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS
        if terminated and self.player_health <= 0:
            reward = -100.0
            self._create_particles(self._grid_to_pixel(self.player_pos), 100, self.COLOR_PLAYER)
            # SFX: player_death.wav
        
        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_combo(self):
        if self.combo_timer > 0:
            self.combo_timer -= 1
            if self.combo_timer == 0:
                self.combo_multiplier = 1

    def _update_portals(self):
        for portal in self.portals:
            portal['duration'] -= 1

    def _update_difficulty(self):
        # Increase spawn rate every 60 beats
        if self.steps > 0 and self.steps % 60 == 0:
            self.base_spawn_rate = min(1.0, self.base_spawn_rate + 0.05)
        # Increase enemy speed every 120 beats
        if self.steps > 0 and self.steps % 120 == 0:
            self.base_enemy_speed = min(2.0, self.base_enemy_speed + 0.02)
            
    def _handle_player_movement(self, movement_action):
        moved = False
        original_pos = self.player_pos.copy()
        
        if movement_action == 1: # Up
            self.player_pos.y -= 1
            self.player_last_move_dir = pygame.math.Vector2(0, -1)
            moved = True
        elif movement_action == 2: # Down
            self.player_pos.y += 1
            self.player_last_move_dir = pygame.math.Vector2(0, 1)
            moved = True
        elif movement_action == 3: # Left
            self.player_pos.x -= 1
            self.player_last_move_dir = pygame.math.Vector2(-1, 0)
            moved = True
        elif movement_action == 4: # Right
            self.player_pos.x += 1
            self.player_last_move_dir = pygame.math.Vector2(1, 0)
            moved = True
            
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.GRID_WIDTH - 1)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.GRID_HEIGHT - 1)

        if moved:
            # SFX: player_move.wav
            trail_pos = self._grid_to_pixel(original_pos)
            self._create_particles(trail_pos, 3, self.COLOR_PLAYER_GLOW, life=5, speed=1)
        return moved
        
    def _create_clone(self):
        if len(self.clones) < self.MAX_CLONES:
            is_duplicate = any(c['pos'] == self.player_pos for c in self.clones)
            if not is_duplicate:
                self.clones.append({'pos': self.player_pos.copy(), 'health': 50})
                # SFX: clone_create.wav
                self._create_particles(self._grid_to_pixel(self.player_pos), 20, self.COLOR_CLONE)

    def _create_portal(self):
        portal_pos = self.player_pos + self.player_last_move_dir
        if 0 <= portal_pos.x < self.GRID_WIDTH and 0 <= portal_pos.y < self.GRID_HEIGHT:
            self.portals.append({
                'pos': portal_pos,
                'duration': self.PORTAL_DURATION,
                'orientation': 'v' if self.player_last_move_dir.x != 0 else 'h'
            })
            # SFX: portal_open.wav
            self._create_particles(self._grid_to_pixel(portal_pos), 30, self.COLOR_PORTAL)

    def _create_player_projectiles(self):
        # Player
        start_pos = self._grid_to_pixel(self.player_pos)
        self.player_projectiles.append({
            'pos': start_pos, 'vel': self.player_last_move_dir * 15, 'life': 40
        })
        # Clones
        for clone in self.clones:
            start_pos = self._grid_to_pixel(clone['pos'])
            self.player_projectiles.append({
                'pos': start_pos, 'vel': self.player_last_move_dir * 15, 'life': 40
            })
            
    def _spawn_enemies(self):
        if self.np_random.random() < self.base_spawn_rate:
            edge = self.np_random.integers(4)
            if edge == 0: x, y = self.np_random.integers(self.GRID_WIDTH), 0
            elif edge == 1: x, y = self.np_random.integers(self.GRID_WIDTH), self.GRID_HEIGHT - 1
            elif edge == 2: x, y = 0, self.np_random.integers(self.GRID_HEIGHT)
            else: x, y = self.GRID_WIDTH - 1, self.np_random.integers(self.GRID_HEIGHT)
            
            enemy_type = 'basic'
            if self.steps > 300 and self.np_random.random() < 0.3:
                enemy_type = 'shooter'

            self.enemies.append({
                'pos': pygame.math.Vector2(x, y),
                'health': 100 if enemy_type == 'basic' else 70,
                'type': enemy_type,
                'attack_cooldown': 0,
            })
            # SFX: enemy_spawn.wav

    def _update_enemies(self, reward):
        for enemy in self.enemies:
            direction_to_player = (self.player_pos - enemy['pos'])
            dist = direction_to_player.length()
            
            if enemy['type'] == 'basic':
                if dist > 0:
                    enemy['pos'] += direction_to_player.normalize() * self.base_enemy_speed * 0.2
            
            elif enemy['type'] == 'shooter':
                if dist > 5: # Move closer if far
                     enemy['pos'] += direction_to_player.normalize() * self.base_enemy_speed * 0.15
                elif dist < 3: # Move away if too close
                     enemy['pos'] -= direction_to_player.normalize() * self.base_enemy_speed * 0.15
                
                if enemy['attack_cooldown'] <= 0 and dist < 8:
                    proj_vel = direction_to_player.normalize() * 8
                    self.enemy_projectiles.append({
                        'pos': self._grid_to_pixel(enemy['pos']),
                        'vel': proj_vel,
                        'life': 60
                    })
                    enemy['attack_cooldown'] = 90 # beats
                    # SFX: enemy_shoot.wav
                else:
                    enemy['attack_cooldown'] -= 1
    
    def _update_projectiles(self):
        for proj in self.player_projectiles + self.enemy_projectiles:
            proj['pos'] += proj['vel']
            proj['life'] -= 1
        
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] *= 0.95

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Enemies
        for proj in self.player_projectiles:
            for enemy in self.enemies:
                if enemy['health'] > 0:
                    enemy_px_pos = self._grid_to_pixel(enemy['pos'])
                    if (proj['pos'] - enemy_px_pos).length() < self.GRID_SIZE / 2:
                        proj['life'] = 0
                        enemy['health'] -= 25
                        self._create_particles(proj['pos'], 10, self.COLOR_ENEMY_BASIC)
                        # SFX: enemy_hit.wav
                        reward += 1.0
                        self.combo_timer = self.COMBO_TIMEOUT
                        if enemy['health'] <= 0:
                            self.combo_multiplier = min(10, self.combo_multiplier + 1)
                            if self.combo_multiplier > 1:
                                reward += 5.0 # Combo reward
                                # SFX: combo_success.wav
                            self._create_particles(enemy_px_pos, 50, self.COLOR_ENEMY_BASIC)
        
        # Enemy projectiles vs Player/Clones/Portals
        for proj in self.enemy_projectiles:
            # Vs Portals
            for portal in self.portals:
                portal_px_pos = self._grid_to_pixel(portal['pos'])
                if (proj['pos'] - portal_px_pos).length() < self.GRID_SIZE:
                    proj['life'] = 0
                    portal['duration'] -= 10 # Portal takes damage
                    self._create_particles(proj['pos'], 15, self.COLOR_PORTAL)
                    # SFX: portal_block.wav

            # Vs Player
            player_px_pos = self._grid_to_pixel(self.player_pos)
            if (proj['pos'] - player_px_pos).length() < self.GRID_SIZE / 2.5:
                proj['life'] = 0
                self.player_health -= 10
                self._create_particles(proj['pos'], 20, self.COLOR_PLAYER)
                # SFX: player_hit.wav
            
            # Vs Clones
            for clone in self.clones:
                clone_px_pos = self._grid_to_pixel(clone['pos'])
                if (proj['pos'] - clone_px_pos).length() < self.GRID_SIZE / 2.5:
                    proj['life'] = 0
                    clone['health'] -= 25
                    self._create_particles(proj['pos'], 10, self.COLOR_CLONE)

        # Enemies vs Player
        for enemy in self.enemies:
            if (enemy['pos'] - self.player_pos).length() < 0.8:
                self.player_health -= 1
                self.combo_multiplier = 1 # Reset combo
                self._create_particles(self._grid_to_pixel(self.player_pos), 5, self.COLOR_ENEMY_BASIC)
        
        return reward

    def _cleanup_entities(self):
        self.enemies = [e for e in self.enemies if e['health'] > 0]
        self.clones = [c for c in self.clones if c['health'] > 0]
        self.portals = [p for p in self.portals if p['duration'] > 0]
        self.player_projectiles = [p for p in self.player_projectiles if p['life'] > 0]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if p['life'] > 0]

    def _get_observation(self):
        self._render_background()
        self._render_grid()
        self._render_beat_indicator()
        
        self._render_portals()
        self._render_enemies()
        self._render_clones()
        self._render_player()
        
        self._render_projectiles()
        self._render_particles()
        
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.player_health, "combo": self.combo_multiplier}

    def _grid_to_pixel(self, grid_pos):
        return pygame.math.Vector2(
            grid_pos.x * self.GRID_SIZE + self.GRID_SIZE / 2,
            grid_pos.y * self.GRID_SIZE + self.GRID_SIZE / 2
        )

    # --- Rendering Methods ---
    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        if not hasattr(self, '_bg_buildings') or self.np_random is None:
            # Initialize np_random if it's not already
            if self.np_random is None:
                self.np_random, _ = gym.utils.seeding.np_random()
            
            self._bg_buildings = []
            for _ in range(30):
                x = self.np_random.integers(0, self.SCREEN_WIDTH)
                w = self.np_random.integers(10, 50)
                h = self.np_random.integers(50, 200)
                y = self.SCREEN_HEIGHT - h
                color_val = self.np_random.integers(20, 40)
                color = (color_val, color_val // 2, color_val + 10)
                self._bg_buildings.append( (pygame.Rect(x, y, w, h), color) )
        for rect, color in self._bg_buildings:
            pygame.draw.rect(self.screen, color, rect)

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
    def _render_beat_indicator(self):
        if self.beat_pulse > 0:
            alpha = int(255 * self.beat_pulse)
            color = (*self.COLOR_GRID, alpha)
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect(), width=int(8 * self.beat_pulse))
            self.screen.blit(s, (0,0))
            self.beat_pulse -= 0.1

    def _render_player(self):
        pos = self._grid_to_pixel(self.player_pos)
        size = self.GRID_SIZE / 2.5
        # Glow effect
        for i in range(5):
            alpha = 150 - i * 30
            radius = size + i * 2
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), (*self.COLOR_PLAYER_GLOW, alpha))
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(size), self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(size), self.COLOR_PLAYER)

    def _render_clones(self):
        for clone in self.clones:
            pos = self._grid_to_pixel(clone['pos'])
            size = self.GRID_SIZE / 3
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(size), self.COLOR_CLONE)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(size), self.COLOR_CLONE)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = self._grid_to_pixel(enemy['pos'])
            size = self.GRID_SIZE / 3
            color = self.COLOR_ENEMY_BASIC if enemy['type'] == 'basic' else self.COLOR_ENEMY_SHOOTER
            points = [
                (pos.x, pos.y - size),
                (pos.x - size, pos.y + size),
                (pos.x + size, pos.y + size),
            ]
            pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), color)
            pygame.gfxdraw.aatrigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), color)

    def _render_portals(self):
        for portal in self.portals:
            pos = self._grid_to_pixel(portal['pos'])
            size = self.GRID_SIZE * 0.8
            alpha = int(100 + 155 * (portal['duration'] / self.PORTAL_DURATION))
            color = (*self.COLOR_PORTAL, alpha)
            
            s = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
            
            if portal['orientation'] == 'h':
                rect = pygame.Rect(0, self.GRID_SIZE/2 - 2, self.GRID_SIZE, 4)
            else: # 'v'
                rect = pygame.Rect(self.GRID_SIZE/2 - 2, 0, 4, self.GRID_SIZE)
            
            # Shimmer effect
            offset = math.sin(self.steps * 0.5 + portal['pos'].x) * 3
            if portal['orientation'] == 'h': rect.y += offset
            else: rect.x += offset

            pygame.draw.rect(s, color, rect, border_radius=2)
            pygame.draw.rect(s, (255,255,255,alpha), rect, width=1, border_radius=2)
            
            self.screen.blit(s, (pos.x - self.GRID_SIZE/2, pos.y - self.GRID_SIZE/2))
            
    def _render_projectiles(self):
        for proj in self.player_projectiles:
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE_PLAYER, proj['pos'], proj['pos'] - proj['vel'] * 0.5, 3)
        for proj in self.enemy_projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE_ENEMY, proj['pos'], 4)

    def _create_particles(self, pos, count, color, life=20, speed=5):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * self.np_random.uniform(1, speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(life // 2, life),
                'size': self.np_random.uniform(2, 5),
                'color': color
            })
            
    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, p['pos'] - (p['size'], p['size']))
            
    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, bar_width * health_ratio, 20))
        
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        # Combo Multiplier
        if self.combo_multiplier > 1:
            combo_text = self.font_large.render(f"x{self.combo_multiplier}", True, self.COLOR_PLAYER)
            pos = self._grid_to_pixel(self.player_pos)
            self.screen.blit(combo_text, (pos.x + 20, pos.y - 20))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # Un-dummy the video driver for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    
    # --- Manual Play ---
    pygame.display.set_caption("Cyber-Clone Survival")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset(seed=42)
    terminated = False
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("R: Reset")
    print("Q: Quit")
    
    while True:
        movement = 0 # None
        space = 0 # Released
        shift = 0 # Released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    env.close()
                    quit()
                if event.key == pygame.K_r:
                    obs, info = env.reset(seed=42)
                    terminated = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation to the display screen
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            
        clock.tick(10) # Run at 10 beats per second for playability