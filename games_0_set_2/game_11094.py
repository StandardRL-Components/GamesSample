import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# The original code used pygame.gfxdraw, which is not a standard part of pygame
# and often needs to be installed separately. To ensure the code is runnable,
# we will use standard pygame drawing functions. If gfxdraw is available,
# it can be used for anti-aliasing, but it's not essential for the logic.
try:
    import pygame.gfxdraw
    GFXDRAW_AVAILABLE = True
except ImportError:
    GFXDRAW_AVAILABLE = False

from gymnasium.spaces import MultiDiscrete

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # --- Game Description ---
    game_description = (
        "Defend your central castle from waves of incoming enemies by aiming and firing your turrets."
    )
    user_guide = (
        "Controls: Use ←→ to rotate the turret and ↑↓ to adjust the launch angle. "
        "Press space to fire. Press shift to cycle between turrets."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1500 # Increased for longer gameplay waves

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 45)
    COLOR_CASTLE = (0, 255, 128)
    COLOR_CASTLE_GLOW = (0, 128, 64)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_OUTLINE = (255, 150, 150)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)
    COLOR_HEALTH_BAR_FG = (0, 255, 0)
    
    # Castle
    CASTLE_MAX_HEALTH = 100
    CASTLE_SIZE = (40, 40)

    # Physics
    GRAVITY = 0.2
    
    # Player/Turret
    AIM_SPEED = 1.5 # degrees per step
    ROTATION_SPEED = 0.05 # radians per step
    FIRE_COOLDOWN_BASE = 10 # steps
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
            self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
            self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 20)
            self.font_large = pygame.font.Font(None, 30)
            self.font_game_over = pygame.font.Font(None, 60)

        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.castle_health = 0
        self.wave = 0
        self.resources = 0
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.turrets = []
        self.current_turret_idx = 0
        self.unlocked_turret_types = []
        self.aim_angle = 0
        self.current_turret_angle_rad = 0
        self.fire_cooldown = 0
        self.prev_shift_held = False
        self.screen_shake = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.castle_health = self.CASTLE_MAX_HEALTH
        self.wave = 1
        self.resources = 0
        
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.castle_pos = pygame.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        
        # Define turret positions relative to castle center
        self.turrets = [
            self.castle_pos + pygame.Vector2(0, -30),
            self.castle_pos + pygame.Vector2(30, 0),
            self.castle_pos + pygame.Vector2(0, 30),
            self.castle_pos + pygame.Vector2(-30, 0),
        ]
        self.current_turret_idx = 0
        self.unlocked_turret_types = ['standard']
        
        self.aim_angle = -45.0  # Initial upward angle
        self.current_turret_angle_rad = math.radians(270) # Pointing up
        
        self.fire_cooldown = 0
        self.prev_shift_held = False
        self.screen_shake = 0

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward_this_step = 0
        
        # --- 1. Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Aiming
        if movement == 1: self.aim_angle = min(-5, self.aim_angle + self.AIM_SPEED) # Up
        elif movement == 2: self.aim_angle = max(-85, self.aim_angle - self.AIM_SPEED) # Down
        if movement == 3: self.current_turret_angle_rad -= self.ROTATION_SPEED # Left
        elif movement == 4: self.current_turret_angle_rad += self.ROTATION_SPEED # Right

        # Firing
        if space_held and self.fire_cooldown <= 0:
            self._fire_turret()
        
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
            
        # Cycle Turret
        if shift_held and not self.prev_shift_held:
            self.current_turret_idx = (self.current_turret_idx + 1) % len(self.turrets)
        self.prev_shift_held = shift_held

        # --- 2. Update Game Logic ---
        self._update_projectiles()
        reward_this_step += self._update_enemies()
        self._update_particles()

        # --- 3. Wave Management ---
        if not self.enemies and not self.game_over:
            self.wave += 1
            reward_this_step += 10 # Wave survived bonus
            self._check_unlocks()
            self._spawn_wave()

        # --- 4. Check Termination ---
        self.steps += 1
        terminated = self.castle_health <= 0 or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS
        if self.castle_health <= 0 and not self.game_over:
            self.game_over = True
            reward_this_step -= 100 # Game over penalty
            self._create_particles(self.castle_pos, self.COLOR_CASTLE, 100, 10)
        
        self.score += reward_this_step
        
        return (
            self._get_observation(),
            reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    def _fire_turret(self):
        turret_type = self.unlocked_turret_types[self.current_turret_idx % len(self.unlocked_turret_types)]
        turret_pos = self.turrets[self.current_turret_idx]
        
        turret_config = {
            'standard': {'speed': 8, 'damage': 5, 'color': (255, 255, 0), 'cooldown': 10, 'bounces': 0},
            'ricochet': {'speed': 7, 'damage': 4, 'color': (0, 255, 255), 'cooldown': 15, 'bounces': 2},
            'fast': {'speed': 15, 'damage': 3, 'color': (255, 100, 255), 'cooldown': 5, 'bounces': 0},
        }
        config = turret_config[turret_type]
        
        self.fire_cooldown = config['cooldown']
        
        # Combine turret rotation and launch angle
        launch_angle_rad = self.current_turret_angle_rad + math.radians(self.aim_angle + 90)
        
        velocity = pygame.Vector2(
            math.cos(launch_angle_rad) * config['speed'],
            math.sin(launch_angle_rad) * config['speed']
        )
        
        self.projectiles.append({
            'pos': turret_pos.copy(),
            'vel': velocity,
            'damage': config['damage'],
            'color': config['color'],
            'bounces_left': config['bounces'],
            'hit_enemy': False,
            'trail': [],
        })
        self._create_particles(turret_pos, config['color'], 5, 1.5, 0.1)

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p['trail'].append(p['pos'].copy())
            if len(p['trail']) > 5:
                p['trail'].pop(0)

            p['vel'].y += self.GRAVITY
            p['pos'] += p['vel']

            # Check for off-screen
            if not (0 <= p['pos'].x < self.SCREEN_WIDTH and 0 <= p['pos'].y < self.SCREEN_HEIGHT):
                if not p['hit_enemy']:
                    self.score -= 0.1 # Miss penalty
                self.projectiles.remove(p)
                continue
            
            # Check for enemy collision
            for enemy in self.enemies[:]:
                if pygame.Rect(enemy['pos'].x - enemy['size'] / 2, enemy['pos'].y - enemy['size'] / 2, enemy['size'], enemy['size']).collidepoint(p['pos']):
                    p['hit_enemy'] = True
                    enemy['health'] -= p['damage']
                    self.score += 0.1 # Hit bonus
                    self._create_particles(p['pos'], self.COLOR_ENEMY_OUTLINE, 10, 3)
                    
                    if enemy['health'] <= 0:
                        self.score += 1 # Kill bonus
                        self.resources += 1
                        self._create_particles(enemy['pos'], (255, 128, 0), 30, 5)
                        self.enemies.remove(enemy)
                    
                    if p['bounces_left'] > 0:
                        p['bounces_left'] -= 1
                        closest_enemy = None
                        min_dist = float('inf')
                        for other_enemy in self.enemies:
                            if other_enemy is not enemy:
                                dist = p['pos'].distance_to(other_enemy['pos'])
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_enemy = other_enemy
                        if closest_enemy:
                            direction = (closest_enemy['pos'] - p['pos']).normalize()
                            p['vel'] = direction * p['vel'].length()
                        else:
                            self.projectiles.remove(p)
                    else:
                        if p in self.projectiles:
                           self.projectiles.remove(p)
                    break

    def _update_enemies(self):
        reward = 0
        for e in self.enemies[:]:
            direction = (self.castle_pos - e['pos']).normalize()
            e['pos'] += direction * e['speed']
            
            enemy_rect = pygame.Rect(e['pos'].x - e['size']/2, e['pos'].y - e['size']/2, e['size'], e['size'])
            castle_rect = pygame.Rect(self.castle_pos.x - self.CASTLE_SIZE[0]/2, self.castle_pos.y - self.CASTLE_SIZE[1]/2, self.CASTLE_SIZE[0], self.CASTLE_SIZE[1])

            if enemy_rect.colliderect(castle_rect):
                self.castle_health -= 10 # Damage
                self.screen_shake = 10
                self._create_particles(e['pos'], self.COLOR_CASTLE, 20, 4)
                self.enemies.remove(e)
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_wave(self):
        num_enemies = 5 + self.wave
        enemy_health = 5 + self.wave
        enemy_speed = 0.5 + self.wave * 0.05
        
        for _ in range(num_enemies):
            side = self.np_random.integers(4)
            if side == 0: x, y = self.np_random.uniform(0, self.SCREEN_WIDTH), -20
            elif side == 1: x, y = self.SCREEN_WIDTH + 20, self.np_random.uniform(0, self.SCREEN_HEIGHT)
            elif side == 2: x, y = self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20
            else: x, y = -20, self.np_random.uniform(0, self.SCREEN_HEIGHT)
            
            self.enemies.append({
                'pos': pygame.Vector2(x, y),
                'health': enemy_health,
                'speed': enemy_speed,
                'size': self.np_random.uniform(12, 18)
            })

    def _check_unlocks(self):
        if self.wave == 5 and 'ricochet' not in self.unlocked_turret_types:
            self.unlocked_turret_types.append('ricochet')
        if self.wave == 10 and 'fast' not in self.unlocked_turret_types:
            self.unlocked_turret_types.append('fast')
    
    def _create_particles(self, pos, color, count, max_speed, life_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.uniform(10, 30) * life_mult,
                'max_life': 30 * life_mult,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _get_observation(self):
        render_offset = pygame.Vector2(0, 0)
        if self.screen_shake > 0:
            self.screen_shake -= 1
            render_offset.x = self.np_random.uniform(-self.screen_shake, self.screen_shake)
            render_offset.y = self.np_random.uniform(-self.screen_shake, self.screen_shake)

        self.screen.fill(self.COLOR_BG)
        self._render_background(render_offset)
        self._render_game(render_offset)
        self._render_ui(render_offset)
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self, offset):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x + offset.x, 0 + offset.y), (x + offset.x, self.SCREEN_HEIGHT + offset.y))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0 + offset.x, y + offset.y), (self.SCREEN_WIDTH + offset.x, y + offset.y))

    def _render_game(self, offset):
        # Castle
        castle_rect = pygame.Rect(
            self.castle_pos.x - self.CASTLE_SIZE[0] / 2, self.castle_pos.y - self.CASTLE_SIZE[1] / 2,
            self.CASTLE_SIZE[0], self.CASTLE_SIZE[1]
        )
        glow_rect = castle_rect.inflate(10, 10)
        pygame.draw.rect(self.screen, self.COLOR_CASTLE_GLOW, glow_rect.move(offset), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_CASTLE, castle_rect.move(offset), border_radius=3)
        
        # Turrets
        for i, pos in enumerate(self.turrets):
            color = self.COLOR_CASTLE if i == self.current_turret_idx else self.COLOR_CASTLE_GLOW
            pygame.draw.circle(self.screen, color, (int(pos.x + offset.x), int(pos.y + offset.y)), 5)
        
        # Aiming line
        turret_pos = self.turrets[self.current_turret_idx]
        launch_angle_rad = self.current_turret_angle_rad + math.radians(self.aim_angle + 90)
        end_pos = turret_pos + pygame.Vector2(math.cos(launch_angle_rad), math.sin(launch_angle_rad)) * 40
        pygame.draw.line(self.screen, (255,255,255,100), turret_pos + offset, end_pos + offset, 2)
        
        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = p['color']
            size = p['size'] * (p['life'] / p['max_life'])
            pos = p['pos'] + offset
            if GFXDRAW_AVAILABLE:
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(size), (*color, int(alpha)))
            else:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*color, int(alpha)), (size, size), size)
                self.screen.blit(s, (pos.x - size, pos.y - size))

        # Projectiles
        for p in self.projectiles:
            # Trail
            for i, trail_pos in enumerate(p['trail']):
                alpha = int(150 * (i / len(p['trail'])))
                if GFXDRAW_AVAILABLE:
                    pygame.gfxdraw.filled_circle(self.screen, int(trail_pos.x + offset.x), int(trail_pos.y + offset.y), 2, (*p['color'], alpha))
                else: # Fallback without gfxdraw
                    s = pygame.Surface((4,4), pygame.SRCALPHA)
                    pygame.draw.circle(s, (*p['color'], alpha), (2,2), 2)
                    self.screen.blit(s, (trail_pos.x + offset.x - 2, trail_pos.y + offset.y - 2))
            # Main projectile
            pos_x, pos_y = int(p['pos'].x + offset.x), int(p['pos'].y + offset.y)
            if GFXDRAW_AVAILABLE:
                pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, 4, p['color'])
                pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, 4, p['color'])
            else:
                pygame.draw.circle(self.screen, p['color'], (pos_x, pos_y), 4)


        # Enemies
        for e in self.enemies:
            rect = pygame.Rect(e['pos'].x - e['size']/2 + offset.x, e['pos'].y - e['size']/2 + offset.y, e['size'], e['size'])
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect, border_radius=2)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_OUTLINE, rect, 1, border_radius=2)

    def _render_ui(self, offset):
        # Health Bar
        health_ratio = max(0, self.castle_health / self.CASTLE_MAX_HEALTH)
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10 + offset.x, 10 + offset.y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10 + offset.x, 10 + offset.y, bar_width * health_ratio, bar_height))
        
        # Text
        wave_text = self.font_large.render(f"WAVE: {self.wave}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10 + offset.x, 10 + offset.y))

        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10 + offset.x, 40 + offset.y))

        turret_type = self.unlocked_turret_types[self.current_turret_idx % len(self.unlocked_turret_types)]
        turret_text = self.font_small.render(f"TURRET: {turret_type.upper()}", True, self.COLOR_UI_TEXT)
        self.screen.blit(turret_text, (10 + offset.x, 40 + offset.y))

    def _render_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0,0,0,180))
        self.screen.blit(s, (0,0))
        
        text = self.font_game_over.render("GAME OVER", True, self.COLOR_ENEMY)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "castle_health": self.castle_health,
            "enemies_left": len(self.enemies),
        }
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Un-set the dummy driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    truncated = False
    
    # Pygame setup for manual play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Castle Defender")
    clock = pygame.time.Clock()

    while running:
        # --- Action Mapping for Human Play ---
        movement = 0 # none
        space = 0 # released
        shift = 0 # released

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Display score and other info
            pygame.display.set_caption(f"Castle Defender | Score: {int(info['score'])} | Wave: {info['wave']}")

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)

        if terminated or truncated:
            # Wait a bit on the game over screen before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False
            truncated = False

    env.close()