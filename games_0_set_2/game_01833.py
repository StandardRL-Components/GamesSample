
# Generated: 2025-08-28T02:51:03.824102
# Source Brief: brief_01833.md
# Brief Index: 1833

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Hold Space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control a jumping, shooting robot to blast through waves of enemies and reach the level's end in a side-scrolling action game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 20, 45)
    COLOR_GROUND = (30, 40, 60)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_GLOW = (50, 255, 150, 50)
    COLOR_ENEMY_WALKER = (255, 80, 80)
    COLOR_ENEMY_FLYER = (255, 150, 50)
    COLOR_ENEMY_SHOOTER = (200, 50, 200)
    COLOR_PLAYER_PROJ = (255, 255, 100)
    COLOR_ENEMY_PROJ = (255, 100, 200)
    COLOR_EXPLOSION = [(255, 200, 50), (255, 100, 50), (200, 50, 50)]
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_HEALTH_BAR_BG = (80, 80, 80)
    COLOR_HEALTH_BAR_FILL = (50, 255, 150)

    # Physics & Gameplay
    FPS = 30
    GRAVITY = 0.8
    PLAYER_SPEED = 6
    PLAYER_JUMP_STRENGTH = 14
    MAX_PLAYER_HEALTH = 100
    LEVEL_LENGTH = SCREEN_WIDTH * 10 # 6400 pixels
    GROUND_Y = SCREEN_HEIGHT - 50
    MAX_STEPS = 5000
    
    # Player
    PLAYER_SHOOT_COOLDOWN = 6 # frames

    # Enemies
    ENEMY_BASE_SPEED = 1.5
    ENEMY_SPAWN_RATE = 60 # frames
    ENEMY_SHOOTER_COOLDOWN = 90 # frames

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # State variables (initialized in reset)
        self.player = {}
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.parallax_bg = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.camera_x = 0.0
        self.player_shoot_cooldown = 0
        self.enemy_spawn_timer = 0
        self.current_enemy_speed = self.ENEMY_BASE_SPEED
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player = {
            'x': 100, 'y': self.GROUND_Y - 40, 'w': 30, 'h': 40,
            'vx': 0, 'vy': 0, 'health': self.MAX_PLAYER_HEALTH,
            'on_ground': False, 'facing_right': True
        }
        
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        
        self.camera_x = 0.0
        self.player_shoot_cooldown = 0
        self.enemy_spawn_timer = self.ENEMY_SPAWN_RATE
        self.current_enemy_speed = self.ENEMY_BASE_SPEED

        self._generate_parallax_bg()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            space_held = action[1] == 1  # Boolean
            # shift_held = action[2] == 1 # Unused in this design

            # --- 1. Handle Player Input ---
            self.player['vx'] = 0
            if movement == 4: # Right
                self.player['vx'] = self.PLAYER_SPEED
                self.player['facing_right'] = True
                reward += 0.01 # Small reward for moving forward
            elif movement == 3: # Left
                self.player['vx'] = -self.PLAYER_SPEED
                self.player['facing_right'] = False
            
            if movement == 1 and self.player['on_ground']: # Jump
                self.player['vy'] = -self.PLAYER_JUMP_STRENGTH
                self.player['on_ground'] = False
                # SFX: Jump

            if space_held and self.player_shoot_cooldown <= 0:
                self._fire_player_projectile()
                self.player_shoot_cooldown = self.PLAYER_SHOOT_COOLDOWN
                # SFX: Player Shoot
            
            # --- 2. Update Game Logic ---
            # Timers
            self.steps += 1
            if self.player_shoot_cooldown > 0: self.player_shoot_cooldown -= 1
            if self.enemy_spawn_timer > 0: self.enemy_spawn_timer -= 1

            # Player physics
            self._update_player()

            # Update and move projectiles
            self._update_projectiles()

            # Update enemies
            self._update_enemies()
            
            # Spawn new enemies
            self._spawn_enemies()

            # Update particles
            self._update_particles()

            # Collision detection
            reward += self._handle_collisions()

            # Update camera
            self.camera_x += (self.player['x'] - self.camera_x - self.SCREEN_WIDTH / 3) * 0.1

            # Difficulty scaling
            if self.steps % 200 == 0 and self.steps > 0:
                self.current_enemy_speed += 0.05

            # --- 3. Check Termination Conditions ---
            if self.player['health'] <= 0:
                self.game_over = True
                reward -= 10 # Penalty for dying
                # SFX: Player Death
            
            if self.player['x'] >= self.LEVEL_LENGTH:
                self.game_over = True
                terminated = True
                reward += 100 # Big reward for finishing
                # SFX: Level Complete
            
            if self.steps >= self.MAX_STEPS:
                self.game_over = True
                terminated = True
        
        # Advance frame
        self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            terminated or self.game_over,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.player['health']}

    # --- Update Methods ---
    def _update_player(self):
        p = self.player
        p['vy'] += self.GRAVITY
        p['x'] += p['vx']
        p['y'] += p['vy']

        # World bounds
        p['x'] = max(p['w']/2, p['x'])
        
        # Ground collision
        if p['y'] + p['h'] / 2 >= self.GROUND_Y:
            p['y'] = self.GROUND_Y - p['h'] / 2
            p['vy'] = 0
            if not p['on_ground']:
                # Landing dust
                self._create_particle_burst(p['x'], self.GROUND_Y, 5, self.COLOR_UI_TEXT)
                p['on_ground'] = True
    
    def _update_projectiles(self):
        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj['x'] += proj['vx']
            if proj['x'] - self.camera_x < 0 or proj['x'] - self.camera_x > self.SCREEN_WIDTH:
                self.player_projectiles.remove(proj)
        
        # Enemy projectiles
        for proj in self.enemy_projectiles[:]:
            proj['x'] += proj['vx']
            if proj['x'] - self.camera_x < 0 or proj['x'] - self.camera_x > self.SCREEN_WIDTH:
                self.enemy_projectiles.remove(proj)

    def _update_enemies(self):
        for e in self.enemies[:]:
            e['x'] += e['vx']
            
            # Type-specific behavior
            if e['type'] == 'flyer':
                e['y'] = e['cy'] + math.sin(self.steps * 0.05 + e['phase']) * 40
            
            if e['type'] == 'shooter':
                e['shoot_timer'] -= 1
                if e['shoot_timer'] <= 0:
                    self._fire_enemy_projectile(e)
                    e['shoot_timer'] = self.ENEMY_SHOOTER_COOLDOWN + self.np_random.integers(-10, 10)
                    # SFX: Enemy Shoot

            # Remove if off-screen to the left
            if e['x'] < self.camera_x - 50:
                self.enemies.remove(e)

    def _update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    # --- Action & Spawning ---
    def _fire_player_projectile(self):
        direction = 1 if self.player['facing_right'] else -1
        self.player_projectiles.append({
            'x': self.player['x'] + (self.player['w'] / 2 + 5) * direction,
            'y': self.player['y'],
            'vx': 15 * direction,
            'w': 10, 'h': 4
        })
        # Muzzle flash
        self._create_particle_burst(
            self.player['x'] + (self.player['w']/2 + 5) * direction, 
            self.player['y'], 
            5, self.COLOR_PLAYER_PROJ, 0.5, 5
        )

    def _fire_enemy_projectile(self, enemy):
        player_dx = self.player['x'] - enemy['x']
        player_dy = self.player['y'] - enemy['y']
        dist = math.hypot(player_dx, player_dy)
        if dist == 0: return

        self.enemy_projectiles.append({
            'x': enemy['x'], 'y': enemy['y'],
            'vx': (player_dx / dist) * 7,
            'vy': (player_dy / dist) * 7,
            'r': 5
        })

    def _spawn_enemies(self):
        if self.enemy_spawn_timer <= 0:
            spawn_x = self.camera_x + self.SCREEN_WIDTH + 50
            enemy_type = self.np_random.choice(['walker', 'flyer', 'shooter'])
            
            if enemy_type == 'walker':
                self.enemies.append({
                    'type': 'walker', 'x': spawn_x, 'y': self.GROUND_Y - 15, 'w': 30, 'h': 30,
                    'health': 20, 'vx': -self.current_enemy_speed
                })
            elif enemy_type == 'flyer':
                y_pos = self.GROUND_Y - 150 + self.np_random.integers(-50, 50)
                self.enemies.append({
                    'type': 'flyer', 'x': spawn_x, 'y': y_pos, 'cy': y_pos,
                    'phase': self.np_random.random() * math.pi * 2,
                    'w': 35, 'h': 20, 'health': 30, 'vx': -self.current_enemy_speed * 1.2
                })
            elif enemy_type == 'shooter':
                self.enemies.append({
                    'type': 'shooter', 'x': spawn_x, 'y': self.GROUND_Y - 25, 'w': 40, 'h': 50,
                    'health': 50, 'vx': -self.current_enemy_speed * 0.5,
                    'shoot_timer': self.ENEMY_SHOOTER_COOLDOWN
                })
            self.enemy_spawn_timer = self.ENEMY_SPAWN_RATE - int((self.current_enemy_speed - self.ENEMY_BASE_SPEED) * 10)

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Enemies
        for proj in self.player_projectiles[:]:
            proj_rect = pygame.Rect(proj['x'] - proj['w']/2, proj['y'] - proj['h']/2, proj['w'], proj['h'])
            for enemy in self.enemies[:]:
                enemy_rect = pygame.Rect(enemy['x'] - enemy['w']/2, enemy['y'] - enemy['h']/2, enemy['w'], enemy['h'])
                if proj_rect.colliderect(enemy_rect):
                    enemy['health'] -= 10
                    reward += 1.0
                    self._create_particle_burst(proj['x'], proj['y'], 5, self.COLOR_EXPLOSION[1])
                    # SFX: Enemy Hit
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    
                    if enemy['health'] <= 0:
                        reward += 10.0
                        self.score += 100
                        self._create_particle_burst(enemy['x'], enemy['y'], 20, self.COLOR_EXPLOSION[0])
                        self.enemies.remove(enemy)
                        # SFX: Explosion
                    break

        # Enemy projectiles vs Player
        player_rect = pygame.Rect(self.player['x'] - self.player['w']/2, self.player['y'] - self.player['h']/2, self.player['w'], self.player['h'])
        for proj in self.enemy_projectiles[:]:
            proj_rect = pygame.Rect(proj['x'] - proj['r'], proj['y'] - proj['r'], proj['r']*2, proj['r']*2)
            if player_rect.colliderect(proj_rect):
                self.player['health'] -= 5
                reward -= 0.5
                self._create_particle_burst(proj['x'], proj['y'], 10, self.COLOR_PLAYER_GLOW)
                self.enemy_projectiles.remove(proj)
                # SFX: Player Hit
                break

        # Player vs Enemies
        for enemy in self.enemies:
            enemy_rect = pygame.Rect(enemy['x'] - enemy['w']/2, enemy['y'] - enemy['h']/2, enemy['w'], enemy['h'])
            if player_rect.colliderect(enemy_rect):
                self.player['health'] -= 10
                reward -= 1.0
                # SFX: Player Hit
                # Simple knockback
                self.player['vx'] = -5 if enemy['x'] < self.player['x'] else 5
                self.player['vy'] = -5
                break
        
        self.player['health'] = max(0, self.player['health'])
        return reward

    # --- Rendering ---
    def _render_game(self):
        # Parallax Background
        for layer in self.parallax_bg:
            for rect in layer['rects']:
                x = (rect[0] - self.camera_x * layer['speed']) % self.SCREEN_WIDTH
                self.screen.fill(layer['color'], (x, rect[1], rect[2], rect[3]))
                if x > 0: # Draw wrapped part
                    self.screen.fill(layer['color'], (x - self.SCREEN_WIDTH, rect[1], rect[2], rect[3]))

        # Ground
        ground_rect = pygame.Rect(0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect)

        # Helper for converting world to screen coords
        def to_screen(x, y):
            return int(x - self.camera_x), int(y)

        # Render Particles
        for p in self.particles:
            sx, sy = to_screen(p['x'], p['y'])
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color']
            if len(color) == 4:
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, int(p['r']), (*color[:3], alpha))
            else:
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, int(p['r']), (*color, alpha))

        # Render Enemies
        for e in self.enemies:
            sx, sy = to_screen(e['x'], e['y'])
            rect = pygame.Rect(sx - e['w']/2, sy - e['h']/2, e['w'], e['h'])
            color = self.COLOR_ENEMY_WALKER if e['type'] == 'walker' else self.COLOR_ENEMY_FLYER if e['type'] == 'flyer' else self.COLOR_ENEMY_SHOOTER
            pygame.draw.rect(self.screen, color, rect, border_radius=3)

        # Render Projectiles
        for p in self.player_projectiles:
            sx, sy = to_screen(p['x'], p['y'])
            rect = pygame.Rect(sx - p['w']/2, sy - p['h']/2, p['w'], p['h'])
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, rect, border_radius=2)
        for p in self.enemy_projectiles:
            sx, sy = to_screen(p['x'], p['y'])
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, p['r'], self.COLOR_ENEMY_PROJ)
            pygame.gfxdraw.aacircle(self.screen, sx, sy, p['r'], self.COLOR_ENEMY_PROJ)

        # Render Player
        p = self.player
        sx, sy = to_screen(p['x'], p['y'])
        player_rect = pygame.Rect(sx - p['w']/2, sy - p['h']/2, p['w'], p['h'])
        
        # Glow effect
        glow_radius = int(p['w'] * 0.8)
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, glow_radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, sx, sy, glow_radius, self.COLOR_PLAYER_GLOW)
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        # Eye
        eye_x = sx + (5 if p['facing_right'] else -5)
        eye_y = sy - 5
        pygame.draw.rect(self.screen, self.COLOR_BG, (eye_x-2, eye_y-1, 4, 2))


    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Health Bar
        health_pct = self.player['health'] / self.MAX_PLAYER_HEALTH
        bar_w, bar_h = 150, 20
        bar_x, bar_y = self.SCREEN_WIDTH - bar_w - 10, 10
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FILL, (bar_x, bar_y, int(bar_w * health_pct), bar_h), border_radius=3)

        # Progress Bar
        progress_pct = self.player['x'] / self.LEVEL_LENGTH
        prog_bar_w = self.SCREEN_WIDTH - 40
        prog_bar_y = self.SCREEN_HEIGHT - 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (20, prog_bar_y, prog_bar_w, 10), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (20, prog_bar_y, int(prog_bar_w * progress_pct), 10), border_radius=3)


        if self.game_over:
            msg = "LEVEL COMPLETE!" if self.player['x'] >= self.LEVEL_LENGTH else "GAME OVER"
            text_surf = self.font_game_over.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    # --- Effects & Helpers ---
    def _create_particle_burst(self, x, y, count, color, speed_mult=1.0, life_mult=1.0):
        colors = color if isinstance(color, list) else [color]
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = (self.np_random.random() * 3 + 1) * speed_mult
            life = (self.np_random.integers(10, 20)) * life_mult
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'r': self.np_random.integers(2, 5),
                'life': life, 'max_life': life,
                'color': random.choice(colors)
            })

    def _generate_parallax_bg(self):
        self.parallax_bg = []
        # Distant stars
        stars = []
        for _ in range(100):
            stars.append((self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.GROUND_Y), 1, 1))
        self.parallax_bg.append({'rects': stars, 'speed': 0.1, 'color': (80, 80, 120)})
        
        # Mid-ground city silhouette
        buildings = []
        for i in range(20):
            w = self.np_random.integers(40, 100)
            h = self.np_random.integers(50, 200)
            x = i * (self.SCREEN_WIDTH / 15) + self.np_random.integers(-20, 20)
            buildings.append((x, self.GROUND_Y - h, w, h))
        self.parallax_bg.append({'rects': buildings, 'speed': 0.3, 'color': (25, 30, 55)})

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()