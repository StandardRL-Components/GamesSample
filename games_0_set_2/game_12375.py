import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An expert-level Gymnasium environment for a visually stunning, momentum-based
    fractal shooter game. The player controls a central core, aiming and launching
    different types of "guardian" projectiles to defeat waves of enemies.
    The environment is designed for high visual quality and engaging gameplay,
    adhering strictly to the specified Gymnasium interface.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Aiming (0: none, 1: up, 2: down, 3: left, 4: right)
    - action[1]: Launch Guardian (0: released, 1: pressed)
    - action[2]: Cycle Guardian Type (0: released, 1: pressed)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - A 640x400 RGB image of the game state.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a central core and launch guardian projectiles to defeat waves of fractal enemies in this momentum-based shooter."
    )
    user_guide = (
        "Controls: Use arrow keys to aim. Press space to launch a guardian. Press shift to cycle between guardian types."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 2000
    TOTAL_WAVES = 20

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 75, 128)
    COLOR_ENEMY_A = (255, 50, 50)
    COLOR_ENEMY_A_GLOW = (128, 25, 25)
    COLOR_ENEMY_B = (255, 120, 50)
    COLOR_ENEMY_B_GLOW = (128, 60, 25)
    COLOR_TEXT = (220, 220, 240)
    COLOR_AIMER = (255, 255, 255, 100)
    
    GUARDIAN_COLORS = [
        (100, 200, 255), # Type 0: Standard
        (255, 255, 100), # Type 1: Heavy
        (150, 255, 150), # Type 2: Terraformer
    ]
    GUARDIAN_TRAIL_COLORS = [
        (50, 100, 128),
        (128, 128, 50),
        (75, 128, 75),
    ]
    
    TERRAIN_COLOR = (120, 120, 120)
    TERRAIN_GLOW_COLOR = (60, 60, 60)

    # Player
    PLAYER_MAX_HEALTH = 100
    PLAYER_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    AIM_SPEED = 0.08

    # Guardians
    GUARDIAN_MAX_COUNT = 15
    GUARDIAN_SPECS = {
        0: {'speed': 8, 'damage': 10, 'size': 6, 'cooldown': 5, 'lifespan': 60}, # Standard
        1: {'speed': 5, 'damage': 30, 'size': 10, 'cooldown': 15, 'lifespan': 80}, # Heavy
        2: {'speed': 6, 'damage': 0, 'size': 8, 'cooldown': 20, 'lifespan': 50}, # Terraformer
    }
    
    # Enemies
    ENEMY_BASE_SPEED = 1.0
    ENEMY_BASE_HEALTH = 20
    ENEMY_BASE_DAMAGE = 10
    ENEMY_BASE_SIZE = 12

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Exact spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # State variables are initialized in reset()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.reward_this_step = 0
        self.game_over = False
        self.wave = 1
        
        # Player state
        self.player_health = self.PLAYER_MAX_HEALTH
        self.aim_angle = 0.0
        self.unlocked_guardian_types = 1
        self.selected_guardian_type = 0
        self.guardian_cooldowns = [0] * len(self.GUARDIAN_SPECS)
        
        # Input state
        self.prev_space_held = False
        self.prev_shift_held = False

        # Entity lists
        self.guardians = []
        self.enemies = []
        self.particles = []
        self.terrain_blocks = []

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing and return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.reward_this_step = 0

        # --- Handle Actions ---
        self._handle_input(action)

        # --- Update Game Logic ---
        self._update_cooldowns()
        self._update_entities()
        self._handle_collisions()

        # --- Wave Progression ---
        if not self.enemies and not self.game_over:
            self.wave += 1
            if self.wave > self.TOTAL_WAVES:
                self.game_over = True
                self.reward_this_step += 100 # Win bonus
            else:
                self._spawn_wave()
                if self.wave % 5 == 0:
                    self.unlocked_guardian_types = min(len(self.GUARDIAN_SPECS), self.unlocked_guardian_types + 1)

        # --- Finalize Step ---
        reward = self.reward_this_step
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over: # Loss condition
            self.game_over = True
            reward -= 100 # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Aiming
        if movement == 1: self.aim_angle -= self.AIM_SPEED
        elif movement == 2: self.aim_angle += self.AIM_SPEED
        elif movement == 3: self.aim_angle -= self.AIM_SPEED
        elif movement == 4: self.aim_angle += self.AIM_SPEED
        self.aim_angle %= (2 * math.pi)

        # Cycle Guardian Type (on press)
        if shift_held and not self.prev_shift_held:
            # sfx: UI_cycle.wav
            self.selected_guardian_type = (self.selected_guardian_type + 1) % self.unlocked_guardian_types

        # Launch Guardian (on press)
        if space_held and not self.prev_space_held:
            self._launch_guardian()
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_cooldowns(self):
        for i in range(len(self.guardian_cooldowns)):
            if self.guardian_cooldowns[i] > 0:
                self.guardian_cooldowns[i] -= 1
    
    def _update_entities(self):
        # Update guardians
        for g in self.guardians[:]:
            g['pos'] += g['vel']
            g['lifespan'] -= 1
            g['trail'].append(pygame.math.Vector2(g['pos']))
            if len(g['trail']) > 15:
                g['trail'].pop(0)
            
            if g['lifespan'] <= 0 or not (0 < g['pos'].x < self.SCREEN_WIDTH and 0 < g['pos'].y < self.SCREEN_HEIGHT):
                if g['type'] == 2 and (0 < g['pos'].x < self.SCREEN_WIDTH and 0 < g['pos'].y < self.SCREEN_HEIGHT): # Terraformer
                    self._create_terrain_block(g['pos'])
                self.guardians.remove(g)
        
        # Update enemies
        for e in self.enemies:
            e['pos'] += e['vel']
            # Simple wall bounce
            if not (e['size'] < e['pos'].x < self.SCREEN_WIDTH - e['size']):
                e['vel'].x *= -1
            if not (e['size'] < e['pos'].y < self.SCREEN_HEIGHT - e['size']):
                e['vel'].y *= -1
        
        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        # Guardians vs Enemies
        for g in self.guardians[:]:
            for e in self.enemies[:]:
                if g['pos'].distance_to(e['pos']) < g['size'] + e['size']:
                    # sfx: impact_heavy.wav or impact_light.wav
                    self._create_particles(g['pos'], g['color'], 20)
                    e['health'] -= g['damage']
                    self.reward_this_step += g['damage'] * 0.1
                    
                    if g in self.guardians: self.guardians.remove(g)

                    if e['health'] <= 0:
                        self.score += 1
                        self.reward_this_step += 1.0
                        # sfx: explosion.wav
                        self._create_particles(e['pos'], e['color'], 40)
                        if e in self.enemies: self.enemies.remove(e)
                    break
        
        # Guardians vs Terrain
        for g in self.guardians[:]:
            for t in self.terrain_blocks[:]:
                if g['pos'].distance_to(t['pos']) < g['size'] + t['size']:
                    self._create_particles(g['pos'], self.TERRAIN_COLOR, 10)
                    # sfx: bounce.wav
                    # Reflect velocity
                    normal = (g['pos'] - t['pos']).normalize()
                    g['vel'] = g['vel'].reflect(normal)
                    t['health'] -= g['damage']
                    if t['health'] <= 0:
                        self.terrain_blocks.remove(t)
                        self._create_particles(t['pos'], self.TERRAIN_COLOR, 30)

        # Enemies vs Player Core
        player_vec = pygame.math.Vector2(self.PLAYER_POS)
        player_radius = max(5, 40 * (self.player_health / self.PLAYER_MAX_HEALTH))
        for e in self.enemies[:]:
            if e['pos'].distance_to(player_vec) < e['size'] + player_radius:
                # sfx: player_hit.wav
                self.player_health -= e['damage']
                self.reward_this_step -= 0.5 # Small penalty for getting hit
                self._create_particles(e['pos'], self.COLOR_PLAYER, 30)
                self.enemies.remove(e)

    def _spawn_wave(self):
        num_enemies = 2 + self.wave
        enemy_speed = self.ENEMY_BASE_SPEED + self.wave * 0.05
        enemy_health = self.ENEMY_BASE_HEALTH + self.wave * 5
        enemy_type_count = 1 + (self.wave -1) // 4
        
        for _ in range(num_enemies):
            # Spawn on edge of screen
            side = random.randint(0, 3)
            if side == 0: pos = pygame.math.Vector2(random.randint(0, self.SCREEN_WIDTH), -20)
            elif side == 1: pos = pygame.math.Vector2(random.randint(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20)
            elif side == 2: pos = pygame.math.Vector2(-20, random.randint(0, self.SCREEN_HEIGHT))
            else: pos = pygame.math.Vector2(self.SCREEN_WIDTH + 20, random.randint(0, self.SCREEN_HEIGHT))
            
            # Velocity towards general center area
            target = pygame.math.Vector2(self.PLAYER_POS) + pygame.math.Vector2(random.uniform(-100, 100), random.uniform(-100, 100))
            vel = (target - pos).normalize() * enemy_speed
            
            enemy_type = random.randint(0, min(1, enemy_type_count - 1))
            color = self.COLOR_ENEMY_A if enemy_type == 0 else self.COLOR_ENEMY_B
            glow_color = self.COLOR_ENEMY_A_GLOW if enemy_type == 0 else self.COLOR_ENEMY_B_GLOW

            self.enemies.append({
                'pos': pos, 'vel': vel, 'health': enemy_health, 'type': enemy_type,
                'size': self.ENEMY_BASE_SIZE, 'damage': self.ENEMY_BASE_DAMAGE,
                'color': color, 'glow_color': glow_color
            })
    
    def _launch_guardian(self):
        spec = self.GUARDIAN_SPECS[self.selected_guardian_type]
        if self.guardian_cooldowns[self.selected_guardian_type] == 0 and len(self.guardians) < self.GUARDIAN_MAX_COUNT:
            # sfx: launch.wav
            vel = pygame.math.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * spec['speed']
            self.guardians.append({
                'pos': pygame.math.Vector2(self.PLAYER_POS),
                'vel': vel,
                'type': self.selected_guardian_type,
                'size': spec['size'],
                'damage': spec['damage'],
                'lifespan': spec['lifespan'],
                'color': self.GUARDIAN_COLORS[self.selected_guardian_type],
                'trail_color': self.GUARDIAN_TRAIL_COLORS[self.selected_guardian_type],
                'trail': []
            })
            self.guardian_cooldowns[self.selected_guardian_type] = spec['cooldown']

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': pygame.math.Vector2(random.uniform(-2, 2), random.uniform(-2, 2)),
                'lifespan': random.randint(10, 20),
                'color': color,
                'size': random.randint(1, 4)
            })

    def _create_terrain_block(self, pos):
        # sfx: terraform_create.wav
        self.terrain_blocks.append({
            'pos': pos,
            'size': 20,
            'health': 100,
        })
        self._create_particles(pos, self.TERRAIN_COLOR, 30)

    def _check_termination(self):
        return self.player_health <= 0 or (self.wave > self.TOTAL_WAVES)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "player_health": self.player_health,
            "enemies_remaining": len(self.enemies),
            "unlocked_guardians": self.unlocked_guardian_types
        }

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        self._render_background_stars()
        
        # Render all game elements
        self._render_terrain()
        self._render_enemies()
        self._render_guardians()
        self._render_player_core()
        self._render_particles()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    # --- Rendering Methods ---

    def _render_background_stars(self):
        # Slow-moving parallax stars for depth
        for i in range(50):
            # Use step count for deterministic but moving positions
            x = (i * 37 + self.steps // 5) % self.SCREEN_WIDTH
            y = (i * 149 + self.steps // 5) % self.SCREEN_HEIGHT
            pygame.gfxdraw.pixel(self.screen, x, y, (50, 50, 80))

    def _render_player_core(self):
        pos = pygame.math.Vector2(self.PLAYER_POS)
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        radius = int(10 + 30 * health_ratio)
        
        # Glow effect
        glow_radius = int(radius * (1.5 + 0.5 * math.sin(self.steps * 0.1)))
        glow_color = (*self.COLOR_PLAYER_GLOW, int(100 * (1 - health_ratio)))
        self._draw_circle_alpha(self.screen, glow_color, (int(pos.x), int(pos.y)), glow_radius)

        # Main core
        self._draw_recursive_fractal(pos, radius, self.COLOR_PLAYER, 5, self.steps * 0.02)
        
        # Aiming reticle
        end_pos = pos + pygame.math.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * (radius + 20)
        pygame.draw.line(self.screen, self.COLOR_TEXT, (int(pos.x), int(pos.y)), (int(end_pos.x), int(end_pos.y)), 1)
        pygame.gfxdraw.aacircle(self.screen, int(end_pos.x), int(end_pos.y), 3, self.COLOR_TEXT)

    def _render_guardians(self):
        for g in self.guardians:
            # Trail
            if len(g['trail']) > 1:
                scaled_trail = [ (int(p.x), int(p.y)) for p in g['trail']]
                pygame.draw.aalines(self.screen, g['trail_color'], False, scaled_trail, 2)
            
            # Guardian body
            pos_int = (int(g['pos'].x), int(g['pos'].y))
            if g['type'] == 0: # Triangle
                p1 = (pos_int[0], pos_int[1] - g['size'])
                p2 = (pos_int[0] - g['size']//2, pos_int[1] + g['size']//2)
                p3 = (pos_int[0] + g['size']//2, pos_int[1] + g['size']//2)
                pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), g['color'])
                pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), g['color'])
            elif g['type'] == 1: # Square
                 pygame.gfxdraw.box(self.screen, (pos_int[0] - g['size']//2, pos_int[1] - g['size']//2, g['size'], g['size']), g['color'])
            elif g['type'] == 2: # Hexagon
                points = []
                for i in range(6):
                    angle = math.pi / 3 * i
                    points.append((pos_int[0] + g['size'] * math.cos(angle), pos_int[1] + g['size'] * math.sin(angle)))
                pygame.gfxdraw.aapolygon(self.screen, points, g['color'])
                pygame.gfxdraw.filled_polygon(self.screen, points, g['color'])

    def _render_enemies(self):
        for e in self.enemies:
            pos = e['pos']
            radius = int(e['size'])
            
            # Glow
            glow_radius = int(radius * 1.8)
            self._draw_circle_alpha(self.screen, (*e['glow_color'], 80), (int(pos.x), int(pos.y)), glow_radius)

            # Body
            self._draw_recursive_fractal(pos, radius, e['color'], 4, -self.steps * 0.03 + e['type'] * math.pi)

    def _render_terrain(self):
        for t in self.terrain_blocks:
            pos_int = (int(t['pos'].x), int(t['pos'].y))
            size = t['size']
            
            # Glow
            self._draw_circle_alpha(self.screen, (*self.TERRAIN_GLOW_COLOR, 100), pos_int, int(size*1.5))
            
            # Body
            rect = pygame.Rect(pos_int[0] - size, pos_int[1] - size, size*2, size*2)
            pygame.draw.rect(self.screen, self.TERRAIN_COLOR, rect, 0, 4)
            pygame.draw.rect(self.screen, tuple(int(c*0.7) for c in self.TERRAIN_COLOR), rect, 2, 4)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['lifespan'] / 20.0))
            color = (*p['color'], alpha)
            rect = pygame.Rect(int(p['pos'].x), int(p['pos'].y), p['size'], p['size'])
            shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
            self.screen.blit(shape_surf, rect)

    def _render_ui(self):
        # Wave counter
        wave_text = self.font_large.render(f"WAVE {self.wave}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        
        # Guardian selection
        y_offset = self.SCREEN_HEIGHT - 30
        for i in range(self.unlocked_guardian_types):
            color = self.GUARDIAN_COLORS[i]
            spec = self.GUARDIAN_SPECS[i]
            
            is_selected = i == self.selected_guardian_type
            is_on_cooldown = self.guardian_cooldowns[i] > 0
            
            box_color = tuple(int(c*0.5) for c in color) if not is_selected else color
            rect = pygame.Rect(self.SCREEN_WIDTH - 120 + i*40, y_offset, 35, 20)
            pygame.draw.rect(self.screen, box_color, rect, 0, 3)

            if is_selected:
                pygame.draw.rect(self.screen, (255,255,255), rect, 2, 3)

            if is_on_cooldown:
                cooldown_ratio = self.guardian_cooldowns[i] / spec['cooldown']
                cd_rect = pygame.Rect(rect.left, rect.top, rect.width, rect.height * cooldown_ratio)
                pygame.draw.rect(self.screen, (0,0,0,150), cd_rect, 0, 3)

    def _draw_recursive_fractal(self, pos, size, color, depth, angle_offset):
        if depth <= 0 or size < 2:
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(size), color)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(size), color)
            return

        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(size), color)
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(size), color)
        
        num_branches = 3 if color == self.COLOR_PLAYER else 2
        for i in range(num_branches):
            angle = (2 * math.pi / num_branches) * i + angle_offset
            new_pos = pos + pygame.math.Vector2(math.cos(angle), math.sin(angle)) * size * 1.5
            self._draw_recursive_fractal(new_pos, size * 0.5, color, depth - 1, angle_offset)
            
    def _draw_circle_alpha(self, surface, color, center, radius):
        target_rect = pygame.Rect(center, (0, 0)).inflate((radius * 2, radius * 2))
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        pygame.draw.circle(shape_surf, color, (radius, radius), radius)
        surface.blit(shape_surf, target_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Example Usage and Manual Play ---
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup for manual play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Fractal Guardian")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    running = True
    total_reward = 0

    # Remove the auto_advance=True behavior for manual play
    # to make it step-on-action
    env.auto_advance = False

    while running:
        # --- Action Mapping for Manual Control ---
        action = [0, 0, 0] # Default no-op
        
        # Poll events to see if any keys are pressed this frame
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                action_taken = True
                if event.key == pygame.K_r:
                    print("--- Resetting Environment ---")
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False
                    truncated = False
        
        # Hold-based controls
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1; action_taken = True
        if keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2; action_taken = True
        if keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3; action_taken = True
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4; action_taken = True
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        if space_held: action_taken = True
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        if shift_held: action_taken = True
        
        action = [movement, space_held, shift_held]

        if (not terminated and not truncated) and action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"--- Game Over ---")
                print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Wave: {info['wave']}")

        # --- Rendering ---
        # Get the latest observation for rendering, even if no action was taken
        obs_render = env._get_observation()
        surf = pygame.surfarray.make_surface(np.transpose(obs_render, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()