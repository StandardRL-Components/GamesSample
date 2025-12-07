import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:41:14.835809
# Source Brief: brief_02761.md
# Brief Index: 2761
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
        "Defend your central cog from waves of enemies by placing and upgrading steampunk turrets. "
        "Manage resources and manipulate time to survive."
    )
    user_guide = (
        "Use the arrow keys to move the cursor. Press space to build or upgrade a turret. "
        "Press shift to toggle slow-motion time manipulation."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_EPISODE_STEPS = 5000

    # Colors (Steampunk Palette)
    COLOR_BG = (34, 32, 52)
    COLOR_BG_GEAR = (44, 42, 62)
    COLOR_COG = (212, 175, 55)
    COLOR_COG_DARK = (160, 130, 40)
    COLOR_TURRET = (0, 191, 255)
    COLOR_TURRET_GLOW = (0, 191, 255, 50)
    COLOR_ENEMY = (255, 69, 0)
    COLOR_ENEMY_GLOW = (255, 69, 0, 50)
    COLOR_PROJECTILE = (255, 255, 0)
    COLOR_PROJECTILE_GLOW = (255, 255, 0, 80)
    COLOR_HEALTH_BAR = (0, 255, 127)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_INVALID_CURSOR = (255, 0, 0)
    COLOR_TIME_WARP_VIGNETTE = (75, 0, 130, 100)

    # Game Parameters
    COG_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    COG_RADIUS = 40
    COG_MAX_HEALTH = 100
    CURSOR_SPEED = 10

    TURRET_BASE_COST = 50
    TURRET_UPGRADE_COST_MULT = 1.5
    TURRET_MIN_SPACING = 50
    TURRET_BASE_RANGE = 80
    TURRET_BASE_COOLDOWN = 60 # Ticks
    TURRET_BASE_DMG = 10

    ENEMY_BASE_HEALTH = 10
    ENEMY_BASE_SPEED = 0.8
    ENEMY_DAMAGE = 10
    WAVE_COOLDOWN = 180 # Ticks (6 seconds at 30fps)

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
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Pre-render static elements for performance
        self._background_surface = None
        self._vignette_surface = self._create_vignette()
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cog_health = 0
        self.resources = 0
        self.wave_number = 0
        self.time_until_next_wave = 0
        self.cursor_pos = np.array([0, 0], dtype=np.float32)
        self.turrets = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.time_manipulation_active = False
        self.last_space_press = False
        self.last_shift_press = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.cog_health = self.COG_MAX_HEALTH
        self.resources = 100 # Start with enough for 2 turrets
        self.wave_number = 0
        self.time_until_next_wave = 0
        
        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        
        self.turrets = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.time_manipulation_active = False
        self.last_space_press = False
        self.last_shift_press = False
        
        if self._background_surface is None:
            self._background_surface = self._create_background()

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        self.steps += 1
        
        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held, shift_held)

        time_scale = 0.4 if self.time_manipulation_active else 1.0

        # --- Update Game Logic ---
        reward += self._update_turrets(time_scale)
        reward += self._update_projectiles(time_scale)
        reward += self._update_enemies(time_scale)
        self._update_particles(time_scale)
        reward += self._update_wave_logic()

        # --- Check Termination ---
        terminated = self.cog_health <= 0
        truncated = self.steps >= self.MAX_EPISODE_STEPS

        if terminated:
            self.game_over = True
            reward = -100.0 # Penalty for losing
        elif truncated:
            self.game_over = True
            reward = 100.0 # Bonus for surviving

        self.score += reward
        self.clock.tick(self.FPS)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Cursor Movement
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

        # Place/Upgrade Turret (on press)
        if space_held and not self.last_space_press:
            turret_under_cursor = self._get_turret_at(self.cursor_pos)
            if turret_under_cursor:
                self._upgrade_turret(turret_under_cursor)
            else:
                self._place_turret(self.cursor_pos)
        self.last_space_press = space_held

        # Toggle Time Manipulation (on press)
        if shift_held and not self.last_shift_press:
            self.time_manipulation_active = not self.time_manipulation_active
            # SFX: TimeWarp_Activate.wav / TimeWarp_Deactivate.wav
        self.last_shift_press = shift_held

    def _update_turrets(self, time_scale):
        for turret in self.turrets:
            turret['cooldown'] = max(0, turret['cooldown'] - time_scale)
            if turret['cooldown'] == 0:
                target = self._find_target(turret)
                if target:
                    self._fire_projectile(turret, target)
                    turret['cooldown'] = turret['fire_rate']
        return 0.0

    def _update_projectiles(self, time_scale):
        reward = 0.0
        for proj in self.projectiles[:]:
            proj['pos'] += proj['vel'] * time_scale
            # Check for collision with any enemy
            for enemy in self.enemies[:]:
                if np.linalg.norm(proj['pos'] - enemy['pos']) < enemy['radius']:
                    enemy['health'] -= proj['damage']
                    # SFX: Hit_Confirm.wav
                    if enemy['health'] <= 0:
                        reward += 0.1 # Reward for destroying an enemy
                        self.resources += enemy['value']
                        self._create_explosion(enemy['pos'], 20, self.COLOR_ENEMY)
                        self.enemies.remove(enemy)
                        # SFX: Enemy_Explode.wav
                    if proj in self.projectiles:
                        self.projectiles.remove(proj)
                    break 
            # Remove projectile if it goes off-screen
            if not (0 < proj['pos'][0] < self.SCREEN_WIDTH and 0 < proj['pos'][1] < self.SCREEN_HEIGHT):
                 if proj in self.projectiles:
                        self.projectiles.remove(proj)
        return reward

    def _update_enemies(self, time_scale):
        for enemy in self.enemies[:]:
            direction = self.COG_POS - enemy['pos']
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction = direction / dist
            
            enemy['pos'] += direction * enemy['speed'] * time_scale
            
            if dist < self.COG_RADIUS:
                self.cog_health -= self.ENEMY_DAMAGE
                self.cog_health = max(0, self.cog_health)
                self._create_explosion(enemy['pos'], 10, self.COLOR_COG_DARK)
                self.enemies.remove(enemy)
                # SFX: Cog_Damage.wav
        return 0.0

    def _update_particles(self, time_scale):
        for p in self.particles[:]:
            p['pos'] += p['vel'] * time_scale
            p['life'] -= time_scale
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_wave_logic(self):
        reward = 0.0
        if not self.enemies and self.time_until_next_wave == 0:
            self.time_until_next_wave = self.WAVE_COOLDOWN

        if self.time_until_next_wave > 0:
            self.time_until_next_wave -= 1
            if self.time_until_next_wave == 0:
                self.wave_number += 1
                self._spawn_wave()
                reward += 10.0 # Reward for surviving a wave
                if self.wave_number % 10 == 0:
                    reward += 1.0 # Reward for new blueprint (conceptual)
        return reward

    def _spawn_wave(self):
        # Difficulty scaling
        wave_difficulty_mult = 1 + (self.wave_number // 5) * 0.05
        num_enemies = 3 + self.wave_number * 2
        enemy_health = self.ENEMY_BASE_HEALTH * wave_difficulty_mult
        enemy_speed = self.ENEMY_BASE_SPEED * (1 + (self.wave_number // 5) * 0.1)

        for _ in range(num_enemies):
            angle = self.np_random.uniform(0, 2 * math.pi)
            spawn_dist = max(self.SCREEN_WIDTH, self.SCREEN_HEIGHT) / 2 + 30
            pos = np.array([
                self.COG_POS[0] + math.cos(angle) * spawn_dist,
                self.COG_POS[1] + math.sin(angle) * spawn_dist
            ])
            self.enemies.append({
                'pos': pos,
                'health': enemy_health,
                'max_health': enemy_health,
                'speed': enemy_speed,
                'radius': 8,
                'value': 5
            })
        # SFX: New_Wave_Alert.wav

    def _place_turret(self, pos):
        cost = self.TURRET_BASE_COST
        if self.resources >= cost and self._is_valid_placement(pos):
            self.resources -= cost
            self.turrets.append({
                'pos': np.copy(pos),
                'level': 1,
                'range': self.TURRET_BASE_RANGE,
                'fire_rate': self.TURRET_BASE_COOLDOWN,
                'cooldown': 0,
                'damage': self.TURRET_BASE_DMG
            })
            # SFX: Turret_Place.wav
    
    def _upgrade_turret(self, turret):
        cost = int(self.TURRET_BASE_COST * self.TURRET_UPGRADE_COST_MULT ** turret['level'])
        if self.resources >= cost:
            self.resources -= cost
            turret['level'] += 1
            turret['range'] *= 1.1
            turret['fire_rate'] *= 0.9
            turret['damage'] *= 1.2
            self._create_explosion(turret['pos'], 15, self.COLOR_TURRET)
            # SFX: Turret_Upgrade.wav

    def _is_valid_placement(self, pos):
        # Not too close to the central cog
        if np.linalg.norm(pos - np.array(self.COG_POS)) < self.COG_RADIUS + self.TURRET_MIN_SPACING / 2:
            return False
        # Not too close to other turrets
        for t in self.turrets:
            if np.linalg.norm(pos - t['pos']) < self.TURRET_MIN_SPACING:
                return False
        return True

    def _get_turret_at(self, pos):
        for t in self.turrets:
            if np.linalg.norm(pos - t['pos']) < 10 + t['level']: # 10 is base radius
                return t
        return None

    def _find_target(self, turret):
        for enemy in self.enemies:
            if np.linalg.norm(turret['pos'] - enemy['pos']) <= turret['range']:
                return enemy
        return None

    def _fire_projectile(self, turret, target):
        direction = target['pos'] - turret['pos']
        dist = np.linalg.norm(direction)
        if dist > 0:
            direction /= dist
        
        self.projectiles.append({
            'pos': np.copy(turret['pos']),
            'vel': direction * 5, # Projectile speed
            'damage': turret['damage']
        })
        # SFX: Turret_Fire.wav
        # Muzzle flash particle
        self.particles.append({
            'pos': np.copy(turret['pos']),
            'vel': np.zeros(2),
            'life': 5,
            'color': self.COLOR_PROJECTILE,
            'radius': 4
        })

    def _get_observation(self):
        self.screen.blit(self._background_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "cog_health": self.cog_health,
            "resources": self.resources,
            "turrets": len(self.turrets),
            "enemies": len(self.enemies)
        }
    
    # --- Rendering Methods ---
    def _create_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        bg.fill(self.COLOR_BG)
        for _ in range(20):
            pos = (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT))
            radius = self.np_random.integers(20, 80)
            teeth = self.np_random.integers(8, 16)
            self._draw_gear(bg, pos, radius, teeth, self.COLOR_BG_GEAR)
        return bg
    
    def _create_vignette(self):
        vignette = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for i in range(150):
            alpha = i / 1.5
            color = (0, 0, 0, alpha)
            pygame.draw.rect(vignette, color, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), i * 2)
        return vignette

    def _render_game(self):
        self._draw_cog()
        for p in self.particles: self._draw_particle(p)
        for t in self.turrets: self._draw_turret(t)
        for e in self.enemies: self._draw_enemy(e)
        for p in self.projectiles: self._draw_projectile(p)
        self._draw_cursor()

    def _render_ui(self):
        # Time warp effect
        if self.time_manipulation_active:
            self.screen.blit(self._vignette_surface, (0, 0))

        # Cog Health Bar
        health_percent = self.cog_health / self.COG_MAX_HEALTH
        if health_percent > 0:
            start_angle = math.pi / 2
            end_angle = start_angle + (2 * math.pi * health_percent)
            pygame.draw.arc(self.screen, self.COLOR_HEALTH_BAR, 
                            (self.COG_POS[0] - self.COG_RADIUS - 5, self.COG_POS[1] - self.COG_RADIUS - 5, 
                             (self.COG_RADIUS + 5) * 2, (self.COG_RADIUS + 5) * 2),
                            start_angle, end_angle, 5)

        # Wave Counter
        wave_text = self.font_large.render(f"WAVE {self.wave_number}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, 10))

        # Resources
        resource_text = self.font_large.render(f"SCRAP: {self.resources}", True, self.COLOR_COG)
        self.screen.blit(resource_text, (self.SCREEN_WIDTH - resource_text.get_width() - 10, 10))

        # Next wave timer
        if self.time_until_next_wave > 0:
            timer_text = self.font_small.render(f"Next wave in {self.time_until_next_wave / self.FPS:.1f}s", True, self.COLOR_UI_TEXT)
            self.screen.blit(timer_text, (10, 40))

    def _draw_gear(self, surface, pos, radius, num_teeth, color):
        tooth_height = radius / 4
        tooth_width = (2 * math.pi * radius) / (num_teeth * 2)
        
        # Draw main body
        pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]), int(radius), color)
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius), color)
        
        # Draw teeth
        for i in range(num_teeth):
            angle = (i / num_teeth) * 2 * math.pi
            outer_point = (pos[0] + (radius + tooth_height) * math.cos(angle), 
                           pos[1] + (radius + tooth_height) * math.sin(angle))
            
            points = []
            for j in range(4):
                tooth_angle = angle + (j - 1.5) * (tooth_width / radius)
                r = radius if j == 0 or j == 3 else radius + tooth_height
                points.append((pos[0] + r * math.cos(tooth_angle), pos[1] + r * math.sin(tooth_angle)))

            int_points = [(int(p[0]), int(p[1])) for p in points]
            pygame.gfxdraw.aapolygon(surface, int_points, color)
            pygame.gfxdraw.filled_polygon(surface, int_points, color)
        
        # Draw inner hole
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius/2), self.COLOR_BG)

    def _draw_cog(self):
        self._draw_gear(self.screen, self.COG_POS, self.COG_RADIUS, 12, self.COLOR_COG_DARK)
        self._draw_gear(self.screen, self.COG_POS, self.COG_RADIUS-2, 12, self.COLOR_COG)

    def _draw_turret(self, turret):
        pos_i = (int(turret['pos'][0]), int(turret['pos'][1]))
        radius = int(10 + turret['level'])
        
        # Range indicator
        pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1], int(turret['range']), (*self.COLOR_TURRET, 50))
        
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, pos_i[0], pos_i[1], radius + 4, self.COLOR_TURRET_GLOW)
        
        # Base
        pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1], radius, self.COLOR_TURRET)
        pygame.gfxdraw.filled_circle(self.screen, pos_i[0], pos_i[1], radius, self.COLOR_TURRET)
        
        # Barrel
        target = self._find_target(turret)
        angle = 0
        if target:
            angle = math.atan2(target['pos'][1] - pos_i[1], target['pos'][0] - pos_i[0])
        
        barrel_length = radius + 5
        end_x = pos_i[0] + barrel_length * math.cos(angle)
        end_y = pos_i[1] + barrel_length * math.sin(angle)
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, pos_i, (int(end_x), int(end_y)), 3)

    def _draw_enemy(self, enemy):
        pos_i = (int(enemy['pos'][0]), int(enemy['pos'][1]))
        radius = int(enemy['radius'])
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, pos_i[0], pos_i[1], radius + 3, self.COLOR_ENEMY_GLOW)
        # Body
        pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1], radius, self.COLOR_ENEMY)
        pygame.gfxdraw.filled_circle(self.screen, pos_i[0], pos_i[1], radius, self.COLOR_ENEMY)
        # Health bar
        health_pct = enemy['health'] / enemy['max_health']
        bar_width = radius * 2
        bar_height = 4
        bar_x = pos_i[0] - radius
        bar_y = pos_i[1] - radius - 8
        pygame.draw.rect(self.screen, (50, 0, 0), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, bar_width * health_pct, bar_height))

    def _draw_projectile(self, proj):
        pos_i = (int(proj['pos'][0]), int(proj['pos'][1]))
        pygame.gfxdraw.filled_circle(self.screen, pos_i[0], pos_i[1], 5, self.COLOR_PROJECTILE_GLOW)
        pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1], 3, self.COLOR_PROJECTILE)
        pygame.gfxdraw.filled_circle(self.screen, pos_i[0], pos_i[1], 3, self.COLOR_PROJECTILE)

    def _draw_particle(self, p):
        alpha = max(0, min(255, int(255 * (p['life'] / 10.0))))
        color = (*p['color'], alpha)
        radius = int(p['radius'] * (p['life'] / 10.0))
        if radius > 0:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)

    def _draw_cursor(self):
        pos_i = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        is_valid = self._is_valid_placement(self.cursor_pos)
        turret_under_cursor = self._get_turret_at(self.cursor_pos)
        
        if turret_under_cursor:
            color = self.COLOR_COG # Gold for upgrade
        elif is_valid:
            color = self.COLOR_CURSOR
        else:
            color = self.COLOR_INVALID_CURSOR
        
        length = 10
        pygame.draw.line(self.screen, color, (pos_i[0] - length, pos_i[1]), (pos_i[0] + length, pos_i[1]), 2)
        pygame.draw.line(self.screen, color, (pos_i[0], pos_i[1] - length), (pos_i[0], pos_i[1] + length), 2)

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': np.copy(pos),
                'vel': vel,
                'life': self.np_random.uniform(5, 15),
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })
            
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a display window
    game_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Steampunk Cog Defense")
    
    # Game loop for human play
    while not done:
        # Action mapping for human player
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Event handling to close the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
    
    env.close()