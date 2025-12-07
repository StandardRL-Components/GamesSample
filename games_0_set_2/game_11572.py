import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:10:59.875302
# Source Brief: brief_01572.md
# Brief Index: 1572
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player defends a base against fractal enemies
    by deploying units based on colors from a Mandelbrot set visualization.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your base from waves of fractal enemies by deploying units and unleashing powerful gem abilities."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to select a quadrant. Press space to deploy a defender. Press shift to activate a gem power."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 2000
    MAX_WAVES = 5
    BASE_START_HEALTH = 100
    FONT_SIZE_UI = 20
    FONT_SIZE_MSG = 40

    # Colors (Dark Psychedelic Theme)
    COLOR_BG = (10, 5, 20)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_HEALTH_BAR_BG = (100, 20, 20)
    COLOR_HEALTH_BAR_FG = (200, 40, 40)
    COLOR_BASE = (150, 150, 255)
    COLOR_BASE_GLOW = (80, 80, 200)
    COLOR_ENEMY = (255, 100, 100)
    COLOR_ENEMY_GLOW = (180, 50, 50)
    COLOR_SELECTION = (255, 255, 100, 150) # RGBA

    # Unit/Region Colors
    REGION_COLORS = [
        (0, 255, 255),   # Cyan
        (255, 0, 255),   # Magenta
        (255, 255, 0),   # Yellow
        (0, 255, 0),     # Green
    ]

    # Gem Attack Types
    GEM_TYPES = {
        "NOVA": {"color": (255, 165, 0), "cooldown": 300, "unlock_score": 0},
        "FROST": {"color": (100, 100, 255), "cooldown": 450, "unlock_score": 1000},
        "BEAM": {"color": (255, 255, 255), "cooldown": 600, "unlock_score": 2500},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.SysFont("monospace", self.FONT_SIZE_UI, bold=True)
        self.msg_font = pygame.font.SysFont("monospace", self.FONT_SIZE_MSG, bold=True)

        # Mandelbrot background generation
        self._mandelbrot_surface = self._generate_mandelbrot_surface()
        
        # Quadrant definitions for selection
        self.quadrant_rects = [
            pygame.Rect(0, 0, self.WIDTH // 2, self.HEIGHT // 2),
            pygame.Rect(self.WIDTH // 2, 0, self.WIDTH // 2, self.HEIGHT // 2),
            pygame.Rect(0, self.HEIGHT // 2, self.WIDTH // 2, self.HEIGHT // 2),
            pygame.Rect(self.WIDTH // 2, self.HEIGHT // 2, self.WIDTH // 2, self.HEIGHT // 2),
        ]

        # Initialize state variables
        self._init_game_state()

    def _init_game_state(self):
        """Initializes all mutable game state variables."""
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.reward = 0.0

        self.base_health = self.BASE_START_HEALTH
        self.base_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT - 30)

        self.defenders = []
        self.enemies = []
        self.particles = []

        self.selected_quadrant = 0 # 0=none, 1-4 for quadrants
        self.deployable_units = {i: 10 for i in range(4)}

        self.unlocked_gems = []
        self.selected_gem_idx = 0
        self.gem_cooldowns = {name: 0 for name in self.GEM_TYPES}
        
        self.wave_number = 1
        self.enemies_per_wave = 5
        self.enemies_spawned_this_wave = 0
        self.enemies_defeated_this_wave = 0
        
        self.base_enemy_spawn_rate = 90 # Ticks between spawns
        self.enemy_spawn_timer = self.base_enemy_spawn_rate
        self.base_enemy_health = 10
        self.enemy_speed = 0.8
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_game_state()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward = 0.0
        self.steps += 1
        
        # --- 1. Handle Input ---
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_press, shift_press)

        # --- 2. Update Game Logic ---
        self._update_enemies()
        self._update_defenders()
        self._update_particles()
        self._update_cooldowns()
        self._handle_collisions()
        self._update_progression()

        # --- 3. Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
        
        # --- 4. Return Gym Tuple ---
        return (
            self._get_observation(),
            self.reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Core Logic Sub-functions ---

    def _handle_input(self, movement, space_press, shift_press):
        # Action 0: Movement (Quadrant Selection)
        if 1 <= movement <= 4:
            self.selected_quadrant = movement
        else:
            self.selected_quadrant = 0 # No quadrant selected

        # Action 1: Space (Deploy Defender)
        if space_press and self.selected_quadrant != 0:
            quad_idx = self.selected_quadrant - 1
            if self.deployable_units[quad_idx] > 0:
                self.deployable_units[quad_idx] -= 1
                self._spawn_defender(quad_idx)
                # sfx: deploy_unit.wav

        # Action 2: Shift (Activate Gem)
        if shift_press and self.unlocked_gems:
            gem_name = self.unlocked_gems[self.selected_gem_idx]
            if self.gem_cooldowns[gem_name] == 0:
                self._activate_gem(gem_name)
                self.gem_cooldowns[gem_name] = self.GEM_TYPES[gem_name]["cooldown"]
                self.selected_gem_idx = (self.selected_gem_idx + 1) % len(self.unlocked_gems)
                self.reward += 10.0
                # sfx: gem_activate.wav

    def _update_enemies(self):
        # Spawn new enemies if needed
        current_spawn_rate = self.base_enemy_spawn_rate - (self.steps // 200) * 0.05 * self.base_enemy_spawn_rate
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0 and self.enemies_spawned_this_wave < self.enemies_per_wave:
            self._spawn_enemy()
            self.enemy_spawn_timer = max(20, current_spawn_rate)
        
        # Move existing enemies
        for enemy in self.enemies:
            direction = (self.base_pos - enemy['pos']).normalize()
            enemy['pos'] += direction * self.enemy_speed
            enemy['angle'] = math.atan2(direction.y, direction.x)

    def _update_defenders(self):
        for defender in self.defenders:
            # Simple AI: move towards top-center
            target = pygame.Vector2(self.WIDTH / 2, 0)
            direction = (target - defender['pos']).normalize()
            defender['pos'] += direction * 2.0
        
        # Remove defenders that go off-screen
        self.defenders = [d for d in self.defenders if d['pos'].y > -20]

    def _handle_collisions(self):
        for defender in self.defenders[:]:
            for enemy in self.enemies[:]:
                if defender['pos'].distance_to(enemy['pos']) < 15: # Collision radius
                    enemy['health'] -= 1
                    self.reward += 0.1
                    self._create_particles(defender['pos'], defender['color'], 5, 1.5)
                    # sfx: hit_damage.wav
                    
                    if enemy['health'] <= 0:
                        self.score += 10
                        self.reward += 5.0
                        self.enemies_defeated_this_wave += 1
                        self._create_particles(enemy['pos'], self.COLOR_ENEMY, 20, 3.0)
                        self.enemies.remove(enemy)
                        # sfx: enemy_destroyed.wav
                    
                    if defender in self.defenders:
                        self.defenders.remove(defender)
                    break # Defender is consumed on hit
        
        for enemy in self.enemies[:]:
            if enemy['pos'].distance_to(self.base_pos) < 25:
                self.base_health -= 5
                self.reward -= 2.5 # 5 * 0.5
                self._create_particles(self.base_pos, self.COLOR_HEALTH_BAR_FG, 30, 4.0)
                self.enemies.remove(enemy)
                # sfx: base_damage.wav

    def _update_progression(self):
        # Check for wave clear
        if self.enemies_defeated_this_wave >= self.enemies_per_wave:
            self.wave_number += 1
            if self.wave_number > self.MAX_WAVES:
                self.game_won = True
                self.reward += 100.0
            else:
                self.reward += 50.0
                self.enemies_per_wave += 2 # Increase difficulty
                self.enemies_spawned_this_wave = 0
                self.enemies_defeated_this_wave = 0
                self.enemy_speed += 0.1
                # Replenish some deployable units
                for i in range(4):
                    self.deployable_units[i] = min(20, self.deployable_units[i] + 5)

        # Unlock gems based on score
        for name, data in self.GEM_TYPES.items():
            if self.score >= data['unlock_score'] and name not in self.unlocked_gems:
                self.unlocked_gems.append(name)

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            self.reward -= 100.0 # Penalty for losing
            return True
        if self.game_won:
            self.game_over = True
            return True
        return False

    # --- Spawning Functions ---

    def _spawn_defender(self, quad_idx):
        color = self.REGION_COLORS[quad_idx]
        defender = {
            'pos': self.base_pos.copy(),
            'color': color,
        }
        self.defenders.append(defender)
        self._create_particles(self.base_pos, color, 15, 2.5)

    def _spawn_enemy(self):
        health = self.base_enemy_health + (self.steps // 500)
        enemy = {
            'pos': pygame.Vector2(random.uniform(50, self.WIDTH - 50), -20),
            'health': health,
            'max_health': health,
            'angle': 0
        }
        self.enemies.append(enemy)
        self.enemies_spawned_this_wave += 1

    def _activate_gem(self, gem_name):
        if gem_name == "NOVA":
            # Damage all enemies on screen
            for enemy in self.enemies:
                enemy['health'] -= 5
                self.reward += 0.5 # 5 * 0.1
            self._create_particles(self.base_pos, self.GEM_TYPES["NOVA"]["color"], 100, 5, is_nova=True)
        elif gem_name == "FROST":
            # Slow all enemies for a duration (visual effect)
            for enemy in self.enemies:
                enemy['slowed_timer'] = 180 # 6 seconds at 30fps
            self._create_particles(pygame.Vector2(self.WIDTH/2, self.HEIGHT/2), self.GEM_TYPES["FROST"]["color"], 80, 2, is_frost=True)
        elif gem_name == "BEAM":
            # Damage enemies in a line
            if self.enemies:
                target_enemy = random.choice(self.enemies)
                self._create_beam_effect(self.base_pos, target_enemy['pos'], self.GEM_TYPES["BEAM"]["color"])
                for enemy in self.enemies:
                    if self._is_point_on_line(enemy['pos'], self.base_pos, target_enemy['pos']):
                        enemy['health'] -= 15
                        self.reward += 1.5 # 15 * 0.1

    # --- Particle & Effects System ---

    def _create_particles(self, pos, color, count, max_speed, is_nova=False, is_frost=False):
        for _ in range(count):
            if is_nova:
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 10)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            elif is_frost:
                 vel = pygame.Vector2(random.uniform(-2, 2), random.uniform(-2, 2))
            else:
                vel = pygame.Vector2(random.uniform(-max_speed, max_speed), random.uniform(-max_speed, max_speed))
            
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': random.randint(20, 40),
                'max_lifespan': 40,
                'color': color,
                'is_frost': is_frost
            })

    def _create_beam_effect(self, start_pos, end_pos, color):
        self.particles.append({
            'pos': start_pos, 'end_pos': end_pos, 'color': color,
            'lifespan': 15, 'max_lifespan': 15, 'type': 'beam'
        })
    
    def _is_point_on_line(self, pt, start, end, tolerance=15):
        # Check if point is near the line segment
        d_line = (end - start).length()
        if d_line == 0: return (pt - start).length() < tolerance
        d1 = (pt - start).length()
        d2 = (pt - end).length()
        return d1 + d2 >= d_line - tolerance and d1 + d2 <= d_line + tolerance

    def _update_particles(self):
        for p in self.particles[:]:
            p['lifespan'] -= 1
            if p.get('type') != 'beam':
                p['pos'] += p['vel']
                if p.get('is_frost'): # Frost particles slow down
                    p['vel'] *= 0.95
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _update_cooldowns(self):
        for name in self.gem_cooldowns:
            if self.gem_cooldowns[name] > 0:
                self.gem_cooldowns[name] -= 1

    # --- Rendering Functions ---

    def _get_observation(self):
        self.screen.blit(self._mandelbrot_surface, (0, 0))
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_end_screen()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Highlight selected quadrant
        if self.selected_quadrant != 0:
            s = pygame.Surface((self.WIDTH / 2, self.HEIGHT / 2), pygame.SRCALPHA)
            s.fill(self.COLOR_SELECTION)
            self.screen.blit(s, self.quadrant_rects[self.selected_quadrant - 1].topleft)

        # Draw base
        self._draw_glow_circle(self.base_pos, 20, self.COLOR_BASE, self.COLOR_BASE_GLOW)

        # Draw defenders
        for d in self.defenders:
            self._draw_glow_circle(d['pos'], 7, d['color'], tuple(c*0.5 for c in d['color']))

        # Draw enemies
        for e in self.enemies:
            self._draw_fractal_enemy(e['pos'], 12, e['angle'])
            # Health bar for enemies
            bar_width = 20
            health_pct = max(0, e['health'] / e['max_health'])
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (e['pos'].x - bar_width/2, e['pos'].y - 20, bar_width, 3))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (e['pos'].x - bar_width/2, e['pos'].y - 20, bar_width * health_pct, 3))

        # Draw particles
        for p in self.particles:
            if p.get('type') == 'beam':
                alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
                try:
                    # Pygame 2.x requires tuple for color with alpha
                    color_with_alpha = (*p['color'], alpha)
                except TypeError:
                    # Pygame 1.x might not accept tuple for color
                    color_with_alpha = p['color']
                pygame.draw.line(self.screen, color_with_alpha, p['pos'], p['end_pos'], width=5)
            else:
                alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
                color = (*p['color'], alpha)
                size = int(5 * (p['lifespan'] / p['max_lifespan']))
                if size > 0:
                    temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, color, (size, size), size)
                    self.screen.blit(temp_surf, (int(p['pos'].x - size), int(p['pos'].y - size)))

    def _render_ui(self):
        # Score and Wave
        score_text = self.ui_font.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        wave_text = self.ui_font.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Base Health Bar
        health_pct = max(0, self.base_health / self.BASE_START_HEALTH)
        bar_width, bar_height = 200, 15
        bar_x, bar_y = self.WIDTH/2 - bar_width/2, self.HEIGHT - bar_height - 5
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (bar_x, bar_y, bar_width * health_pct, bar_height), border_radius=3)

        # Deployable Units UI
        for i in range(4):
            color = self.REGION_COLORS[i]
            count = self.deployable_units[i]
            pos_x = 20 + i * 50
            pos_y = self.HEIGHT - 40
            pygame.draw.circle(self.screen, color, (pos_x, pos_y), 10)
            count_text = self.ui_font.render(str(count), True, self.COLOR_UI_TEXT)
            self.screen.blit(count_text, (pos_x + 15, pos_y - 10))
        
        # Gem UI
        if self.unlocked_gems:
            for i, gem_name in enumerate(self.unlocked_gems):
                gem_data = self.GEM_TYPES[gem_name]
                pos_x = self.WIDTH - 20 - (len(self.unlocked_gems) - 1 - i) * 40
                pos_y = self.HEIGHT - 30
                
                # Draw gem icon
                pygame.draw.rect(self.screen, gem_data['color'], (pos_x - 10, pos_y - 10, 20, 20), border_radius=4)
                
                # Cooldown overlay
                cooldown_pct = self.gem_cooldowns[gem_name] / gem_data['cooldown']
                if cooldown_pct > 0:
                    overlay_surf = pygame.Surface((20, 20), pygame.SRCALPHA)
                    overlay_surf.fill((0, 0, 0, 180))
                    pygame.draw.rect(overlay_surf, (0,0,0,0), (0, 0, 20, 20 * (1 - cooldown_pct)))
                    self.screen.blit(overlay_surf, (pos_x - 10, pos_y - 10))

                # Selection highlight
                if i == self.selected_gem_idx:
                    pygame.draw.rect(self.screen, (255, 255, 255), (pos_x - 12, pos_y - 12, 24, 24), 2, border_radius=5)

    def _render_end_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        msg = "LEVEL COMPLETE" if self.game_won else "GAME OVER"
        text = self.msg_font.render(msg, True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text, text_rect)

    # --- Helper & Utility Functions ---

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "base_health": self.base_health, "wave": self.wave_number}

    def _draw_glow_circle(self, pos, radius, color, glow_color):
        # Draw multiple transparent circles to create a glow effect
        for i in range(radius, 0, -2):
            alpha = 100 * (1 - i / radius)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius + i*0.5), (*glow_color, int(alpha)))
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), radius, color)
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, color)

    def _draw_fractal_enemy(self, pos, size, angle):
        points = []
        for i in range(5):
            rad = angle + i * 2 * math.pi / 5
            outer_point = (pos.x + math.cos(rad) * size, pos.y + math.sin(rad) * size)
            
            rad_inner = rad + math.pi / 5
            inner_point = (pos.x + math.cos(rad_inner) * size * 0.5, pos.y + math.sin(rad_inner) * size * 0.5)
            points.append(outer_point)
            points.append(inner_point)
        
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_ENEMY_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_ENEMY)

    def _generate_mandelbrot_surface(self):
        surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        max_iter = 60
        
        # Zoomed-in area of the Mandelbrot set
        re_start, re_end = -0.74877, -0.74872
        im_start, im_end = 0.06505, 0.06510

        palette = [
            (int(r*0.5), int(g*0.5), int(b*0.5)) for r,g,b in self.REGION_COLORS
        ]
        
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                c = complex(re_start + (x / self.WIDTH) * (re_end - re_start),
                            im_start + (y / self.HEIGHT) * (im_end - im_start))
                z = 0
                n = 0
                while abs(z) <= 2 and n < max_iter:
                    z = z*z + c
                    n += 1
                
                if n < max_iter:
                    # Color based on iteration count, creating bands
                    color_idx = (n // 4) % len(palette)
                    color = palette[color_idx]
                    surface.set_at((x, y), color)
                else:
                    surface.set_at((x, y), self.COLOR_BG)
        return surface

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Mandelbrot Defender")
    clock = pygame.time.Clock()

    selected_quadrant = 0
    
    while not done:
        movement = 0
        space_press = 0
        shift_press = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    space_press = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_press = 1

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] and keys[pygame.K_LEFT]: selected_quadrant = 1
        elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]: selected_quadrant = 2
        elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]: selected_quadrant = 3
        elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]: selected_quadrant = 4
        elif keys[pygame.K_UP]: selected_quadrant = 1 # Default to a region
        elif keys[pygame.K_DOWN]: selected_quadrant = 3
        elif keys[pygame.K_LEFT]: selected_quadrant = 1
        elif keys[pygame.K_RIGHT]: selected_quadrant = 2
        else: selected_quadrant = 0

        action = [selected_quadrant, space_press, shift_press]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Run at 30 FPS

    env.close()