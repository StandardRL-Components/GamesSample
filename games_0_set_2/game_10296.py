import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = "Defend a central cell from waves of incoming pathogens. Fire different enzymes from portals and manipulate gravity to survive."
    user_guide = "Controls: Use arrow keys (↑↓←→) to select a portal. Press space to fire an enzyme. Press shift to reverse gravity."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CELL_RADIUS = 100
    CELL_CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    MAX_STEPS = 5000
    FPS = 30

    # --- Colors ---
    COLOR_BG = (10, 5, 30) # Dark blue/purple
    COLOR_CELL_HEALTHY = (0, 255, 150)
    COLOR_CELL_MID = (255, 255, 0)
    COLOR_CELL_LOW = (255, 50, 50)
    COLOR_PORTAL = (150, 150, 255)
    COLOR_PORTAL_SELECTED = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    
    ENZYME_SPECS = {
        1: {'color': (100, 150, 255), 'speed': 6, 'damage': 10, 'radius': 5, 'name': "LYSO-BLUE"},
        2: {'color': (255, 255, 100), 'speed': 9, 'damage': 7, 'radius': 4, 'name': "PHAGO-YELLOW"},
        3: {'color': (200, 100, 255), 'speed': 5, 'damage': 5, 'radius': 6, 'name': "PEROXI-PURPLE", 'aoe': 50}
    }
    
    PATHOGEN_SPECS = {
        'triangle': {'sides': 3, 'base_health': 10, 'color': (255, 100, 100)},
        'square': {'sides': 4, 'base_health': 20, 'color': (100, 255, 100)},
        'pentagon': {'sides': 5, 'base_health': 30, 'color': (255, 150, 50)},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        self.portals = self._initialize_portals()
        self.bg_stars = None # Initialized in reset
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cell_integrity = 0
        self.current_wave = 0
        self.pathogens = []
        self.enzymes = []
        self.particles = []
        self.gravity = 1
        self.selected_portal_idx = 0
        self.unlocked_enzyme_level = 1
        self.enzyme_cooldown = 0
        self.gravity_cooldown = 0
        self.damage_flash_timer = 0
        self.gravity_flash_timer = 0
        self.pathogen_base_speed = 0
        self.pathogen_base_health_multiplier = 0

    def _initialize_portals(self):
        cx, cy = self.CELL_CENTER
        r = self.CELL_RADIUS
        return [
            {'pos': pygame.Vector2(cx, cy - r), 'type': 1}, # Up
            {'pos': pygame.Vector2(cx, cy + r), 'type': 2}, # Down
            {'pos': pygame.Vector2(cx - r, cy), 'type': 3}, # Left
            {'pos': pygame.Vector2(cx + r, cy), 'type': 1}, # Right
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cell_integrity = 100
        self.current_wave = 0
        self.pathogens = []
        self.enzymes = []
        self.particles = []
        self.gravity = 1
        self.selected_portal_idx = 0
        self.unlocked_enzyme_level = 1
        self.enzyme_cooldown = 0
        self.gravity_cooldown = 0
        self.damage_flash_timer = 0
        self.gravity_flash_timer = 0
        
        self.bg_stars = self._create_bg_stars()
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held, shift_held)
        
        reward += self._update_enzymes()
        reward += self._update_pathogens()
        self._update_particles()
        self._update_cooldowns()

        if not self.pathogens:
            reward += 10 # Wave survival reward
            self._start_next_wave()

        terminated = self.cell_integrity <= 0 or self.steps >= self.MAX_STEPS
        if self.cell_integrity <= 0 and not self.game_over:
            reward = -100 # Game over penalty
            self.game_over = True
            self._create_particles(self.CELL_CENTER, 200, (255, 0, 0), 10, 120)

        truncated = self.steps >= self.MAX_STEPS

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        if movement in [1, 2, 3, 4]:
            self.selected_portal_idx = movement - 1

        if space_held and self.enzyme_cooldown <= 0:
            portal = self.portals[self.selected_portal_idx]
            if portal['type'] <= self.unlocked_enzyme_level:
                self._fire_enzyme(portal)
                self.enzyme_cooldown = 10 # 1/3 second cooldown

        if shift_held and self.gravity_cooldown <= 0:
            self.gravity *= -1
            self.gravity_cooldown = 60 # 2 second cooldown
            self.gravity_flash_timer = 10

    def _start_next_wave(self):
        self.current_wave += 1
        
        if (self.current_wave - 1) % 10 == 0 and self.current_wave > 1:
            if self.unlocked_enzyme_level < 3:
                self.unlocked_enzyme_level += 1
        
        self.pathogen_base_speed = 1.0 + (self.current_wave // 5) * 0.1
        self.pathogen_base_health_multiplier = 1.0 + (self.current_wave // 10) * 0.5
        
        num_pathogens = 2 + self.current_wave
        for _ in range(num_pathogens):
            self._spawn_pathogen()

    def _spawn_pathogen(self):
        ptype = random.choice(list(self.PATHOGEN_SPECS.keys()))
        spec = self.PATHOGEN_SPECS[ptype]
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        spawn_dist = self.SCREEN_WIDTH / 2 + 50
        pos = pygame.Vector2(
            self.CELL_CENTER[0] + spawn_dist * math.cos(angle),
            self.CELL_CENTER[1] + spawn_dist * math.sin(angle)
        )
        
        health = spec['base_health'] * self.pathogen_base_health_multiplier
        speed = self.pathogen_base_speed * self.np_random.uniform(0.8, 1.2)
        
        self.pathogens.append({
            'pos': pos,
            'type': ptype,
            'health': health,
            'max_health': health,
            'speed': speed,
            'size': 10 + spec['sides'],
            'hit_timer': 0
        })

    def _fire_enzyme(self, portal):
        spec = self.ENZYME_SPECS[portal['type']]
        direction = (portal['pos'] - pygame.Vector2(self.CELL_CENTER)).normalize()
        
        self.enzymes.append({
            'pos': pygame.Vector2(portal['pos']),
            'vel': direction * spec['speed'],
            'type': portal['type'],
            'trail': [pygame.Vector2(portal['pos']) for _ in range(10)]
        })

    def _update_cooldowns(self):
        if self.enzyme_cooldown > 0: self.enzyme_cooldown -= 1
        if self.gravity_cooldown > 0: self.gravity_cooldown -= 1
        if self.damage_flash_timer > 0: self.damage_flash_timer -= 1
        if self.gravity_flash_timer > 0: self.gravity_flash_timer -= 1

    def _update_enzymes(self):
        reward = 0
        for enzyme in self.enzymes[:]:
            enzyme['vel'].y += self.gravity * 0.1
            enzyme['pos'] += enzyme['vel']
            
            enzyme['trail'].pop(0)
            enzyme['trail'].append(pygame.Vector2(enzyme['pos']))

            if not self.screen.get_rect().inflate(50, 50).collidepoint(enzyme['pos']):
                self.enzymes.remove(enzyme)
                continue

            spec = self.ENZYME_SPECS[enzyme['type']]
            for pathogen in self.pathogens[:]:
                if enzyme['pos'].distance_to(pathogen['pos']) < pathogen['size'] + spec['radius']:
                    pathogen['health'] -= spec['damage']
                    pathogen['hit_timer'] = 5
                    reward += 0.1
                    self.score += 1

                    if spec.get('aoe'):
                        self._create_particles(enzyme['pos'], 30, spec['color'], 4, 30)
                        for other_pathogen in self.pathogens:
                            if other_pathogen is not pathogen and enzyme['pos'].distance_to(other_pathogen['pos']) < spec['aoe']:
                                other_pathogen['health'] -= spec['damage'] * 0.5
                                other_pathogen['hit_timer'] = 5
                                reward += 0.05
                                self.score += 1
                    
                    if enzyme in self.enzymes: self.enzymes.remove(enzyme)

                    if pathogen['health'] <= 0:
                        reward += 1
                        self.score += 10
                        self._create_particles(pathogen['pos'], 50, self.PATHOGEN_SPECS[pathogen['type']]['color'], 3, 60)
                        self.pathogens.remove(pathogen)
                    break
        return reward

    def _update_pathogens(self):
        reward = 0
        for p in self.pathogens[:]:
            if p['hit_timer'] > 0: p['hit_timer'] -= 1
            direction = (pygame.Vector2(self.CELL_CENTER) - p['pos']).normalize()
            p['pos'] += direction * p['speed']
            
            if p['pos'].distance_to(self.CELL_CENTER) < self.CELL_RADIUS + p['size']:
                self.cell_integrity -= 10
                reward -= 0.1
                self.damage_flash_timer = 10
                self._create_particles(p['pos'], 20, self.COLOR_CELL_LOW, 2, 20)
                self.pathogens.remove(p)
                if self.cell_integrity < 0: self.cell_integrity = 0
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color, speed_max, lifespan_max):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_max)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'lifespan': self.np_random.integers(lifespan_max // 2, lifespan_max + 1),
                'max_lifespan': lifespan_max,
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave}
    
    def _create_bg_stars(self):
        stars = []
        for _ in range(150):
            stars.append({
                'pos': (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)),
                'radius': self.np_random.uniform(0.5, 1.5),
                'color': (self.np_random.integers(20, 50), self.np_random.integers(15, 40), self.np_random.integers(40, 70))
            })
        return stars

    def _render_game(self):
        # Background stars
        if self.bg_stars:
            for star in self.bg_stars:
                pygame.draw.circle(self.screen, star['color'], star['pos'], star['radius'])

        # Gravity direction indicator
        indicator_y = 10 if self.gravity > 0 else self.SCREEN_HEIGHT - 10
        pygame.draw.line(self.screen, (50, 50, 80), (0, indicator_y), (self.SCREEN_WIDTH, indicator_y), 1)

        # Cell wall
        health_percent = self.cell_integrity / 100.0
        if health_percent > 0.66:
            cell_color = self.COLOR_CELL_HEALTHY
        elif health_percent > 0.33:
            cell_color = self.COLOR_CELL_MID
        else:
            cell_color = self.COLOR_CELL_LOW
        
        vibration = 0
        if self.damage_flash_timer > 0:
            vibration = self.np_random.integers(-2, 3)
            flash_color = (255, 100, 100)
            self._draw_glowing_circle(self.CELL_CENTER, self.CELL_RADIUS, flash_color, 10)

        pygame.draw.circle(self.screen, cell_color, self.CELL_CENTER, self.CELL_RADIUS + vibration, 3)
        self._draw_glowing_circle(self.CELL_CENTER, self.CELL_RADIUS + vibration, cell_color, 15)

        # Portals
        for i, portal in enumerate(self.portals):
            is_selected = (i == self.selected_portal_idx)
            is_unlocked = (portal['type'] <= self.unlocked_enzyme_level)
            
            color = self.COLOR_PORTAL_SELECTED if is_selected else self.COLOR_PORTAL
            if not is_unlocked:
                color = (50, 50, 70)
            
            self._draw_glowing_circle(portal['pos'], 8 if is_selected else 6, color, 10 if is_selected else 5)
            
            if is_unlocked:
                enzyme_color = self.ENZYME_SPECS[portal['type']]['color']
                pygame.draw.circle(self.screen, enzyme_color, portal['pos'], 3)

        # Enzymes
        for e in self.enzymes:
            spec = self.ENZYME_SPECS[e['type']]
            # Trail
            for i, pos in enumerate(e['trail']):
                alpha = int(255 * (i / len(e['trail'])))
                color = spec['color']
                radius = spec['radius'] * (i / len(e['trail']))
                if radius < 1: continue
                
                s_size = int(radius * 2)
                s = pygame.Surface((s_size, s_size), pygame.SRCALPHA)
                pygame.draw.circle(s, (*color, alpha), (radius, radius), radius)
                self.screen.blit(s, pos - pygame.Vector2(radius, radius), special_flags=pygame.BLEND_RGBA_ADD)

            # Head
            self._draw_glowing_circle(e['pos'], spec['radius'], spec['color'], 10)

        # Pathogens
        for p in self.pathogens:
            spec = self.PATHOGEN_SPECS[p['type']]
            color = spec['color']
            if p['hit_timer'] > 0: color = (255, 255, 255)
            self._draw_polygon_reg(p['pos'], spec['sides'], p['size'], color)
            self._draw_glowing_polygon_reg(p['pos'], spec['sides'], p['size'], color, 15)

        # Particles
        for p in self.particles:
            alpha = 255 * (p['lifespan'] / p['max_lifespan'])
            size = 2 * (p['lifespan'] / p['max_lifespan'])
            if size < 1: size = 1
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*p['color'], int(alpha)), (size, size), size)
            self.screen.blit(temp_surf, (p['pos'][0] - size, p['pos'][1] - size))


        # Gravity flip flash
        if self.gravity_flash_timer > 0:
            alpha = 150 * (self.gravity_flash_timer / 10)
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((200, 200, 255, int(alpha)))
            self.screen.blit(flash_surface, (0,0))

    def _render_ui(self):
        # Score
        score_surf = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Wave
        wave_surf = self.font_main.render(f"WAVE {self.current_wave}", True, self.COLOR_TEXT)
        self.screen.blit(wave_surf, (self.SCREEN_WIDTH/2 - wave_surf.get_width()/2, self.SCREEN_HEIGHT - 40))

        # Cell Integrity
        health_rect = pygame.Rect(self.SCREEN_WIDTH - 110, 10, 100, 20)
        health_percent = self.cell_integrity / 100.0
        
        if health_percent > 0.66: health_color = self.COLOR_CELL_HEALTHY
        elif health_percent > 0.33: health_color = self.COLOR_CELL_MID
        else: health_color = self.COLOR_CELL_LOW
        
        pygame.draw.rect(self.screen, (50, 50, 50), health_rect, 0, 5)
        if health_percent > 0:
            fill_rect = pygame.Rect(health_rect.left, health_rect.top, health_rect.width * health_percent, health_rect.height)
            pygame.draw.rect(self.screen, health_color, fill_rect, 0, 5)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, health_rect, 1, 5)

        # Enzyme selection UI
        selected_portal = self.portals[self.selected_portal_idx]
        if selected_portal['type'] <= self.unlocked_enzyme_level:
            enzyme_name = self.ENZYME_SPECS[selected_portal['type']]['name']
            enzyme_color = self.ENZYME_SPECS[selected_portal['type']]['color']
        else:
            enzyme_name = "LOCKED"
            enzyme_color = (100, 100, 100)
        
        enzyme_surf = self.font_small.render(enzyme_name, True, enzyme_color)
        self.screen.blit(enzyme_surf, (self.SCREEN_WIDTH/2 - enzyme_surf.get_width()/2, 10))

    def _draw_glowing_circle(self, pos, radius, color, glow_size):
        if radius <= 0: return
        surf = pygame.Surface((radius * 2 + glow_size, radius * 2 + glow_size), pygame.SRCALPHA)
        center = (surf.get_width() // 2, surf.get_height() // 2)
        
        # Glow
        pygame.gfxdraw.filled_circle(surf, center[0], center[1], int(radius + glow_size/2), (*color, 20))
        pygame.gfxdraw.filled_circle(surf, center[0], center[1], int(radius + glow_size/4), (*color, 40))
        
        # Solid circle
        pygame.gfxdraw.filled_circle(surf, center[0], center[1], int(radius), color)
        pygame.gfxdraw.aacircle(surf, center[0], center[1], int(radius), color)

        self.screen.blit(surf, (int(pos[0] - center[0]), int(pos[1] - center[1])), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_polygon_reg(self, pos, n, r, color):
        if r <= 0: return
        points = []
        for i in range(n):
            angle = i * (2 * math.pi / n) + (math.pi / n if n % 2 == 0 else 0)
            points.append((pos[0] + r * math.sin(angle), pos[1] + r * math.cos(angle)))
        if len(points) > 2:
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _draw_glowing_polygon_reg(self, pos, n, r, color, glow_size):
        if r <= 0: return
        points = []
        for i in range(n):
            angle = i * (2 * math.pi / n) + (math.pi / n if n % 2 == 0 else 0)
            points.append((pos[0] + (r + glow_size) * math.sin(angle), pos[1] + (r + glow_size) * math.cos(angle)))
        if len(points) > 2:
            pygame.gfxdraw.filled_polygon(self.screen, points, (*color, 20))
            
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Re-initialize pygame with a visible display for manual play
    pygame.display.quit()
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Lysosome Defender")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Wave: {info['wave']}")
            
    env.close()