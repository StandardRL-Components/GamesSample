import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()
pygame.font.init()

# Generated: 2025-08-26T17:16:00.682854
# Source Brief: brief_02067.md
# Brief Index: 2067
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player defends a central mining base from waves of space pirates.
    The player manages resources mined from nearby planets to build defensive turrets and upgrade
    their firepower. The goal is to survive as long as possible.

    Action Space: MultiDiscrete([5, 2, 2])
    - a[0]: Cursor Movement (0: none, 1: up, 2: down, 3: left, 4: right)
    - a[1]: Action Button (Space) (0: released, 1: held)
    - a[2]: Mode Switch (Shift) (0: released, 1: held)

    Observation Space: Box(0, 255, (400, 640, 3), uint8) - A rendered frame of the game.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your central mining base from space pirates. Mine resources from planets to build and upgrade defensive turrets."
    )
    user_guide = (
        "Use arrow keys to select a planet. Press space to perform an action (mine, build, upgrade). Press shift to cycle between modes."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000
    WAVE_INTERVAL = 500

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_STAR = (200, 200, 220)
    COLOR_PLAYER_BASE = (0, 200, 150)
    COLOR_PLAYER_BASE_DMG = (255, 100, 100)
    COLOR_PLANET = (100, 80, 150)
    COLOR_PLANET_DEPLETED = (50, 40, 75)
    COLOR_PIRATE = (255, 50, 50)
    COLOR_TURRET_BASE = (150, 150, 150)
    COLOR_TURRET_BARREL = (200, 200, 200)
    COLOR_PROJECTILE = (255, 255, 0)
    COLOR_MINING_LASER = (50, 150, 255, 150) # RGBA
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_RESOURCE = (255, 200, 0)

    # Game Parameters
    BASE_STARTING_HEALTH = 100
    PLANET_COUNT = 7
    PLANET_MIN_RESOURCES = 500
    PLANET_MAX_RESOURCES = 1500
    MINING_RATE = 0.5
    TURRET_COST = 100
    TURRET_RANGE = 150
    TURRET_COOLDOWN = 60 # frames
    UPGRADE_COST_BASE = 150
    PIRATE_STARTING_HEALTH = 50
    PIRATE_STARTING_SPEED = 0.75
    PIRATE_DAMAGE = 10

    # Modes
    MODES = ["MINE", "BUILD", "UPGRADE"]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # Initialize state variables to None, they will be set in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.base_pos = None
        self.base_health = None
        self.resources = None
        self.cannon_level = None
        self.planets = None
        self.turrets = None
        self.pirates = None
        self.projectiles = None
        self.particles = None
        self.stars = None
        self.selected_planet_idx = None
        self.current_mode = None
        self.last_space_held = None
        self.last_shift_held = None
        self.wave = None
        self.step_reward = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.step_reward = 0
        self.game_over = False

        self.base_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float64)
        self.base_health = self.BASE_STARTING_HEALTH
        self.resources = 0
        self.cannon_level = 1

        self._generate_stars()
        self._generate_planets()

        self.turrets = []
        self.pirates = []
        self.projectiles = []
        self.particles = []

        self.selected_planet_idx = 0
        self.current_mode = 0  # 0: MINE, 1: BUILD, 2: UPGRADE
        self.last_space_held = False
        self.last_shift_held = False

        self.wave = 0
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.step_reward = 0
        self.steps += 1

        self._handle_input(action)
        self._update_game_state()

        terminated = self._check_termination()
        reward = self.step_reward

        if terminated and self.base_health > 0: # Survived
            reward += 100
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Mode Switching (on press) ---
        if shift_held and not self.last_shift_held:
            self.current_mode = (self.current_mode + 1) % len(self.MODES)
            # SFX: UI_SWITCH_MODE

        # --- Cursor Movement ---
        if movement != 0:
            self._move_cursor(movement)

        # --- Action Button (on press) ---
        if space_held and not self.last_space_held:
            mode_name = self.MODES[self.current_mode]
            selected_planet = self.planets[self.selected_planet_idx]

            if mode_name == "MINE":
                if selected_planet['resources'] > 0:
                    selected_planet['is_mining'] = not selected_planet['is_mining']
                    # SFX: MINE_TOGGLE_ON / MINE_TOGGLE_OFF

            elif mode_name == "BUILD":
                if not selected_planet['has_turret'] and self.resources >= self.TURRET_COST:
                    self.resources -= self.TURRET_COST
                    selected_planet['has_turret'] = True
                    self.turrets.append({
                        'pos': selected_planet['pos'],
                        'angle': 0.0,
                        'cooldown': 0,
                        'target': None
                    })
                    # SFX: BUILD_TURRET

            elif mode_name == "UPGRADE":
                cost = self.UPGRADE_COST_BASE * self.cannon_level
                if self.resources >= cost:
                    self.resources -= cost
                    self.cannon_level += 1
                    # SFX: UPGRADE_SUCCESS

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_game_state(self):
        # --- Mining ---
        for planet in self.planets:
            if planet['is_mining'] and planet['resources'] > 0:
                mined = self.MINING_RATE
                planet['resources'] -= mined
                self.resources += mined
                self.step_reward += 0.1 * mined
                if planet['resources'] <= 0:
                    planet['is_mining'] = False
                    planet['resources'] = 0

        # --- Turrets ---
        for turret in self.turrets:
            turret['cooldown'] = max(0, turret['cooldown'] - 1)
            # Target aquisition
            if not self.pirates:
                turret['target'] = None
            else:
                in_range_pirates = [p for p in self.pirates if np.linalg.norm(p['pos'] - turret['pos']) < self.TURRET_RANGE]
                if in_range_pirates:
                    closest_pirate = min(in_range_pirates, key=lambda p: np.linalg.norm(p['pos'] - turret['pos']))
                    turret['target'] = closest_pirate
                    
                    # Aiming
                    target_angle = math.atan2(closest_pirate['pos'][1] - turret['pos'][1], closest_pirate['pos'][0] - turret['pos'][0])
                    # Smooth rotation
                    turret['angle'] = self._lerp_angle(turret['angle'], target_angle, 0.2)

                    # Firing
                    if turret['cooldown'] == 0:
                        self.projectiles.append({
                            'pos': turret['pos'].copy(),
                            'vel': np.array([math.cos(turret['angle']), math.sin(turret['angle'])], dtype=np.float64) * 5,
                            'damage': 10 * self.cannon_level
                        })
                        turret['cooldown'] = self.TURRET_COOLDOWN
                        # SFX: TURRET_FIRE
                else:
                    turret['target'] = None

        # --- Projectiles ---
        new_projectiles = []
        for proj in self.projectiles:
            proj['pos'] += proj['vel']
            # Check for hits
            hit = False
            for pirate in self.pirates:
                if np.linalg.norm(proj['pos'] - pirate['pos']) < 10:
                    pirate['health'] -= proj['damage']
                    self._create_explosion(proj['pos'], 5, self.COLOR_PROJECTILE)
                    hit = True
                    # SFX: PIRATE_HIT
                    break
            # Boundary check
            if not hit and 0 < proj['pos'][0] < self.WIDTH and 0 < proj['pos'][1] < self.HEIGHT:
                new_projectiles.append(proj)
        self.projectiles = new_projectiles

        # --- Pirates ---
        new_pirates = []
        for pirate in self.pirates:
            if pirate['health'] <= 0:
                self.step_reward += 1
                self._create_explosion(pirate['pos'], 20, self.COLOR_PIRATE)
                # SFX: PIRATE_EXPLODE
                continue
            
            # Movement
            direction = (self.base_pos - pirate['pos'])
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction = direction / dist
            
            pirate['pos'] += direction * pirate['speed']
            pirate['angle'] = math.atan2(direction[1], direction[0])

            # Check if reached base
            if dist < 15:
                self.base_health -= self.PIRATE_DAMAGE
                self.step_reward -= 0.1 * self.PIRATE_DAMAGE
                self._create_explosion(pirate['pos'], 15, self.COLOR_PLAYER_BASE_DMG)
                # SFX: BASE_TAKE_DAMAGE
            else:
                new_pirates.append(pirate)
        self.pirates = new_pirates

        # --- Particles ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # --- Waves ---
        if self.steps > 0 and self.steps % self.WAVE_INTERVAL == 0:
            self.wave += 1
            self._spawn_wave()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "resources": self.resources,
            "base_health": self.base_health,
            "wave": self.wave,
            "cannon_level": self.cannon_level,
            "pirates_remaining": len(self.pirates)
        }
    
    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    # --- Generation and Spawning ---

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append((
                random.randint(0, self.WIDTH),
                random.randint(0, self.HEIGHT),
                random.choice([1, 2, 2])
            ))

    def _generate_planets(self):
        self.planets = []
        min_dist_from_base = 80
        min_dist_between_planets = 60
        
        while len(self.planets) < self.PLANET_COUNT:
            pos = np.array([random.uniform(40, self.WIDTH - 40), random.uniform(40, self.HEIGHT - 40)], dtype=np.float64)
            radius = random.randint(15, 25)
            
            # Check distance from base
            if np.linalg.norm(pos - self.base_pos) < min_dist_from_base + radius:
                continue

            # Check distance from other planets
            too_close = False
            for p in self.planets:
                if np.linalg.norm(pos - p['pos']) < min_dist_between_planets + radius + p['radius']:
                    too_close = True
                    break
            if too_close:
                continue

            self.planets.append({
                'pos': pos,
                'radius': radius,
                'resources': random.randint(self.PLANET_MIN_RESOURCES, self.PLANET_MAX_RESOURCES),
                'has_turret': False,
                'is_mining': False
            })

    def _spawn_wave(self):
        num_pirates = 3 + self.wave
        speed = self.PIRATE_STARTING_SPEED + 0.05 * self.wave
        health = self.PIRATE_STARTING_HEALTH + 10 * self.wave

        for _ in range(num_pirates):
            edge = random.randint(0, 3)
            if edge == 0: # Top
                pos = np.array([random.uniform(0, self.WIDTH), -20.0], dtype=np.float64)
            elif edge == 1: # Bottom
                pos = np.array([random.uniform(0, self.WIDTH), self.HEIGHT + 20.0], dtype=np.float64)
            elif edge == 2: # Left
                pos = np.array([-20.0, random.uniform(0, self.HEIGHT)], dtype=np.float64)
            else: # Right
                pos = np.array([self.WIDTH + 20.0, random.uniform(0, self.HEIGHT)], dtype=np.float64)

            self.pirates.append({
                'pos': pos,
                'health': health,
                'max_health': health,
                'speed': speed,
                'angle': 0
            })
    
    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': random.randint(10, 20),
                'color': color,
                'size': random.randint(2, 4)
            })

    # --- Cursor Logic ---
    
    def _move_cursor(self, direction):
        current_pos = self.planets[self.selected_planet_idx]['pos']
        best_planet_idx = -1
        min_score = float('inf')

        for i, planet in enumerate(self.planets):
            if i == self.selected_planet_idx:
                continue
            
            diff = planet['pos'] - current_pos
            dist = np.linalg.norm(diff)
            if dist == 0: continue

            # Direction: 1=up, 2=down, 3=left, 4=right
            if direction == 1 and diff[1] < 0: # Up
                score = -diff[1] + abs(diff[0]) * 0.5
            elif direction == 2 and diff[1] > 0: # Down
                score = diff[1] + abs(diff[0]) * 0.5
            elif direction == 3 and diff[0] < 0: # Left
                score = -diff[0] + abs(diff[1]) * 0.5
            elif direction == 4 and diff[0] > 0: # Right
                score = diff[0] + abs(diff[1]) * 0.5
            else:
                continue
            
            if score < min_score:
                min_score = score
                best_planet_idx = i

        if best_planet_idx != -1:
            self.selected_planet_idx = best_planet_idx

    # --- Rendering ---

    def _render_game(self):
        # Background Stars
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, self.COLOR_STAR, (x, y, size, size))

        # Mining Lasers
        for planet in self.planets:
            if planet['is_mining']:
                # Pulsating alpha effect
                alpha = 180 + 60 * math.sin(self.steps * 0.2)
                color = (*self.COLOR_MINING_LASER[:3], alpha)
                pygame.draw.line(self.screen, color, self.base_pos, planet['pos'], 2)

        # Planets
        for planet in self.planets:
            color = self.COLOR_PLANET if planet['resources'] > 0 else self.COLOR_PLANET_DEPLETED
            pos = tuple(map(int, planet['pos']))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], planet['radius'], color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], planet['radius'], tuple(int(c*0.7) for c in color))

        # Base
        base_color = self._lerp_color(self.COLOR_PLAYER_BASE_DMG, self.COLOR_PLAYER_BASE, self.base_health / self.BASE_STARTING_HEALTH)
        pygame.gfxdraw.filled_circle(self.screen, int(self.base_pos[0]), int(self.base_pos[1]), 15, base_color)
        pygame.gfxdraw.aacircle(self.screen, int(self.base_pos[0]), int(self.base_pos[1]), 15, tuple(int(c*0.7) for c in base_color))
        
        # Turrets
        for turret in self.turrets:
            pos = tuple(map(int, turret['pos']))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_TURRET_BASE)
            end_x = pos[0] + 12 * math.cos(turret['angle'])
            end_y = pos[1] + 12 * math.sin(turret['angle'])
            pygame.draw.line(self.screen, self.COLOR_TURRET_BARREL, pos, (end_x, end_y), 4)

        # Projectiles
        for proj in self.projectiles:
            pos = tuple(map(int, proj['pos']))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

        # Pirates
        for pirate in self.pirates:
            self._draw_rotated_triangle(pirate['pos'], pirate['angle'], self.COLOR_PIRATE)
            # Health bar
            health_percent = pirate['health'] / pirate['max_health']
            bar_pos = pirate['pos'] - np.array([10, 10])
            pygame.draw.rect(self.screen, (50,0,0), (*bar_pos, 20, 3))
            pygame.draw.rect(self.screen, self.COLOR_PIRATE, (*bar_pos, 20 * health_percent, 3))

        # Particles
        for p in self.particles:
            alpha = p['life'] / 20.0
            color = (*p['color'], int(alpha * 255))
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, tuple(map(int, p['pos'] - p['size'])))

        # Cursor
        selected_planet = self.planets[self.selected_planet_idx]
        cursor_radius = selected_planet['radius'] + 5 + 2 * math.sin(self.steps * 0.3)
        pygame.gfxdraw.aacircle(self.screen, int(selected_planet['pos'][0]), int(selected_planet['pos'][1]), int(cursor_radius), self.COLOR_CURSOR)

    def _render_ui(self):
        # Resources
        res_text = self.font_main.render(f"RESOURCES: {int(self.resources)}", True, self.COLOR_RESOURCE)
        self.screen.blit(res_text, (10, 10))
        
        # Base Health
        health_text = self.font_main.render(f"BASE HEALTH: {int(self.base_health)}%", True, self.COLOR_PLAYER_BASE)
        self.screen.blit(health_text, (10, 35))

        # Wave Info
        wave_text = self.font_main.render(f"WAVE: {self.wave + 1}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Mode Info
        mode_str = self.MODES[self.current_mode]
        mode_text = self.font_main.render(f"MODE: {mode_str}", True, self.COLOR_TEXT)
        self.screen.blit(mode_text, (self.WIDTH - mode_text.get_width() - 10, 35))
        
        # Upgrade Level
        if self.cannon_level > 1:
            upgrade_text = self.font_main.render(f"CANNON LV: {self.cannon_level}", True, self.COLOR_PROJECTILE)
            self.screen.blit(upgrade_text, (10, 60))

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            status = "SURVIVED" if self.base_health > 0 else "BASE DESTROYED"
            go_text = self.font_main.render(f"GAME OVER: {status}", True, self.COLOR_TEXT)
            go_rect = go_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(go_text, go_rect)

            score_text = self.font_main.render(f"FINAL SCORE: {int(self.score)}", True, self.COLOR_TEXT)
            score_rect = score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(score_text, score_rect)

    # --- Helper Methods ---

    def _draw_rotated_triangle(self, pos, angle, color):
        size = 8
        points = [
            (size, 0),
            (-size/2, -size/2),
            (-size/2, size/2)
        ]
        
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        
        rotated_points = [
            (p[0]*cos_a - p[1]*sin_a + pos[0], p[0]*sin_a + p[1]*cos_a + pos[1])
            for p in points
        ]
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, color)
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, color)

    @staticmethod
    def _lerp_angle(a, b, t):
        diff = b - a
        if diff > math.pi: diff -= 2 * math.pi
        if diff < -math.pi: diff += 2 * math.pi
        return a + diff * t
    
    @staticmethod
    def _lerp_color(c1, c2, t):
        t = max(0, min(1, t))
        return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Key mapping for manual play
    key_map = {
        pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4,
    }

    # Set up display for manual play
    pygame.display.init()
    display_surface = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Space Mining Defense")
    
    print("\n--- Manual Control ---")
    print("Arrows: Move cursor")
    print("Space: Perform action (Mine/Build/Upgrade)")
    print("Shift: Switch mode")
    print("Q: Quit")
    print("----------------------")

    clock = pygame.time.Clock()
    
    while not done:
        movement_action = 0
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                done = True

        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement_action = move_val
                break # Prioritize one movement key
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # For human play, we need to render the screen
        # The environment already renders to its internal surface, we just need to display it
        # Need to flip the observation array back to pygame's format
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_surface.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS) # Control the frame rate

    env.close()