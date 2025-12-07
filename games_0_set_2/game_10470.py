import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Classes for Game Entities ---

class Particle:
    def __init__(self, x, y, color, life, vx_range, vy_range, radius_range):
        self.x = x
        self.y = y
        self.vx = random.uniform(*vx_range)
        self.vy = random.uniform(*vy_range)
        self.life = life
        self.max_life = life
        self.color = color
        self.radius = random.uniform(*radius_range)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # Gravity effect
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            color = self.color + (alpha,)
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.radius), color)

class Projectile:
    def __init__(self, x, y, target_x, target_y, speed, damage, color):
        self.x, self.y = x, y
        angle = math.atan2(target_y - y, target_x - x)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.damage = damage
        self.color = color
        self.life = 100 # Lifespan in steps

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

    def draw(self, surface):
        pygame.draw.line(surface, self.color, (int(self.x - self.vx*0.5), int(self.y - self.vy*0.5)), (int(self.x), int(self.y)), 3)

class Nightmare:
    def __init__(self, x, y, health, speed, wave):
        self.x, self.y = x, y
        self.max_health = health
        self.health = health
        self.speed = speed
        self.radius = 15
        self.wave = wave

    def update(self):
        self.y += self.speed

    def draw(self, surface):
        # Body with glow
        GameEnv.draw_glow_circle(surface, (int(self.x), int(self.y)), self.radius, (200, 30, 80))
        # Health bar
        if self.health < self.max_health:
            bar_width = 30
            bar_height = 5
            health_pct = self.health / self.max_health
            pygame.draw.rect(surface, (100, 0, 0), (self.x - bar_width/2, self.y - self.radius - 10, bar_width, bar_height))
            pygame.draw.rect(surface, (0, 200, 0), (self.x - bar_width/2, self.y - self.radius - 10, bar_width * health_pct, bar_height))


class CelestialBeing:
    def __init__(self, grid_pos, being_type, unlock_wave):
        self.grid_pos = grid_pos
        self.type = being_type
        self.attack_cooldown = 0
        self.attack_rate = 60 if being_type == 1 else 90 # Slower for AoE
        self.attack_range = 150 if being_type == 1 else 100
        self.unlock_wave = unlock_wave
        self.pos = GameEnv.grid_to_pixel(grid_pos)

    def update(self, nightmares, projectiles):
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
            return

        # Find closest nightmare in range
        target = None
        min_dist = float('inf')
        for n in nightmares:
            dist = math.hypot(n.x - self.pos[0], n.y - self.pos[1])
            if dist < self.attack_range and dist < min_dist:
                min_dist = dist
                target = n
        
        if target:
            # sfx: celestial_attack
            self.attack_cooldown = self.attack_rate
            if self.type == 1: # Single target
                projectiles.append(Projectile(self.pos[0], self.pos[1], target.x, target.y, 5, 10, (255, 215, 0)))
            # Type 2 (AoE) could be implemented here

    def draw(self, surface):
        color = (255, 215, 0)
        pos = (int(self.pos[0]), int(self.pos[1]))
        GameEnv.draw_glow_circle(surface, pos, 12, color)
        pygame.gfxdraw.filled_trigon(surface, pos[0], pos[1]-12, pos[0]-10, pos[1]+8, pos[0]+10, pos[1]+8, color)


class Portal:
    def __init__(self, grid_pos):
        self.grid_pos = grid_pos
        self.energy = 0
        self.level = 1
        self.max_energy = 100 * self.level
        self.being = None
        self.pos = GameEnv.grid_to_pixel(grid_pos)

    def add_energy(self, amount):
        self.energy = min(self.max_energy, self.energy + amount)
        # sfx: charge_portal

    def upgrade(self):
        if self.energy >= self.max_energy:
            # sfx: upgrade_portal
            self.energy = 0
            self.level += 1
            self.max_energy = 100 * self.level
            return True
        return False

    def summon_being(self, unlocked_beings):
        if not self.being and self.energy >= self.max_energy and unlocked_beings:
            # sfx: summon_being
            self.energy = 0
            being_type = unlocked_beings[-1] # Summon the best available
            self.being = CelestialBeing(self.grid_pos, being_type['type'], being_type['wave'])
            return True
        return False
    
    def draw(self, surface):
        # Draw frame
        color = (100, 100, 255)
        radius = 18
        pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), radius, color)
        pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), radius-1, color)
        
        # Draw energy level
        if self.energy > 0:
            energy_pct = self.energy / self.max_energy
            glow_radius = int(radius * energy_pct)
            glow_color = (150, 150, 255)
            GameEnv.draw_glow_circle(surface, self.pos, glow_radius, glow_color, num_layers=5)
        
        if self.being:
            self.being.draw(surface)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend your core by matching cosmic energies to power up portals and summon celestial defenders against waves of encroaching nightmares."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the selector. Press space to match energies or summon defenders. Hold shift and press space to upgrade a portal."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 16, 10
    CELL_WIDTH = SCREEN_WIDTH // GRID_COLS
    CELL_HEIGHT = SCREEN_HEIGHT // GRID_ROWS
    MAX_STEPS = 2500
    INITIAL_HEALTH = 100
    
    # --- Colors ---
    COLOR_BG = (10, 5, 25)
    COLOR_GRID = (30, 20, 50)
    COLOR_TEXT = (220, 220, 255)
    COLOR_SELECTOR = (255, 255, 0, 150)
    ENERGY_COLORS = {1: (0, 150, 255), 2: (0, 255, 150)}

    # --- Celestial Beings Unlock Schedule ---
    CELESTIAL_UNLOCKS = [
        {'type': 1, 'wave': 1, 'name': 'Starlight Sentinel'},
        # {'type': 2, 'wave': 5, 'name': 'Nebula Warden'}, # Example for future expansion
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = 0
        self.wave_number = 0
        self.wave_cooldown = 0
        self.selector_pos = [0, 0]
        self.selected_energy_pos = None
        self.was_space_held = False
        self.energy_grid = np.zeros((self.GRID_COLS, self.GRID_ROWS), dtype=int)
        self.portals = []
        self.nightmares = []
        self.projectiles = []
        self.particles = []
        self.stars = []
        
        # self.reset() # reset is called by the environment wrapper
        # self.validate_implementation() # Not needed in production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.INITIAL_HEALTH
        self.wave_number = 0
        self.wave_cooldown = 120 # Cooldown before first wave
        
        self.selector_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_energy_pos = None
        self.was_space_held = False

        self.energy_grid = np.zeros((self.GRID_COLS, self.GRID_ROWS), dtype=int)
        self.portals = []
        self.nightmares = []
        self.projectiles = []
        self.particles = []

        self._populate_stars()
        self._spawn_initial_energies()

        # Place two starting portals
        self.portals.append(Portal([3, self.GRID_ROWS // 2]))
        self.portals.append(Portal([self.GRID_COLS - 4, self.GRID_ROWS // 2]))

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- 1. Handle Player Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.was_space_held

        self._handle_movement(movement)
        if space_pressed:
            reward += self._handle_action(shift_held)
        
        self.was_space_held = space_held

        # --- 2. Update Game Logic ---
        self._update_wave_spawning()
        reward += self._update_nightmares()
        self._update_celestials()
        self._update_projectiles()
        reward += self._check_collisions()
        self._update_particles()
        self._refill_energies()

        # --- 3. Calculate Final Reward & Termination ---
        # Small penalty for existing to encourage efficiency
        reward -= 0.01

        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS
        truncated = False
        if terminated and not self.game_over:
            reward -= 50 # Game over penalty
            self.game_over = True
            # sfx: game_over

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Update Sub-functions ---

    def _handle_movement(self, movement):
        if movement == 1: self.selector_pos[1] = max(0, self.selector_pos[1] - 1)
        elif movement == 2: self.selector_pos[1] = min(self.GRID_ROWS - 1, self.selector_pos[1] + 1)
        elif movement == 3: self.selector_pos[0] = max(0, self.selector_pos[0] - 1)
        elif movement == 4: self.selector_pos[0] = min(self.GRID_COLS - 1, self.selector_pos[0] + 1)

    def _handle_action(self, shift_held):
        sel_x, sel_y = self.selector_pos
        
        # Check for portal interaction first
        for portal in self.portals:
            if portal.grid_pos == [sel_x, sel_y]:
                if shift_held: # Upgrade
                    if portal.upgrade():
                        self._create_particles(portal.pos[0], portal.pos[1], (255, 255, 100), 20)
                        return 10 # Reward for upgrading
                else: # Summon
                    unlocked = [b for b in self.CELESTIAL_UNLOCKS if self.wave_number >= b['wave']]
                    if portal.summon_being(unlocked):
                        self._create_particles(portal.pos[0], portal.pos[1], (255, 215, 0), 30)
                        return 5 # Reward for summoning
                return 0

        # Check for energy interaction
        energy_type = self.energy_grid[sel_x, sel_y]
        if energy_type > 0:
            # sfx: select_energy
            if self.selected_energy_pos is None:
                self.selected_energy_pos = [sel_x, sel_y]
            else:
                # Check for match
                prev_x, prev_y = self.selected_energy_pos
                if self.energy_grid[prev_x, prev_y] == energy_type and self._are_adjacent(self.selected_energy_pos, [sel_x, sel_y]):
                    self.energy_grid[sel_x, sel_y] = 0
                    self.energy_grid[prev_x, prev_y] = 0
                    
                    match_pos = self.grid_to_pixel([(sel_x + prev_x)/2, (sel_y + prev_y)/2])
                    self._create_particles(match_pos[0], match_pos[1], self.ENERGY_COLORS[energy_type], 15)
                    self._distribute_energy(match_pos, 50) # Give 50 energy per match
                    self.selected_energy_pos = None
                    # sfx: energy_match
                    return 1 # Reward for matching
                else:
                    self.selected_energy_pos = [sel_x, sel_y] # Select new one
            return 0
        
        # If nothing else, deselect
        self.selected_energy_pos = None
        return 0

    def _update_wave_spawning(self):
        if not self.nightmares and self.wave_cooldown > 0:
            self.wave_cooldown -= 1
            if self.wave_cooldown == 0:
                self.wave_number += 1
                self._start_wave()
                if self.wave_number > 1:
                    return 50 # Wave survived reward
        return 0

    def _update_nightmares(self):
        damage_to_player = 0
        for n in self.nightmares[:]:
            n.update()
            if n.y > self.SCREEN_HEIGHT:
                self.player_health -= 10
                damage_to_player += 1
                self.nightmares.remove(n)
                # sfx: player_damage
        
        if not self.nightmares and self.wave_cooldown <= 0:
            self.wave_cooldown = 180 # 6 seconds at 30fps
        
        return -5 * damage_to_player # Penalty for taking damage

    def _update_celestials(self):
        for p in self.portals:
            if p.being:
                p.being.update(self.nightmares, self.projectiles)

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p.update()
            if not (0 < p.x < self.SCREEN_WIDTH and 0 < p.y < self.SCREEN_HEIGHT) or p.life <= 0:
                self.projectiles.remove(p)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def _check_collisions(self):
        reward = 0
        for p in self.projectiles[:]:
            for n in self.nightmares[:]:
                if math.hypot(p.x - n.x, p.y - n.y) < n.radius:
                    n.health -= p.damage
                    self._create_particles(n.x, n.y, (255,100,100), 5, vx_range=(-1,1), vy_range=(-1,1))
                    if p in self.projectiles: self.projectiles.remove(p)
                    
                    if n.health <= 0:
                        self.score += 10
                        reward += 10 # Reward for defeating nightmare
                        self._create_particles(n.x, n.y, (220,0,80), 30)
                        self.nightmares.remove(n)
                        # sfx: nightmare_death
                    break
        return reward

    # --- Spawning and Helper Logic ---

    def _populate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)),
                'radius': random.uniform(0.5, 1.5),
                'color': random.randint(50, 100)
            })

    def _spawn_initial_energies(self):
        for _ in range(20):
            self._spawn_single_energy()

    def _refill_energies(self):
        if self.steps % 15 == 0 and np.count_nonzero(self.energy_grid) < 30:
            self._spawn_single_energy()

    def _spawn_single_energy(self):
        empty_cells = np.argwhere(self.energy_grid == 0)
        if len(empty_cells) > 0:
            x, y = random.choice(empty_cells)
            self.energy_grid[x, y] = random.choice(list(self.ENERGY_COLORS.keys()))

    def _start_wave(self):
        num_nightmares = 3 + self.wave_number * 2
        health = 20 * (1.05 ** (self.wave_number - 1))
        speed = 0.5 * (1.05 ** (self.wave_number - 1))
        for _ in range(num_nightmares):
            x = random.randint(20, self.SCREEN_WIDTH - 20)
            y = random.randint(-80, -20)
            self.nightmares.append(Nightmare(x, y, health, speed, self.wave_number))

    def _distribute_energy(self, pos, amount):
        if not self.portals: return
        # Find nearest portal
        nearest_portal = min(self.portals, key=lambda p: math.hypot(p.pos[0] - pos[0], p.pos[1] - pos[1]))
        nearest_portal.add_energy(amount)

    @staticmethod
    def _are_adjacent(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    @classmethod
    def grid_to_pixel(cls, grid_pos):
        x = grid_pos[0] * cls.CELL_WIDTH + cls.CELL_WIDTH / 2
        y = grid_pos[1] * cls.CELL_HEIGHT + cls.CELL_HEIGHT / 2
        return x, y

    def _create_particles(self, x, y, color, count, vx_range=(-2,2), vy_range=(-3,0)):
        for _ in range(count):
            self.particles.append(Particle(x, y, color, random.randint(20, 40), vx_range, vy_range, (1, 4)))

    # --- Rendering ---

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_starfield()
        self._draw_grid()
        
        for p in self.particles: p.draw(self.screen)
        for p in self.portals: p.draw(self.screen)
        self._draw_energies()
        for n in self.nightmares: n.draw(self.screen)
        for p in self.projectiles: p.draw(self.screen)
        
        self._draw_selector()
        self._render_ui()

        if self.game_over:
            self._draw_game_over()

    def _draw_starfield(self):
        for star in self.stars:
            c = star['color']
            pygame.gfxdraw.filled_circle(self.screen, int(star['pos'][0]), int(star['pos'][1]), int(star['radius']), (c,c,c,150))

    def _draw_grid(self):
        for x in range(self.GRID_COLS + 1):
            px = x * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_ROWS + 1):
            py = y * self.CELL_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py))

    def _draw_energies(self):
        pulse = abs(math.sin(self.steps * 0.1)) * 3
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                energy_type = self.energy_grid[x, y]
                if energy_type > 0:
                    px, py = self.grid_to_pixel([x, y])
                    color = self.ENERGY_COLORS[energy_type]
                    radius = self.CELL_HEIGHT / 3 + pulse
                    self.draw_glow_circle(self.screen, (px, py), radius, color)

    def _draw_selector(self):
        px, py = self.grid_to_pixel(self.selector_pos)
        rect = pygame.Rect(px - self.CELL_WIDTH/2, py - self.CELL_HEIGHT/2, self.CELL_WIDTH, self.CELL_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, rect, 2, border_radius=4)
        
        if self.selected_energy_pos:
            px, py = self.grid_to_pixel(self.selected_energy_pos)
            rect = pygame.Rect(px - self.CELL_WIDTH/2, py - self.CELL_HEIGHT/2, self.CELL_WIDTH, self.CELL_HEIGHT)
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 3, border_radius=4)

    def _render_ui(self):
        # Wave
        wave_text = self.font_large.render(f"Wave: {self.wave_number}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        # Health
        health_bar_width = 200
        health_pct = max(0, self.player_health / self.INITIAL_HEALTH)
        pygame.draw.rect(self.screen, (100,0,0), (self.SCREEN_WIDTH/2 - health_bar_width/2, 15, health_bar_width, 20))
        pygame.draw.rect(self.screen, (0,200,100), (self.SCREEN_WIDTH/2 - health_bar_width/2, 15, health_bar_width * health_pct, 20))
        health_text = self.font_small.render("CORE INTEGRITY", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.SCREEN_WIDTH/2 - health_text.get_width()/2, 16))

        # Selected Object Info
        info_text = ""
        sel_x, sel_y = self.selector_pos
        for portal in self.portals:
            if portal.grid_pos == [sel_x, sel_y]:
                info_text = f"Portal Lvl {portal.level} | Energy: {int(portal.energy)}/{portal.max_energy}"
                if portal.being:
                    info_text += f" | Sentinel Active"
        
        if info_text:
            text_surf = self.font_small.render(info_text, True, self.COLOR_TEXT)
            self.screen.blit(text_surf, (self.SCREEN_WIDTH/2 - text_surf.get_width()/2, self.SCREEN_HEIGHT - 30))

    def _draw_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        game_over_text = self.font_large.render("GAME OVER", True, (255, 50, 50))
        self.screen.blit(game_over_text, (self.SCREEN_WIDTH/2 - game_over_text.get_width()/2, self.SCREEN_HEIGHT/2 - 30))
        
        final_score_text = self.font_small.render(f"Final Score: {self.score} | Waves Survived: {self.wave_number -1}", True, self.COLOR_TEXT)
        self.screen.blit(final_score_text, (self.SCREEN_WIDTH/2 - final_score_text.get_width()/2, self.SCREEN_HEIGHT/2 + 20))

    @staticmethod
    def draw_glow_circle(surface, pos, radius, color, num_layers=10):
        if radius <= 0: return
        for i in range(num_layers, 0, -1):
            alpha = int(150 * (1 - (i / num_layers)))
            glow_color = color + (alpha,)
            pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius * (i / num_layers) * 1.5), glow_color)
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius), color)
        pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]), int(radius), color)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number, "health": self.player_health}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play Code ---
    # This block requires a graphical display. If you are running in a headless environment,
    # this will not work. The environment itself is headless-compatible.
    try:
        env = GameEnv()
        obs, info = env.reset()
        done = False
        
        pygame.display.init()
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Astral Portal Defense")
        clock = pygame.time.Clock()
        
        movement = 0
        space_held = 0
        shift_held = 0

        print("\n--- Controls ---")
        print("Arrow Keys: Move Selector")
        print("Space: Select/Match Energy | Summon Being")
        print("Shift + Space: Upgrade Portal")
        print("Q: Quit")
        
        running = True
        while running:
            # --- Action Mapping for Human Play ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False

            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Step Environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if reward != 0 and reward != -0.01:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Wave: {info['wave']}")

            # --- Render ---
            # The observation is already a rendered frame
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if done:
                print("Game Over!")
                pygame.time.wait(3000)
                obs, info = env.reset()

            clock.tick(30) # Run at 30 FPS

        env.close()
    except pygame.error as e:
        print(f"Pygame error: {e}")
        print("Could not initialize display. This is expected in a headless environment.")
        print("The GameEnv class is still valid and can be used by an agent.")
        # Example of headless usage
        print("\nRunning a short headless test...")
        env = GameEnv()
        obs, info = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print("Episode finished.")
                obs, info = env.reset()
        env.close()
        print("Headless test complete.")