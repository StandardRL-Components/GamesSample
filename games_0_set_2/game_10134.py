import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:58:49.969382
# Source Brief: brief_00134.md
# Brief Index: 134
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Classes for Game Entities ---

class Particle:
    """A simple class for a cosmetic particle."""
    def __init__(self, x, y, color, life, size, angle, speed):
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.max_life = life
        self.size = size
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            color = self.color + (alpha,)
            rect = pygame.Rect(int(self.x - self.size / 2), int(self.y - self.size / 2), int(self.size), int(self.size))
            shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
            surface.blit(shape_surf, rect)

class Nanite:
    """Represents an enemy nanite."""
    def __init__(self, x, y, speed, health, radius, damage):
        self.x = x
        self.y = y
        self.speed = speed
        self.max_health = health
        self.health = health
        self.radius = radius
        self.damage = damage
        self.is_slowed = False

    def update(self, target_pos):
        current_speed = self.speed * 0.5 if self.is_slowed else self.speed
        dx = target_pos[0] - self.x
        dy = target_pos[1] - self.y
        dist = math.hypot(dx, dy)
        if dist > 1:
            self.x += (dx / dist) * current_speed
            self.y += (dy / dist) * current_speed
        self.is_slowed = False # Reset slow status each frame

    def draw(self, surface):
        # Body with glow
        glow_color = (255, 50, 50, 60)
        body_color = (255, 0, 0)
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), self.radius + 3, glow_color)
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), self.radius, body_color)
        pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), self.radius, body_color)
        
        # Health bar
        if self.health < self.max_health:
            bar_w = self.radius * 2
            bar_h = 4
            bar_x = self.x - self.radius
            bar_y = self.y - self.radius - 8
            health_pct = self.health / self.max_health
            pygame.draw.rect(surface, (50, 0, 0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(surface, (255, 0, 0), (bar_x, bar_y, bar_w * health_pct, bar_h))

class Nanobot:
    """Base class for player-controlled nanobots."""
    def __init__(self, grid_x, grid_y, grid_size):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.x = (grid_x + 0.5) * grid_size
        self.y = (grid_y + 0.5) * grid_size
        self.target = None

    def find_target(self, nanites):
        closest_nanite = None
        min_dist_sq = self.range * self.range
        for nanite in nanites:
            dist_sq = (self.x - nanite.x)**2 + (self.y - nanite.y)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_nanite = nanite
        self.target = closest_nanite

class BlasterBot(Nanobot):
    cost = 100
    range = 150
    fire_rate = 30 # frames per shot
    damage = 25
    
    def __init__(self, grid_x, grid_y, grid_size):
        super().__init__(grid_x, grid_y, grid_size)
        self.cooldown = 0

    def update(self, nanites, projectiles):
        self.find_target(nanites)
        if self.cooldown > 0:
            self.cooldown -= 1
        if self.target and self.cooldown == 0:
            # Sfx: blaster_fire.wav
            projectiles.append(Projectile(self.x, self.y, self.target, self.damage))
            self.cooldown = self.fire_rate

    def draw(self, surface):
        rect = pygame.Rect(0, 0, 20, 20)
        rect.center = (int(self.x), int(self.y))
        pygame.draw.rect(surface, (0, 150, 255), rect, 0, 3)
        pygame.draw.rect(surface, (200, 220, 255), rect, 2, 3)

class LaserBot(Nanobot):
    cost = 175
    range = 120
    dps = 20 # damage per second
    
    def update(self, nanites, particles, fps):
        self.find_target(nanites)
        if self.target:
            damage_per_frame = self.dps / fps
            self.target.health -= damage_per_frame
            # Sfx: laser_hum.wav (continuous)
            if random.random() < 0.5:
                 particles.append(Particle(self.target.x, self.target.y, (255, 255, 0), 5, 2, random.uniform(0, 2*math.pi), 1))

    def draw(self, surface):
        points = [(self.x, self.y - 10), (self.x - 10, self.y + 10), (self.x + 10, self.y + 10)]
        pygame.gfxdraw.filled_trigon(surface, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), (0, 150, 255))
        pygame.gfxdraw.aatrigon(surface, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), (200, 220, 255))
        if self.target:
            pygame.draw.aaline(surface, (255, 255, 0, 200), (self.x, self.y), (self.target.x, self.target.y), 2)
            pygame.draw.aaline(surface, (255, 255, 255, 255), (self.x, self.y), (self.target.x, self.target.y), 1)

class SlowerBot(Nanobot):
    cost = 75
    range = 100
    slow_factor = 0.5
    
    def update(self, nanites):
        for nanite in nanites:
            dist_sq = (self.x - nanite.x)**2 + (self.y - nanite.y)**2
            if dist_sq < self.range * self.range:
                nanite.is_slowed = True

    def draw(self, surface):
        # Draw aura
        aura_color = (0, 100, 255, 30)
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), self.range, aura_color)
        pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), self.range, (0, 100, 255, 60))
        # Draw body
        points = []
        for i in range(6):
            angle = math.pi / 3 * i
            points.append((self.x + 10 * math.cos(angle), self.y + 10 * math.sin(angle)))
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.filled_polygon(surface, int_points, (0, 150, 255))
        pygame.gfxdraw.aapolygon(surface, int_points, (200, 220, 255))

class Projectile:
    def __init__(self, x, y, target, damage):
        self.x = x
        self.y = y
        self.target = target
        self.damage = damage
        self.speed = 10
        dx = target.x - x
        dy = target.y - y
        dist = math.hypot(dx, dy)
        self.vx = (dx / dist) * self.speed
        self.vy = (dy / dist) * self.speed

    def update(self):
        self.x += self.vx
        self.y += self.vy

    def draw(self, surface):
        pygame.draw.circle(surface, (150, 200, 255), (int(self.x), int(self.y)), 4)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend your central factory from waves of attacking nanites by strategically "
        "placing various types of defensive nanobots."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place the "
        "selected nanobot. Press shift to cycle between different nanobot types."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Game constants
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_GRID = (20, 40, 60)
        self.COLOR_UI_TEXT = (200, 220, 255)
        self.COLOR_RESOURCE = (255, 220, 0)
        self.COLOR_FACTORY_HEALTH = (0, 255, 100)
        
        self.MAX_STEPS = 3000
        self.MAX_WAVES = 20
        self.GRID_SIZE = 32
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE
        
        self.bot_types = [BlasterBot, LaserBot, SlowerBot]
        
        # State variables are initialized in reset()
        self.reset()
        
        # self.validate_implementation() # Commented out for submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.reward_sum = 0
        self.game_over = False
        
        self.factory_pos = (self.WIDTH // 2, self.HEIGHT // 2)
        self.factory_radius = 40
        self.factory_max_health = 1000
        self.factory_health = self.factory_max_health
        
        self.resources = 250
        self.nanobots = []
        self.nanites = []
        self.projectiles = []
        self.particles = []
        
        self.current_wave = 0
        self.wave_cooldown = self.FPS * 5
        self.wave_cooldown_timer = self.wave_cooldown
        
        self.cursor_grid_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_bot_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.occupied_cells = set()
        factory_grid_x = self.factory_pos[0] // self.GRID_SIZE
        factory_grid_y = self.factory_pos[1] // self.GRID_SIZE
        for i in range(-1, 2):
            for j in range(-1, 2):
                self.occupied_cells.add((factory_grid_x + i, factory_grid_y + j))

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward_buffer = 0.0
        
        self._handle_input(movement, space_held, shift_held)
        
        # --- Game Logic Updates ---
        self._update_wave_manager()
        self._update_nanobots()
        factory_damage, destruction_rewards = self._update_nanites_and_projectiles()
        self.factory_health -= factory_damage
        reward_buffer += destruction_rewards
        self._update_particles()
        
        # --- State and Reward Updates ---
        self.steps += 1
        self.reward_sum += reward_buffer
        
        terminated = False
        truncated = False
        if self.factory_health <= 0:
            terminated = True
            reward_buffer -= 100.0
            self.game_over = True
        elif self.current_wave > self.MAX_WAVES:
            terminated = True
            reward_buffer += 100.0
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
        
        if self.wave_completed_this_frame:
            reward_buffer += 1.0

        return (
            self._get_observation(),
            reward_buffer,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1: self.cursor_grid_pos[1] -= 1  # Up
        if movement == 2: self.cursor_grid_pos[1] += 1  # Down
        if movement == 3: self.cursor_grid_pos[0] -= 1  # Left
        if movement == 4: self.cursor_grid_pos[0] += 1  # Right
        self.cursor_grid_pos[0] = np.clip(self.cursor_grid_pos[0], 0, self.GRID_W - 1)
        self.cursor_grid_pos[1] = np.clip(self.cursor_grid_pos[1], 0, self.GRID_H - 1)
        
        # Cycle bot type (on press)
        if shift_held and not self.prev_shift_held:
            self.selected_bot_idx = (self.selected_bot_idx + 1) % len(self.bot_types)
            # Sfx: ui_cycle.wav
        
        # Place bot (on press)
        if space_held and not self.prev_space_held:
            bot_class = self.bot_types[self.selected_bot_idx]
            pos_tuple = tuple(self.cursor_grid_pos)
            if self.resources >= bot_class.cost and pos_tuple not in self.occupied_cells:
                self.resources -= bot_class.cost
                new_bot = bot_class(self.cursor_grid_pos[0], self.cursor_grid_pos[1], self.GRID_SIZE)
                self.nanobots.append(new_bot)
                self.occupied_cells.add(pos_tuple)
                # Sfx: place_bot.wav
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_wave_manager(self):
        self.wave_completed_this_frame = False
        if not self.nanites and self.current_wave > 0 and self.current_wave <= self.MAX_WAVES:
            self.wave_cooldown_timer -= 1
            if self.wave_cooldown_timer <= 0:
                self._spawn_next_wave()
                self.wave_cooldown_timer = self.wave_cooldown
        elif self.current_wave == 0: # Start of game
             self.wave_cooldown_timer -= 1
             if self.wave_cooldown_timer <= 0:
                self._spawn_next_wave()
                self.wave_cooldown_timer = self.wave_cooldown

    def _spawn_next_wave(self):
        self.current_wave += 1
        if self.current_wave > 1:
            self.wave_completed_this_frame = True
            self.resources += 100 # Wave completion resource bonus
        
        num_nanites = 5 + (self.current_wave - 1) * 2
        speed = 0.5 + (self.current_wave - 1) * 0.05
        health = 50 + (self.current_wave - 1) * 10
        damage = 10 + self.current_wave
        
        for _ in range(num_nanites):
            edge = random.randint(0, 3)
            if edge == 0: x, y = random.randint(0, self.WIDTH), -20
            elif edge == 1: x, y = self.WIDTH + 20, random.randint(0, self.HEIGHT)
            elif edge == 2: x, y = random.randint(0, self.WIDTH), self.HEIGHT + 20
            else: x, y = -20, random.randint(0, self.HEIGHT)
            self.nanites.append(Nanite(x, y, speed, health, 6, damage))

    def _update_nanobots(self):
        for bot in self.nanobots:
            if isinstance(bot, BlasterBot):
                bot.update(self.nanites, self.projectiles)
            elif isinstance(bot, LaserBot):
                bot.update(self.nanites, self.particles, self.FPS)
            elif isinstance(bot, SlowerBot):
                bot.update(self.nanites)

    def _update_nanites_and_projectiles(self):
        factory_damage_taken = 0
        destruction_rewards = 0

        # Update and check projectiles
        for p in self.projectiles[:]:
            p.update()
            hit = False
            if p.x < 0 or p.x > self.WIDTH or p.y < 0 or p.y > self.HEIGHT:
                self.projectiles.remove(p)
                continue
            
            for nanite in self.nanites:
                if math.hypot(p.x - nanite.x, p.y - nanite.y) < nanite.radius:
                    nanite.health -= p.damage
                    hit = True
                    break
            if hit:
                self.projectiles.remove(p)

        # Update nanites
        for nanite in self.nanites[:]:
            nanite.update(self.factory_pos)
            
            if nanite.health <= 0:
                # Sfx: nanite_destroy.wav
                self._create_explosion(nanite.x, nanite.y, 20, (255, 100, 100))
                self.nanites.remove(nanite)
                self.resources += 10
                destruction_rewards += 0.1
                continue

            if math.hypot(nanite.x - self.factory_pos[0], nanite.y - self.factory_pos[1]) < self.factory_radius:
                # Sfx: factory_damage.wav
                factory_damage_taken += nanite.damage
                self._create_explosion(nanite.x, nanite.y, 10, (100, 200, 255))
                self.nanites.remove(nanite)
        
        return factory_damage_taken, destruction_rewards

    def _create_explosion(self, x, y, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            life = random.randint(10, 20)
            size = random.uniform(1, 4)
            self.particles.append(Particle(x, y, color, life, size, angle, speed))

    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        for bot in self.nanobots:
            if isinstance(bot, SlowerBot): # Draw auras first
                bot.draw(self.screen)
        self._render_factory()
        for nanite in self.nanites:
            nanite.draw(self.screen)
        for bot in self.nanobots:
            if not isinstance(bot, SlowerBot):
                bot.draw(self.screen)
        for proj in self.projectiles:
            proj.draw(self.screen)
        for particle in self.particles:
            particle.draw(self.screen)
        self._render_cursor()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_factory(self):
        x, y = self.factory_pos
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.factory_radius + 5, (0, 150, 50, 50))
        # Body
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.factory_radius, (50, 80, 120))
        pygame.gfxdraw.aacircle(self.screen, x, y, self.factory_radius, (100, 150, 200))
        # Core
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.factory_radius // 2, (100, 180, 255))
        pygame.gfxdraw.aacircle(self.screen, x, y, self.factory_radius // 2, (200, 220, 255))
        
        # Health bar
        health_pct = max(0, self.factory_health / self.factory_max_health)
        bar_width = 200
        bar_height = 15
        bar_x = self.WIDTH / 2 - bar_width / 2
        bar_y = self.HEIGHT - 30
        pygame.draw.rect(self.screen, (50, 0, 0), (bar_x, bar_y, bar_width, bar_height), 0, 4)
        pygame.draw.rect(self.screen, self.COLOR_FACTORY_HEALTH, (bar_x, bar_y, bar_width * health_pct, bar_height), 0, 4)
        pygame.draw.rect(self.screen, (255, 255, 255), (bar_x, bar_y, bar_width, bar_height), 1, 4)
        health_text = self.font_small.render(f"Factory Core", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (bar_x + bar_width / 2 - health_text.get_width() / 2, bar_y - 20))

    def _render_cursor(self):
        bot_class = self.bot_types[self.selected_bot_idx]
        can_afford = self.resources >= bot_class.cost
        is_occupied = tuple(self.cursor_grid_pos) in self.occupied_cells
        can_place = can_afford and not is_occupied
        
        color = (0, 255, 0, 100) if can_place else (255, 0, 0, 100)
        x = self.cursor_grid_pos[0] * self.GRID_SIZE
        y = self.cursor_grid_pos[1] * self.GRID_SIZE
        rect = pygame.Rect(x, y, self.GRID_SIZE, self.GRID_SIZE)
        
        shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
        self.screen.blit(shape_surf, rect.topleft)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)

    def _render_ui(self):
        # Resources
        res_text = self.font_small.render(f"Resources: {self.resources}", True, self.COLOR_RESOURCE)
        self.screen.blit(res_text, (10, 10))
        
        # Wave info
        if self.current_wave > 0 and self.current_wave <= self.MAX_WAVES:
            if self.nanites:
                wave_text = self.font_small.render(f"Wave: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
            else:
                next_wave_in = math.ceil(self.wave_cooldown_timer / self.FPS)
                wave_text = self.font_small.render(f"Next wave in: {next_wave_in}", True, self.COLOR_UI_TEXT)
        else:
            wave_text = self.font_small.render(f"Wave: -/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Bot selection UI
        panel_w, panel_h = 240, 60
        panel_x, panel_y = (self.WIDTH - panel_w) // 2, 10
        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        
        s = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        s.fill((10, 20, 40, 180))
        self.screen.blit(s, (panel_x, panel_y))
        pygame.draw.rect(self.screen, (100, 150, 200, 200), panel_rect, 1, 4)

        for i, bot_class in enumerate(self.bot_types):
            box_x = panel_x + 20 + i * 70
            box_y = panel_y + 10
            
            # Draw bot preview
            dummy_bot = bot_class(0,0,0)
            dummy_bot.x, dummy_bot.y = box_x + 12, box_y + 12
            dummy_bot.draw(self.screen)

            # Cost text
            cost_text = self.font_small.render(f"${bot_class.cost}", True, self.COLOR_RESOURCE)
            self.screen.blit(cost_text, (box_x, box_y + 25))

            if i == self.selected_bot_idx:
                pygame.draw.rect(self.screen, (255, 255, 255), (box_x - 5, box_y - 5, 35, 45), 2, 3)

    def _render_game_over(self):
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        s.fill((0,0,0,180))
        self.screen.blit(s, (0,0))
        
        if self.current_wave > self.MAX_WAVES:
            text = "VICTORY"
            color = (0, 255, 0)
        else:
            text = "FACTORY DESTROYED"
            color = (255, 0, 0)
            
        text_surf = self.font_large.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.reward_sum,
            "steps": self.steps,
            "wave": self.current_wave,
            "resources": self.resources,
            "factory_health": self.factory_health,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a display window
    pygame.display.set_caption("Nanite Defense")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    action = [0, 0, 0] # No-op, not holding space, not holding shift
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # Manual keyboard controls
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
        
        # Draw the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
    env.close()