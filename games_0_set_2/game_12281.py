import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:33:27.317279
# Source Brief: brief_02281.md
# Brief Index: 2281
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your central colony from waves of pathogens by controlling gravity to maneuver your defenses and deploying repair units."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to change the direction of gravity. "
        "Press space and shift to deploy repair units from the portals."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2500
    TOTAL_WAVES = 20

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 40, 60)
    COLOR_COLONY = (255, 255, 200)
    COLOR_COLONY_GLOW = (255, 255, 150)
    COLOR_DEFENSE = (0, 255, 150)
    COLOR_DEFENSE_GLOW = (0, 200, 120)
    COLOR_PATHOGEN = (255, 50, 50)
    COLOR_PATHOGEN_GLOW = (200, 40, 40)
    COLOR_REPAIR = (50, 150, 255)
    COLOR_REPAIR_GLOW = (40, 120, 200)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR = (0, 200, 100)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)
    
    # Physics
    GRAVITY_STRENGTH = 0.08
    DRAG = 0.99
    BOUNCE_DAMPING = 0.75

    # --- Entity Base Class ---
    class Entity:
        def __init__(self, pos, radius):
            self.pos = pygame.math.Vector2(pos)
            self.vel = pygame.math.Vector2(0, 0)
            self.radius = radius
            self.to_remove = False

        def update(self, gravity, bounds):
            self.vel += gravity
            self.vel *= GameEnv.DRAG
            self.pos += self.vel

            # Bounce off walls
            if self.pos.x - self.radius < 0:
                self.pos.x = self.radius
                self.vel.x *= -GameEnv.BOUNCE_DAMPING
            elif self.pos.x + self.radius > bounds[0]:
                self.pos.x = bounds[0] - self.radius
                self.vel.x *= -GameEnv.BOUNCE_DAMPING
            if self.pos.y - self.radius < 0:
                self.pos.y = self.radius
                self.vel.y *= -GameEnv.BOUNCE_DAMPING
            elif self.pos.y + self.radius > bounds[1]:
                self.pos.y = bounds[1] - self.radius
                self.vel.y *= -GameEnv.BOUNCE_DAMPING
    
    # --- Game Entity Classes ---
    class Defense(Entity):
        def __init__(self, pos, radius=10):
            super().__init__(pos, radius)

    class Pathogen(Entity):
        def __init__(self, pos, radius=8, speed_multiplier=1.0):
            super().__init__(pos, radius)
            self.speed_multiplier = speed_multiplier
            self.damage = 5

        def update(self, gravity, bounds, colony_pos):
            # Pathogens are drawn towards the colony
            direction_to_colony = (colony_pos - self.pos).normalize()
            self.vel += direction_to_colony * 0.05 * self.speed_multiplier
            super().update(gravity, bounds)

    class RepairUnit(Entity):
        def __init__(self, pos, radius=6, speed_multiplier=1.0):
            super().__init__(pos, radius)
            self.speed_multiplier = speed_multiplier
            self.repair_amount = 2.5

        def update(self, gravity, bounds, colony_pos):
            # Repair units are drawn towards the colony
            direction_to_colony = (colony_pos - self.pos).normalize()
            self.vel += direction_to_colony * 0.1 * self.speed_multiplier
            super().update(gravity, bounds)

    class Particle:
        def __init__(self, pos, color, duration, start_radius, end_radius=0):
            self.pos = pygame.math.Vector2(pos)
            self.color = color
            self.duration = duration
            self.life = duration
            self.start_radius = start_radius
            self.end_radius = end_radius

        def update(self):
            self.life -= 1
            return self.life <= 0

        def draw(self, surface):
            if self.life > 0:
                progress = self.life / self.duration
                radius = self.start_radius * progress + self.end_radius * (1 - progress)
                alpha = int(255 * progress)
                color = self.color + (alpha,)
                pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(radius), color)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18, bold=True)
        
        self.render_mode = render_mode
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        
        # Game state
        self.colony_pos = pygame.math.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.colony_radius = 35
        self.colony_max_health = 200
        self.colony_health = self.colony_max_health
        self.gravity = pygame.math.Vector2(0, 0)

        # Entity lists
        self.defenses = []
        self.pathogens = []
        self.repair_units = []
        self.particles = deque(maxlen=200)

        # Wave system
        self.wave_number = 0
        self.wave_timer = 120 # Time until first wave
        self.pathogens_to_spawn_in_wave = 0
        self.pathogen_spawn_timer = 0
        
        # Player action cooldowns
        self.repair_cooldowns = [0, 0]
        self.repair_cooldown_time = 45 # steps

        # Upgrades
        self.repair_speed_multiplier = 1.0

        # Portals
        self.portals = [
            pygame.math.Vector2(50, 50),
            pygame.math.Vector2(self.SCREEN_WIDTH - 50, 50)
        ]
        
        # Initial setup
        self._spawn_initial_defenses(5)
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = 0
        self.steps += 1

        self._handle_input(action)
        self._update_game_state()
        self._handle_collisions()
        self._cleanup_entities()
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        final_reward = self.reward_this_step
        if terminated and not truncated: # Game ended due to win/loss condition
            if self.colony_health <= 0:
                final_reward -= 100
                self.score -= 100
            elif self.wave_number > self.TOTAL_WAVES:
                final_reward += 100
                self.score += 100
        
        self.score += self.reward_this_step

        return (
            self._get_observation(),
            final_reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Gravity
        if movement == 1: self.gravity = pygame.math.Vector2(0, -self.GRAVITY_STRENGTH)
        elif movement == 2: self.gravity = pygame.math.Vector2(0, self.GRAVITY_STRENGTH)
        elif movement == 3: self.gravity = pygame.math.Vector2(-self.GRAVITY_STRENGTH, 0)
        elif movement == 4: self.gravity = pygame.math.Vector2(self.GRAVITY_STRENGTH, 0)
        else: self.gravity = pygame.math.Vector2(0, 0) # movement == 0 is no-op

        # Repair units
        if space_held and self.repair_cooldowns[0] <= 0:
            unit = self.RepairUnit(self.portals[0].copy(), speed_multiplier=self.repair_speed_multiplier)
            unit.vel = (self.colony_pos - unit.pos).normalize() * 0.5
            self.repair_units.append(unit)
            self.repair_cooldowns[0] = self.repair_cooldown_time

        if shift_held and self.repair_cooldowns[1] <= 0:
            unit = self.RepairUnit(self.portals[1].copy(), speed_multiplier=self.repair_speed_multiplier)
            unit.vel = (self.colony_pos - unit.pos).normalize() * 0.5
            self.repair_units.append(unit)
            self.repair_cooldowns[1] = self.repair_cooldown_time

    def _update_game_state(self):
        # Update cooldowns
        for i in range(len(self.repair_cooldowns)):
            if self.repair_cooldowns[i] > 0:
                self.repair_cooldowns[i] -= 1

        # Update waves
        if not self.pathogens and self.pathogens_to_spawn_in_wave == 0:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                if self.wave_number > 0: # Don't reward for wave 0
                    self.reward_this_step += 1.0
                self._start_next_wave()
        
        # Spawn pathogens for current wave
        if self.pathogens_to_spawn_in_wave > 0:
            self.pathogen_spawn_timer -= 1
            if self.pathogen_spawn_timer <= 0:
                self._spawn_pathogen()
                self.pathogens_to_spawn_in_wave -= 1
                self.pathogen_spawn_timer = max(10, 40 - self.wave_number)

        # Update entities
        bounds = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        for entity_list in [self.defenses, self.pathogens, self.repair_units]:
            for entity in entity_list:
                if isinstance(entity, (self.Pathogen, self.RepairUnit)):
                    entity.update(self.gravity, bounds, self.colony_pos)
                else:
                    entity.update(self.gravity, bounds)
        
        # Update particles
        for p in list(self.particles):
            if p.update():
                self.particles.remove(p)

    def _handle_collisions(self):
        # Pathogen vs Defense
        for pathogen in self.pathogens:
            for defense in self.defenses:
                if pathogen.pos.distance_to(defense.pos) < pathogen.radius + defense.radius:
                    pathogen.to_remove = True
                    defense.to_remove = True
                    self.reward_this_step += 0.1
                    self._create_explosion(pathogen.pos, self.COLOR_PATHOGEN, 15)
                    break 

        # Pathogen vs Colony
        for pathogen in self.pathogens:
            if not pathogen.to_remove and pathogen.pos.distance_to(self.colony_pos) < pathogen.radius + self.colony_radius:
                pathogen.to_remove = True
                damage = pathogen.damage
                self.colony_health -= damage
                self.reward_this_step -= 0.1 * damage
                self._create_explosion(pathogen.pos, self.COLOR_COLONY, 10, 5)

        # Repair Unit vs Colony
        for unit in self.repair_units:
            if unit.pos.distance_to(self.colony_pos) < unit.radius + self.colony_radius:
                unit.to_remove = True
                self.colony_health = min(self.colony_max_health, self.colony_health + unit.repair_amount)
                for _ in range(5):
                    offset = pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.colony_radius
                    self.particles.append(self.Particle(self.colony_pos + offset, self.COLOR_REPAIR, 20, 5))

    def _cleanup_entities(self):
        self.defenses = [d for d in self.defenses if not d.to_remove]
        self.pathogens = [p for p in self.pathogens if not p.to_remove]
        self.repair_units = [r for r in self.repair_units if not r.to_remove]
        
        # Respawn defenses if they fall off screen
        if len(self.defenses) < 5:
            self._spawn_initial_defenses(5 - len(self.defenses))

    def _check_termination(self):
        if self.colony_health <= 0:
            self.game_over = True
            return True
        if self.wave_number > self.TOTAL_WAVES and not self.pathogens:
            self.game_over = True
            return True
        # Truncation is handled in step()
        return False

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
            "colony_health": self.colony_health,
            "wave": self.wave_number,
        }

    def _spawn_initial_defenses(self, count):
        for _ in range(count):
            pos = self.colony_pos + pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * random.uniform(80, 150)
            self.defenses.append(self.Defense(pos))

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.TOTAL_WAVES:
            return
            
        self.pathogens_to_spawn_in_wave = 5 + self.wave_number * 2
        self.wave_timer = 180 # Time between waves
        
        # Upgrades
        if self.wave_number == 5: self.repair_speed_multiplier = 1.25
        if self.wave_number == 10: self.repair_speed_multiplier = 1.5
        if self.wave_number == 15: self.repair_speed_multiplier = 1.75

    def _spawn_pathogen(self):
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top': pos = (random.uniform(0, self.SCREEN_WIDTH), -20)
        elif edge == 'bottom': pos = (random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20)
        elif edge == 'left': pos = (-20, random.uniform(0, self.SCREEN_HEIGHT))
        elif edge == 'right': pos = (self.SCREEN_WIDTH + 20, random.uniform(0, self.SCREEN_HEIGHT))
        
        speed_mult = 1.0 + (self.wave_number * 0.05)
        self.pathogens.append(self.Pathogen(pos, speed_multiplier=speed_mult))

    def _create_explosion(self, pos, color, count, radius=10):
        for _ in range(count):
            p_pos = pos + pygame.math.Vector2(random.uniform(-5, 5), random.uniform(-5, 5))
            self.particles.append(self.Particle(p_pos, color, random.randint(15, 30), random.uniform(2, radius)))

    def _render_game(self):
        # Draw grid
        for i in range(0, self.SCREEN_WIDTH, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

        # Draw portals
        for p in self.portals:
            self._draw_glowing_circle(self.screen, self.COLOR_REPAIR, (int(p.x), int(p.y)), 12, glow_factor=2.0)

        # Draw colony
        glow_radius = self.colony_radius + 10 * (0.5 + 0.5 * math.sin(self.steps * 0.05))
        self._draw_glowing_circle(self.screen, self.COLOR_COLONY_GLOW, (int(self.colony_pos.x), int(self.colony_pos.y)), int(glow_radius), glow_factor=2.5, glow_alpha=50)
        self._draw_glowing_circle(self.screen, self.COLOR_COLONY, (int(self.colony_pos.x), int(self.colony_pos.y)), self.colony_radius)
        
        # Draw entities
        for defense in self.defenses:
            self._draw_glowing_circle(self.screen, self.COLOR_DEFENSE, (int(defense.pos.x), int(defense.pos.y)), defense.radius)
        for repair in self.repair_units:
            self._draw_glowing_square(self.screen, self.COLOR_REPAIR, repair.pos, repair.radius)
        for pathogen in self.pathogens:
            self._draw_glowing_triangle(self.screen, self.COLOR_PATHOGEN, pathogen.pos, pathogen.radius, pathogen.vel)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Health bar
        health_ratio = max(0, self.colony_health / self.colony_max_health)
        bar_width = self.SCREEN_WIDTH - 40
        bar_height = 15
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (20, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (20, 10, int(bar_width * health_ratio), bar_height))
        
        # Text
        wave_text = f"Wave: {self.wave_number}/{self.TOTAL_WAVES}"
        score_text = f"Score: {int(self.score)}"
        
        wave_surf = self.font.render(wave_text, True, self.COLOR_UI_TEXT)
        score_surf = self.font.render(score_text, True, self.COLOR_UI_TEXT)
        
        self.screen.blit(wave_surf, (25, 30))
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 25, 30))
        
        # Wave timer
        if not self.pathogens and self.pathogens_to_spawn_in_wave == 0 and self.wave_number <= self.TOTAL_WAVES:
            timer_text = f"Next wave in {self.wave_timer // 30 + 1}"
            timer_surf = self.font.render(timer_text, True, self.COLOR_UI_TEXT)
            self.screen.blit(timer_surf, (self.SCREEN_WIDTH // 2 - timer_surf.get_width() // 2, 15))

    def _draw_glowing_circle(self, surface, color, center, radius, glow_factor=1.5, glow_alpha=30):
        glow_radius = int(radius * glow_factor)
        if glow_radius <= radius: return
        
        for i in range(radius, glow_radius):
            alpha = glow_alpha * (1 - (i - radius) / (glow_radius - radius))
            pygame.gfxdraw.aacircle(surface, center[0], center[1], i, color + (int(alpha),))
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)
    
    def _draw_glowing_square(self, surface, color, center_pos, size):
        rect = pygame.Rect(center_pos.x - size, center_pos.y - size, size*2, size*2)
        pygame.draw.rect(surface, color, rect, border_radius=2)
        # Simple glow
        glow_rect = rect.inflate(4, 4)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, color + (50,), s.get_rect(), border_radius=4)
        surface.blit(s, glow_rect.topleft)

    def _draw_glowing_triangle(self, surface, color, center_pos, size, velocity):
        if velocity.length() > 0.1:
            angle = velocity.angle_to(pygame.math.Vector2(1, 0))
        else:
            angle = 0
        
        points = []
        for i in range(3):
            theta = math.radians(angle + i * 120)
            point = center_pos + pygame.math.Vector2(size, 0).rotate_rad(theta)
            points.append((int(point.x), int(point.y)))
        
        # Glow
        pygame.gfxdraw.aapolygon(surface, points, color + (80,))
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()
        return None

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires the SDL_VIDEODRIVER to be set to a valid backend, not "dummy"
    # For example, you can comment out the os.environ line at the top of the file.
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Cannot run human player mode with SDL_VIDEODRIVER=dummy.")
        print("Comment out the os.environ.setdefault line to run this example.")
    else:
        env = GameEnv()
        obs, info = env.reset()
        done = False
        
        # Pygame setup for human play
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Gravity Cell Defense")
        clock = pygame.time.Clock()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            movement = 0 # No-op
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = keys[pygame.K_SPACE]
            shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            
            action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
            
            obs, reward, terminated, truncated, info = env.step(action)

            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            clock.tick(30) # Run at 30 FPS for human play

        pygame.quit()
        print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")