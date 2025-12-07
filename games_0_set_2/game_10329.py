import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:12:40.999704
# Source Brief: brief_00329.md
# Brief Index: 329
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a sci-fi tower defense game.
    The player must place magnetic defense units to protect a base from
    incoming waves of asteroids.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A sci-fi tower defense game where you place magnetic units to protect your base from incoming waves of asteroids."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place a defense unit and shift to cycle between unit types."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self._define_constants()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans-serif", 18)
        self.font_large = pygame.font.SysFont("sans-serif", 24)

        self.render_mode = render_mode
        self.np_random = None # Will be seeded in reset

    def _define_constants(self):
        """Define all game constants for easy tuning."""
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 3000 # Increased from brief to allow for 20 waves
        self.WIN_WAVE = 20

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_BASE = (0, 100, 50)
        self.COLOR_BASE_GLOW = (0, 150, 75)
        self.COLOR_HEALTH = (0, 220, 110)
        self.COLOR_HEALTH_BG = (50, 0, 0)
        self.COLOR_ASTEROID = (220, 50, 50)
        self.COLOR_ASTEROID_GLOW = (255, 100, 100)
        self.COLOR_RESOURCE = (255, 200, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_CURSOR_INVALID = (255, 0, 0)

        # Base properties
        self.BASE_MAX_HEALTH = 100
        self.BASE_RECT = pygame.Rect(self.SCREEN_WIDTH // 2 - 50, self.SCREEN_HEIGHT - 30, 100, 20)

        # Unit specifications
        self.UNIT_SPECS = [
            {'name': 'Magnetron', 'cost': 25, 'radius': 100, 'strength': 0.08, 'damage': 15, 'color': (50, 150, 255), 'unlock_wave': 1},
            {'name': 'Pulverizer', 'cost': 50, 'radius': 80, 'strength': 0.04, 'damage': 40, 'color': (255, 150, 50), 'unlock_wave': 5},
            {'name': 'Graviton', 'cost': 75, 'radius': 150, 'strength': 0.12, 'damage': 10, 'color': (150, 50, 255), 'unlock_wave': 10},
            {'name': 'Singularity', 'cost': 100, 'radius': 90, 'strength': 0.1, 'damage': 25, 'aoe_radius': 70, 'color': (255, 255, 255), 'unlock_wave': 15}
        ]
        self.CURSOR_SPEED = 8
        self.MIN_PLACEMENT_DISTANCE = 30

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Seed the python random module for legacy code
        if seed is not None:
            random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.base_health = self.BASE_MAX_HEALTH
        self.resources = 100
        self.wave_number = 1
        
        self.units = []
        self.asteroids = []
        self.particles = []

        self.cursor_pos = pygame.math.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.selected_unit_index = 0
        self.unlocked_unit_indices = [0]
        self.placement_error_timer = 0

        self.last_shift_state = 0
        self.last_space_state = 0

        self.wave_in_progress = False
        self.wave_cooldown = 120 # 4 seconds at 30fps

        self._start_new_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        self.placement_error_timer = max(0, self.placement_error_timer - 1)

        if not self.game_over:
            self._handle_input(action)
            
            reward += self._update_asteroids()
            self._update_particles()
            reward += self._update_wave_logic()

            self._check_termination_conditions()
            if self.game_over:
                if self.base_health <= 0 or self.steps >= self.MAX_STEPS:
                    reward -= 100 # Lose
                elif self.wave_number > self.WIN_WAVE:
                    reward += 100 # Win
        
        self.score += reward
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Move cursor
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.SCREEN_WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.SCREEN_HEIGHT - self.BASE_RECT.height - 10)

        # Cycle unit type (on press)
        if shift_held and not self.last_shift_state:
            self.selected_unit_index = (self.selected_unit_index + 1) % len(self.unlocked_unit_indices)
            # sfx: UI_Cycle.wav

        # Place unit (on press)
        if space_held and not self.last_space_state:
            spec = self.UNIT_SPECS[self.unlocked_unit_indices[self.selected_unit_index]]
            can_place = True
            if self.resources < spec['cost']:
                can_place = False
            
            if can_place:
                for unit in self.units:
                    if self.cursor_pos.distance_to(unit['pos']) < self.MIN_PLACEMENT_DISTANCE:
                        can_place = False
                        break
            
            if can_place:
                self.resources -= spec['cost']
                self.units.append({
                    'pos': self.cursor_pos.copy(),
                    'type_index': self.unlocked_unit_indices[self.selected_unit_index],
                    'spec': spec,
                })
                # sfx: Place_Unit.wav
            else:
                self.placement_error_timer = 15 # Flash cursor red for 0.5s
                # sfx: Error.wav

        self.last_space_state = space_held
        self.last_shift_state = shift_held
        
    def _update_asteroids(self):
        reward = 0
        asteroids_to_remove = []
        units_to_remove = []

        for i, asteroid in enumerate(self.asteroids):
            # Apply magnetic force from units
            total_force = pygame.math.Vector2(0, 0)
            for unit in self.units:
                dist_vec = unit['pos'] - asteroid['pos']
                dist = dist_vec.length()
                if dist < unit['spec']['radius'] and dist > 0:
                    force = dist_vec.normalize() * unit['spec']['strength'] * (1 - dist / unit['spec']['radius'])
                    total_force += force
            
            asteroid['vel'] += total_force
            asteroid['pos'] += asteroid['vel']

            # Check collision with units
            collided_with_unit = False
            for j, unit in enumerate(self.units):
                if asteroid['pos'].distance_to(unit['pos']) < asteroid['size']:
                    collided_with_unit = True
                    asteroid['health'] -= unit['spec']['damage']
                    self._create_explosion(asteroid['pos'], 5, self.COLOR_ASTEROID, 0.5)
                    # sfx: Hit_Damage.wav
                    
                    # Handle Singularity (AoE) unit
                    if unit['spec'].get('aoe_radius'):
                        self._create_explosion(unit['pos'], 40, unit['spec']['color'], 2.0)
                        # sfx: Explosion_Large.wav
                        for other_ast_idx, other_ast in enumerate(self.asteroids):
                            if i != other_ast_idx and other_ast['pos'].distance_to(unit['pos']) < unit['spec']['aoe_radius']:
                                other_ast['health'] -= unit['spec']['damage']
                        if j not in units_to_remove:
                            units_to_remove.append(j)

                    if asteroid['health'] <= 0 and i not in asteroids_to_remove:
                        asteroids_to_remove.append(i)
                        reward += 0.1
                        self.resources += 5 + int(asteroid['initial_size'] / 5)
                        self._create_explosion(asteroid['pos'], int(asteroid['initial_size']), self.COLOR_ASTEROID, 1.0)
                        # sfx: Explosion_Small.wav
                    break
            
            # Check collision with base
            if not collided_with_unit and self.BASE_RECT.collidepoint(asteroid['pos'].x, asteroid['pos'].y):
                damage = int(asteroid['size'])
                self.base_health -= damage
                reward -= 0.5 * damage
                if i not in asteroids_to_remove:
                    asteroids_to_remove.append(i)
                self._create_explosion(asteroid['pos'], 15, self.COLOR_BASE_GLOW, 1.5)
                # sfx: Base_Hit.wav

            # Check out of bounds
            if not self.screen.get_rect().inflate(100, 100).collidepoint(asteroid['pos'].x, asteroid['pos'].y):
                 if i not in asteroids_to_remove:
                    asteroids_to_remove.append(i)

        # Remove destroyed entities
        for i in sorted(list(set(asteroids_to_remove)), reverse=True):
            del self.asteroids[i]
        for i in sorted(list(set(units_to_remove)), reverse=True):
            del self.units[i]

        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['vel'] *= 0.98 # Damping

    def _update_wave_logic(self):
        reward = 0
        if self.wave_in_progress and not self.asteroids:
            self.wave_in_progress = False
            self.wave_cooldown = 120 # 4 seconds
            reward += 1.0
            # sfx: Wave_Complete.wav
        
        if not self.wave_in_progress and self.wave_number <= self.WIN_WAVE:
            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0:
                self.wave_number += 1
                if self.wave_number <= self.WIN_WAVE:
                    self._start_new_wave()
        return reward
    
    def _start_new_wave(self):
        self.wave_in_progress = True
        
        # Unlock new units
        self.unlocked_unit_indices = [i for i, spec in enumerate(self.UNIT_SPECS) if self.wave_number >= spec['unlock_wave']]
        self.selected_unit_index = min(self.selected_unit_index, len(self.unlocked_unit_indices) - 1)

        # Wave progression
        num_asteroids = 5 + self.wave_number
        speed = 0.5 + self.wave_number * 0.05
        max_size = 10 + (self.wave_number // 5) * 2

        for _ in range(num_asteroids):
            edge = random.choice(['top', 'left', 'right'])
            if edge == 'top':
                pos = pygame.math.Vector2(random.uniform(0, self.SCREEN_WIDTH), -20)
            elif edge == 'left':
                pos = pygame.math.Vector2(-20, random.uniform(0, self.SCREEN_HEIGHT * 0.7))
            else: # right
                pos = pygame.math.Vector2(self.SCREEN_WIDTH + 20, random.uniform(0, self.SCREEN_HEIGHT * 0.7))
            
            target = pygame.math.Vector2(self.BASE_RECT.centerx + random.uniform(-50, 50), self.BASE_RECT.centery)
            vel = (target - pos).normalize() * speed
            size = random.uniform(5, max_size)

            self.asteroids.append({
                'pos': pos,
                'vel': vel,
                'size': size,
                'initial_size': size,
                'health': size * 2
            })

    def _create_explosion(self, pos, num_particles, color, speed_mult):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifetime': random.randint(20, 40),
                'color': color,
                'size': random.uniform(1, 3)
            })

    def _check_termination_conditions(self):
        if self.base_health <= 0:
            self.base_health = 0
            self.game_over = True
        if self.wave_number > self.WIN_WAVE and not self.asteroids:
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        # Pygame uses (width, height), but Gymnasium expects (height, width) for image observations.
        # `pygame.surfarray.array3d` creates a (width, height, 3) array.
        # We transpose it to (height, width, 3) to match the observation space.
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw base
        pygame.draw.rect(self.screen, self.COLOR_BASE_GLOW, self.BASE_RECT.inflate(6, 6), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASE, self.BASE_RECT, border_radius=5)

        # Draw magnetic fields
        for unit in self.units:
            for asteroid in self.asteroids:
                if unit['pos'].distance_to(asteroid['pos']) < unit['spec']['radius']:
                    pygame.draw.aaline(self.screen, unit['spec']['color'], unit['pos'], asteroid['pos'], 2)

        # Draw units
        for unit in self.units:
            spec = unit['spec']
            p1 = unit['pos'] + pygame.math.Vector2(0, -10)
            p2 = unit['pos'] + pygame.math.Vector2(-8.66, 5)
            p3 = unit['pos'] + pygame.math.Vector2(8.66, 5)
            self._draw_antialiased_polygon(self.screen, spec['color'], [p1, p2, p3], glow=True)
            
        # Draw asteroids
        for asteroid in self.asteroids:
            self._draw_antialiased_circle(self.screen, self.COLOR_ASTEROID, asteroid['pos'], asteroid['size'], self.COLOR_ASTEROID_GLOW)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 40))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, p['pos'] - pygame.math.Vector2(p['size'], p['size']))

        # Draw cursor
        cursor_color = self.COLOR_CURSOR_INVALID if self.placement_error_timer > 0 else self.COLOR_CURSOR
        pygame.draw.circle(self.screen, cursor_color, (int(self.cursor_pos.x), int(self.cursor_pos.y)), 10, 1)
        pygame.draw.line(self.screen, cursor_color, self.cursor_pos - (15,0), self.cursor_pos - (5,0))
        pygame.draw.line(self.screen, cursor_color, self.cursor_pos + (5,0), self.cursor_pos + (15,0))
        pygame.draw.line(self.screen, cursor_color, self.cursor_pos - (0,15), self.cursor_pos - (0,5))
        pygame.draw.line(self.screen, cursor_color, self.cursor_pos + (0,5), self.cursor_pos + (0,15))

    def _render_ui(self):
        # Resources
        res_text = self.font_large.render(f"ENERGY: {self.resources}", True, self.COLOR_RESOURCE)
        self.screen.blit(res_text, (10, 10))

        # Selected Unit Info
        spec = self.UNIT_SPECS[self.unlocked_unit_indices[self.selected_unit_index]]
        unit_text = self.font_small.render(f"SELECT: {spec['name']} (Cost: {spec['cost']})", True, self.COLOR_TEXT)
        self.screen.blit(unit_text, (10, 40))

        # Wave number
        wave_text = self.font_large.render(f"WAVE: {self.wave_number} / {self.WIN_WAVE}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        # Base health bar
        health_pct = self.base_health / self.BASE_MAX_HEALTH
        bar_width = 200
        bar_height = 20
        bar_x = self.SCREEN_WIDTH // 2 - bar_width // 2
        bar_y = self.SCREEN_HEIGHT - bar_height - 5
        
        bg_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        fg_width = int(bar_width * health_pct)
        fg_rect = pygame.Rect(bar_x, bar_y, fg_width, bar_height)

        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, bg_rect, border_radius=4)
        if fg_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH, fg_rect, border_radius=4)
        
        health_text = self.font_small.render(f"{self.base_health}/{self.BASE_MAX_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (bar_x + bar_width/2 - health_text.get_width()/2, bar_y + bar_height/2 - health_text.get_height()/2))
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.wave_number > self.WIN_WAVE and not self.asteroids:
                msg = "VICTORY"
                color = self.COLOR_RESOURCE
            else:
                msg = "GAME OVER"
                color = self.COLOR_ASTEROID
            
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH/2 - end_text.get_width()/2, self.SCREEN_HEIGHT/2 - end_text.get_height()/2))

    def _draw_antialiased_circle(self, surface, color, center, radius, glow_color=None):
        center_int = (int(center.x), int(center.y))
        radius_int = int(max(1, radius))
        if glow_color:
            pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius_int + 2, glow_color + (50,))
            pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius_int + 2, glow_color + (50,))
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius_int, color)
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius_int, color)

    def _draw_antialiased_polygon(self, surface, color, points, glow=False):
        points_int = [(int(p.x), int(p.y)) for p in points]
        if glow:
            glow_color = color + (80,)
            pygame.gfxdraw.filled_polygon(surface, points_int, glow_color)
            pygame.gfxdraw.aapolygon(surface, points_int, glow_color)
        pygame.gfxdraw.filled_polygon(surface, points_int, color)
        pygame.gfxdraw.aapolygon(surface, points_int, color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "base_health": self.base_health,
            "units_placed": len(self.units),
            "asteroids_on_screen": len(self.asteroids)
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    os.environ.pop("SDL_VIDEODRIVER", None)
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroid Magnetism")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Final Wave: {info['wave']}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()