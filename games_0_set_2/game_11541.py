import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:11:58.448062
# Source Brief: brief_01541.md
# Brief Index: 1541
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for animals to encapsulate their state and drawing logic
class DreamAnimal:
    def __init__(self, pos, animal_type):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)
        self.type_info = animal_type
        self.size = self.type_info['base_size']
        self.mass = self.type_info['mass']
        self.color = self.type_info['color']
        self.max_size = self.type_info['base_size'] * 3
        self.age = 0

    def update(self):
        self.age += 1
        self.pos += self.vel
        self.vel *= 0.95  # Velocity damping for a viscous, dreamlike feel

    def grow(self):
        if self.size < self.max_size:
            self.size += 0.5
            self.mass += 0.1

    def draw(self, surface, time_step):
        # Brighter color for the core shape
        main_color = tuple(min(255, c + 50) for c in self.color)

        # Glow effect using additive blending for a vibrant look
        for i in range(4):
            alpha = 60 - i * 15
            radius = int(self.size + i * 3)
            glow_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.color, alpha), (radius, radius), radius)
            surface.blit(glow_surf, (int(self.pos.x - radius), int(self.pos.y - radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Morphing polygon shape for a surreal, non-static appearance
        num_points = 8
        points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            # Morphing effect using sine waves based on time and position
            offset = math.sin(time_step * 0.1 + i * 1.5) * self.size * 0.1
            radius = self.size + offset
            x = self.pos.x + radius * math.cos(angle)
            y = self.pos.y + radius * math.sin(angle)
            points.append((int(x), int(y)))

        if len(points) > 2:
            pygame.gfxdraw.aapolygon(surface, points, main_color)
            pygame.gfxdraw.filled_polygon(surface, points, main_color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Protect the dream sanctuary by creating and nurturing magical animals to hold back an encroaching tide."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to select an animal type. Press space to create an animal. Hold shift to grow the last created animal."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CENTER = pygame.Vector2(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    MAX_STEPS = 1000
    SANCTUARY_RADIUS = 180
    SAFE_ZONE_RADIUS = 30

    # --- Colors ---
    COLOR_BG = (15, 10, 40)
    COLOR_TIDE = [(20, 0, 30), (30, 5, 45), (40, 10, 60)]
    COLOR_SANCTUARY = (100, 200, 255)
    COLOR_SAFE_ZONE = (255, 220, 100)
    COLOR_TEXT = (240, 240, 240)
    
    # --- Animal Types ---
    ANIMAL_TYPES = [
        {'name': 'Glimmerfin', 'base_size': 12, 'mass': 1.0, 'color': (50, 255, 200)},
        {'name': 'Amberpede', 'base_size': 18, 'mass': 2.0, 'color': (255, 180, 50)},
        {'name': 'Void-hopper', 'base_size': 8, 'mass': 0.5, 'color': (200, 100, 255)},
        {'name': 'Sun-pupa', 'base_size': 25, 'mass': 3.5, 'color': (255, 100, 100)},
    ]

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)

        # State variables are initialized in reset()
        self.animals = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.tide_radius = 0
        self.base_tide_expansion_rate = 0
        self.unlocked_animal_indices = []
        self.selected_animal_type_idx = 0
        self.last_cloned_animal = None
        self.prev_space_held = False
        self.prev_movement = 0
        self.tide_anim_offset = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.tide_radius = self.SANCTUARY_RADIUS - 5
        self.base_tide_expansion_rate = 0.05
        self.animals.clear()
        self.particles.clear()
        
        self.unlocked_animal_indices = [0]
        self.selected_animal_type_idx = 0
        self.last_cloned_animal = None
        self.prev_space_held = True # Prevent clone on first step if action is [x,1,x]
        self.prev_movement = 0
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        old_tide_radius = self.tide_radius

        self._handle_actions(action)
        self._update_game_state()
        
        reward = self._calculate_reward(old_tide_radius)
        self.score += reward
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if terminated and not truncated: # Lost by tide
                reward -= 100.0 # Loss penalty
            elif truncated and not terminated: # Won by time
                reward += 100.0 # Win bonus
        
        # Gymnasium API requires terminated and truncated to be separate
        is_terminated = self.tide_radius <= self.SAFE_ZONE_RADIUS
        is_truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            is_terminated,
            is_truncated,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Animal Selection ---
        # Action happens on button press (change in movement value) to avoid rapid cycling
        if movement != 0 and movement != self.prev_movement:
            if movement in [1, 4]: # Up or Right for next
                self.selected_animal_type_idx = (self.selected_animal_type_idx + 1) % len(self.unlocked_animal_indices)
            elif movement in [2, 3]: # Down or Left for previous
                self.selected_animal_type_idx = (self.selected_animal_type_idx - 1 + len(self.unlocked_animal_indices)) % len(self.unlocked_animal_indices)
        self.prev_movement = movement

        # --- Cloning ---
        # Action happens on button press (transition from not held to held)
        if space_held and not self.prev_space_held:
            self._clone_animal()
            # SFX: clone_sound.play()
        self.prev_space_held = space_held

        # --- Resizing ---
        if shift_held and self.last_cloned_animal:
            self.last_cloned_animal.grow()
            # SFX: grow_sound.play()

    def _clone_animal(self):
        if not self.unlocked_animal_indices:
            return
            
        animal_idx = self.unlocked_animal_indices[self.selected_animal_type_idx]
        animal_type = self.ANIMAL_TYPES[animal_idx]

        # Find a valid spawn position away from other animals
        for _ in range(50): # Try 50 times to find a spot
            angle = self.np_random.uniform(0, 2 * math.pi)
            dist = self.np_random.uniform(self.SAFE_ZONE_RADIUS, self.SANCTUARY_RADIUS * 0.6)
            pos = self.CENTER + pygame.Vector2(math.cos(angle), math.sin(angle)) * dist
            
            is_overlapping = False
            for other in self.animals:
                if pos.distance_to(other.pos) < other.size + animal_type['base_size']:
                    is_overlapping = True
                    break
            if not is_overlapping:
                new_animal = DreamAnimal(pos, animal_type)
                self.animals.append(new_animal)
                self.last_cloned_animal = new_animal
                self._create_particle_burst(pos, animal_type['color'], 30)
                break

    def _update_game_state(self):
        self._update_progression()

        # Update tide based on animal pressure and natural expansion
        tide_pushback = 0
        for animal in self.animals:
            dist_to_center = animal.pos.distance_to(self.CENTER)
            if dist_to_center + animal.size > self.tide_radius:
                 overlap = (dist_to_center + animal.size) - self.tide_radius
                 tide_pushback += overlap * animal.mass * 0.02
        
        difficulty_multiplier = 1 + (self.steps // 200) * 0.2
        current_expansion_rate = self.base_tide_expansion_rate * difficulty_multiplier
        self.tide_radius -= current_expansion_rate
        self.tide_radius += tide_pushback
        self.tide_radius = min(self.tide_radius, self.SANCTUARY_RADIUS)

        # Update animals
        for animal in self.animals:
            animal.update()
            self._handle_animal_collisions(animal)
            self._handle_boundary_conditions(animal)
        
        # Memory management: remove oldest animals to prevent performance degradation
        if len(self.animals) > 50:
            removed_animal = self.animals.pop(0)
            if self.last_cloned_animal is removed_animal:
                self.last_cloned_animal = None

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.98
            p['life'] -= 1

    def _update_progression(self):
        # Unlock new animals every 250 steps
        if self.steps > 0 and self.steps % 250 == 0:
            num_unlocked = len(self.unlocked_animal_indices)
            if num_unlocked < len(self.ANIMAL_TYPES):
                self.unlocked_animal_indices.append(num_unlocked)
                # SFX: unlock_sound.play()
                self._create_particle_burst(self.CENTER, (255,255,255), 50)

    def _handle_animal_collisions(self, animal):
        for other in self.animals:
            if animal is other:
                continue
            dist_vec = animal.pos - other.pos
            dist = dist_vec.length()
            min_dist = animal.size + other.size
            if dist < min_dist and dist > 0:
                # Resolve overlap to prevent sticking
                overlap = (min_dist - dist) / 2
                push_vec = dist_vec.normalize() * overlap
                total_mass = animal.mass + other.mass
                animal.pos += push_vec * (other.mass / total_mass) * 2
                other.pos -= push_vec * (animal.mass / total_mass) * 2

    def _handle_boundary_conditions(self, animal):
        # Keep animals inside the main sanctuary circle
        dist_to_center = animal.pos.distance_to(self.CENTER)
        if dist_to_center > self.SANCTUARY_RADIUS - animal.size:
            normal = (self.CENTER - animal.pos).normalize()
            animal.vel = animal.vel.reflect(normal) * 0.8 # Reflect velocity with some energy loss
            animal.pos = self.CENTER - normal * (self.SANCTUARY_RADIUS - animal.size)

    def _calculate_reward(self, old_tide_radius):
        reward = 0.0
        tide_change = self.tide_radius - old_tide_radius
        
        if tide_change > 0.01: # Pushed back significantly
            reward += min(10.0, tide_change * 5)
        elif tide_change > -0.01: # Held steady
            reward += 0.1
        
        reward += max(-10.0, tide_change) # Small penalty/reward for tide movement
        
        return reward

    def _check_termination(self):
        return self.tide_radius <= self.SAFE_ZONE_RADIUS

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
            "tide_radius": self.tide_radius,
            "animals": len(self.animals),
            "unlocked_types": len(self.unlocked_animal_indices)
        }

    def _render_game(self):
        self.tide_anim_offset += 0.02
        
        # Render Tide with animated swirling effect
        for i, color in enumerate(self.COLOR_TIDE):
            radius = int(self.tide_radius + math.sin(self.tide_anim_offset + i) * 3)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(self.CENTER.x), int(self.CENTER.y), radius, color)

        # Render Sanctuary and Safe Zone boundaries with anti-aliasing
        pygame.gfxdraw.aacircle(self.screen, int(self.CENTER.x), int(self.CENTER.y), self.SANCTUARY_RADIUS, self.COLOR_SANCTUARY)
        pygame.gfxdraw.aacircle(self.screen, int(self.CENTER.x), int(self.CENTER.y), self.SANCTUARY_RADIUS-1, self.COLOR_SANCTUARY)
        
        # Safe zone glow
        for i in range(5):
             alpha = 80 - i*15
             pygame.gfxdraw.filled_circle(self.screen, int(self.CENTER.x), int(self.CENTER.y), self.SAFE_ZONE_RADIUS + i*2, (*self.COLOR_SAFE_ZONE, alpha))
        pygame.gfxdraw.filled_circle(self.screen, int(self.CENTER.x), int(self.CENTER.y), self.SAFE_ZONE_RADIUS, self.COLOR_SAFE_ZONE)

        # Render Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color_with_alpha = (*p['color'], alpha)
            surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color_with_alpha, (p['size'], p['size']), p['size'])
            self.screen.blit(surf, p['pos'] - pygame.Vector2(p['size'], p['size']), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Render Animals
        for animal in self.animals:
            animal.draw(self.screen, self.steps)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer/Steps
        time_text = self.font_small.render(f"TIME: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # Unlocked Abilities UI
        ui_y = self.SCREEN_HEIGHT - 30
        total_width = len(self.unlocked_animal_indices) * 40
        start_x = self.SCREEN_WIDTH // 2 - total_width // 2

        for i, animal_master_idx in enumerate(self.unlocked_animal_indices):
            animal_type = self.ANIMAL_TYPES[animal_master_idx]
            pos_x = start_x + i * 40
            pygame.draw.circle(self.screen, animal_type['color'], (pos_x, ui_y), 12)
            if i == self.selected_animal_type_idx:
                pygame.draw.circle(self.screen, self.COLOR_TEXT, (pos_x, ui_y), 15, 2)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            msg, color = ("SANCTUARY SECURE", self.COLOR_SAFE_ZONE) if self.steps >= self.MAX_STEPS and not self._check_termination() else ("THE TIDE CONSUMES", self.COLOR_TIDE[2])
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2)))

    def _create_particle_burst(self, pos, color, count):
        # SFX: particle_burst.play()
        for _ in range(count):
            life = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize() * self.np_random.uniform(1, 4),
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.uniform(1, 4)
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # The main loop is for human play, not required by the environment API.
    # It's included for testing and demonstration.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a visible driver for human play
    pygame.display.init()
    pygame.display.set_caption("Dream Sanctuary")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # no-op, released, released

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        mov_action = 0
        if keys[pygame.K_UP]: mov_action = 1
        elif keys[pygame.K_DOWN]: mov_action = 2
        elif keys[pygame.K_LEFT]: mov_action = 3
        elif keys[pygame.K_RIGHT]: mov_action = 4
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov_action, space_action, shift_action]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)

    env.close()