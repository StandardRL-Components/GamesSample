import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:15:17.273050
# Source Brief: brief_01577.md
# Brief Index: 1577
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

# Helper classes for game objects
class Particle:
    def __init__(self, pos, vel, color, radius, lifetime):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.color = color
        self.radius = radius
        self.lifetime = lifetime
        self.initial_lifetime = lifetime

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1
        self.vel *= 0.98  # Damping

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.initial_lifetime))
            current_radius = int(self.radius * (self.lifetime / self.initial_lifetime))
            if current_radius > 0:
                color = (*self.color, alpha)
                temp_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (current_radius, current_radius), current_radius)
                surface.blit(temp_surf, self.pos - pygame.math.Vector2(current_radius, current_radius), special_flags=pygame.BLEND_RGBA_ADD)

class Lure:
    def __init__(self, pos):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(0, 0.5) # Slight initial downward velocity
        self.lifetime = 300 # 10 seconds at 30fps
        self.radius = 8
        self.light_radius = 120
        self.color = (100, 255, 200)

    def update(self):
        self.vel.y += 0.02  # Gravity
        self.vel *= 0.97    # Water resistance
        self.pos += self.vel
        self.lifetime -= 1

    def draw(self, surface, game):
        # Draw glow
        game._draw_glowing_circle(surface, self.color, self.pos, self.radius, 0.6)
        # Draw core
        pygame.draw.circle(surface, (200, 255, 230), self.pos, self.radius * 0.6)

class Predator:
    PREDATOR_TYPES = {
        'small': {'size': 6, 'color': (255, 150, 50), 'speed': 1.8, 'eats': [], 'lure_attraction': 1.0, 'reward': 2},
        'medium': {'size': 10, 'color': (255, 100, 100), 'speed': 1.5, 'eats': ['small'], 'lure_attraction': 0.7, 'reward': 5},
        'large': {'size': 16, 'color': (255, 50, 150), 'speed': 1.2, 'eats': ['small', 'medium'], 'lure_attraction': 0.3, 'reward': 10}
    }

    def __init__(self, p_type, pos):
        self.type = p_type
        self.stats = self.PREDATOR_TYPES[p_type]
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.stats['speed']
        self.size = self.stats['size']
        self.color = self.stats['color']
        self.max_speed = self.stats['speed']
        self.wander_angle = random.uniform(0, 2 * math.pi)
        self.state = 'wandering'
        self.target = None

    def update(self, lures, predators, bounds):
        target_force = self._get_target_force(lures, predators)
        flock_force = self._get_flock_force(predators)
        
        if target_force.length() > 0:
            self.state = 'chasing'
            steering = target_force
        else:
            self.state = 'wandering'
            steering = self._get_wander_force()
        
        steering += flock_force * 0.5 # Add flocking behavior with a weight

        if steering.length() > 0:
            steering = steering.normalize() * self.max_speed
        
        accel = steering - self.vel
        if accel.length() > 0.5: # Limit acceleration
             accel = accel.normalize() * 0.5

        self.vel += accel
        if self.vel.length() > self.max_speed:
            self.vel.scale_to_length(self.max_speed)

        self.pos += self.vel
        self._check_bounds(bounds)
    
    def _get_target_force(self, lures, predators):
        # Find best lure
        best_lure = None
        min_dist_sq = float('inf')
        for lure in lures:
            dist_sq = self.pos.distance_squared_to(lure.pos)
            if dist_sq < lure.light_radius ** 2 and dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_lure = lure
        
        if best_lure:
            return (best_lure.pos - self.pos) * self.stats['lure_attraction']

        # Find best prey
        best_prey = None
        min_dist_sq = (self.size * 15)**2 # Vision range for prey
        for other in predators:
            if other.type in self.stats['eats']:
                dist_sq = self.pos.distance_squared_to(other.pos)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_prey = other
        
        if best_prey:
            return best_prey.pos - self.pos
        
        return pygame.math.Vector2(0, 0)

    def _get_wander_force(self):
        self.wander_angle += random.uniform(-0.3, 0.3)
        wander_force = pygame.math.Vector2(math.cos(self.wander_angle), math.sin(self.wander_angle))
        return wander_force.normalize() * self.max_speed * 0.5

    def _get_flock_force(self, predators):
        NEIGHBOR_RADIUS_SQ = (self.size * 10)**2
        separation_force = pygame.math.Vector2()
        alignment_force = pygame.math.Vector2()
        cohesion_force = pygame.math.Vector2()
        neighbor_count = 0

        for other in predators:
            if other is not self:
                dist_sq = self.pos.distance_squared_to(other.pos)
                if dist_sq < NEIGHBOR_RADIUS_SQ:
                    # Separation
                    if dist_sq < (self.size * 2.5)**2 and dist_sq > 0:
                        diff = self.pos - other.pos
                        separation_force += diff / dist_sq
                    # Alignment & Cohesion
                    alignment_force += other.vel
                    cohesion_force += other.pos
                    neighbor_count += 1
        
        if neighbor_count > 0:
            alignment_force /= neighbor_count
            if alignment_force.length() > 0:
                alignment_force.scale_to_length(self.max_speed)
            
            cohesion_force /= neighbor_count
            cohesion_force = cohesion_force - self.pos
            if cohesion_force.length() > 0:
                cohesion_force.scale_to_length(self.max_speed)

        total_force = separation_force * 1.5 + alignment_force * 1.0 + cohesion_force * 1.0
        return total_force

    def _check_bounds(self, bounds):
        buffer = self.size
        if self.pos.x < buffer: self.pos.x = buffer; self.vel.x *= -1
        if self.pos.x > bounds[0] - buffer: self.pos.x = bounds[0] - buffer; self.vel.x *= -1
        if self.pos.y < buffer: self.pos.y = buffer; self.vel.y *= -1
        if self.pos.y > bounds[1] - buffer: self.pos.y = bounds[1] - buffer; self.vel.y *= -1

    def draw(self, surface, game):
        # Body
        points = []
        angle = self.vel.angle_to(pygame.math.Vector2(1, 0))
        if self.type == 'small':
            points = [pygame.math.Vector2(self.size, 0), pygame.math.Vector2(-self.size, -self.size/2), pygame.math.Vector2(-self.size, self.size/2)]
        elif self.type == 'medium':
            points = [pygame.math.Vector2(self.size, 0), pygame.math.Vector2(-self.size/2, -self.size), pygame.math.Vector2(-self.size, 0), pygame.math.Vector2(-self.size/2, self.size)]
        else: # large
            points = [pygame.math.Vector2(self.size*1.2, 0), pygame.math.Vector2(-self.size, -self.size*0.8), pygame.math.Vector2(-self.size*0.7, 0), pygame.math.Vector2(-self.size, self.size*0.8)]
        
        rotated_points = [p.rotate(-angle) + self.pos for p in points]
        
        pygame.draw.polygon(surface, self.color, rotated_points)
        pygame.draw.aalines(surface, (255,255,255), True, rotated_points)

        # Eye
        eye_pos = pygame.math.Vector2(self.size * 0.5, 0).rotate(-angle) + self.pos
        game._draw_glowing_circle(surface, (255, 255, 255), eye_pos, 2, 0.8)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Deploy glowing lures into the abyss to capture exotic deep-sea creatures. "
        "Fulfill capture quotas to balance the ecosystem before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the targeting cursor. "
        "Press space to deploy a lure to trap creatures."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400
        
        # Colors
        self.COLOR_BG = (5, 10, 25)
        self.COLOR_CURSOR = (150, 255, 255)
        self.COLOR_KRAKEN = (80, 20, 120)
        self.COLOR_KRAKEN_EYE = (255, 0, 50)
        self.COLOR_UI_TEXT = (200, 220, 255)
        self.COLOR_UI_BAR_BG = (50, 50, 80)
        self.COLOR_UI_BAR_FILL = (100, 200, 255)

        # Game constants
        self.MAX_STEPS = 2000
        self.KRAKEN_START_TIME = 1800 # 60 seconds at 30 FPS
        self.CURSOR_SPEED = 8
        self.LURE_COOLDOWN_TIME = 30 # 1 second

        self.INITIAL_PREDATORS = {'small': 20, 'medium': 10, 'large': 4}
        self.TARGET_COUNTS = {'small': 15, 'medium': 8, 'large': 3}

        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 15, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = pygame.math.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        self.lures = []
        self.predators = []
        self.particles = []
        self.kraken_timer = self.KRAKEN_START_TIME
        self.lure_cooldown = 0
        self.captured_counts = {p_type: 0 for p_type in Predator.PREDATOR_TYPES}
        self.last_action_history = deque(maxlen=5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = pygame.math.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        self.lures.clear()
        self.particles.clear()
        self.kraken_timer = self.KRAKEN_START_TIME
        self.lure_cooldown = 0
        self.captured_counts = {p_type: 0 for p_type in Predator.PREDATOR_TYPES}
        self.last_action_history.clear()

        self._spawn_predators()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.last_action_history.append(action[0])
        
        prev_balance = self._calculate_ecosystem_balance()

        self._handle_input(action)
        
        self._update_lures()
        self._update_predators()
        self._update_particles()
        
        events = self._handle_interactions()

        reward = self._calculate_reward(events, prev_balance)
        self.score += reward
        
        terminated = self._check_termination()
        truncated = False
        if self.steps >= self.MAX_STEPS:
            terminated = True # Per user request, but can also be truncated
        
        if terminated:
            self.game_over = True
            if self._is_win_condition_met():
                reward += 100
            else:
                reward -= 100
            self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Move cursor
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

        # Launch lure
        if self.lure_cooldown > 0:
            self.lure_cooldown -= 1

        if space_held and self.lure_cooldown == 0:
            self.lures.append(Lure(self.cursor_pos))
            self.lure_cooldown = self.LURE_COOLDOWN_TIME
            # Sound placeholder: # sfx_lure_launch()
            for _ in range(20):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 3)
                vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
                self.particles.append(Particle(self.cursor_pos, vel, (150, 255, 200), random.randint(2,4), 30))

    def _update_lures(self):
        for lure in self.lures:
            lure.update()
        self.lures = [lure for lure in self.lures if lure.lifetime > 0 and 0 < lure.pos.x < self.WIDTH and 0 < lure.pos.y < self.HEIGHT]

    def _update_predators(self):
        for predator in self.predators:
            predator.update(self.lures, self.predators, (self.WIDTH, self.HEIGHT))

    def _update_particles(self):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifetime > 0]

    def _handle_interactions(self):
        events = {'trapped': [], 'eaten': []}
        
        predators_to_remove = set()

        # Lure trapping
        for lure in self.lures:
            for i, predator in enumerate(self.predators):
                if i in predators_to_remove: continue
                if predator.pos.distance_to(lure.pos) < predator.size + lure.radius:
                    predators_to_remove.add(i)
                    events['trapped'].append(predator)
                    # Sound placeholder: # sfx_trap()
                    for _ in range(30):
                        angle = random.uniform(0, 2 * math.pi)
                        speed = random.uniform(2, 5)
                        vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
                        self.particles.append(Particle(predator.pos, vel, predator.color, random.randint(2,5), 40))

        # Predation
        for i, predator in enumerate(self.predators):
            if i in predators_to_remove: continue
            for j, prey in enumerate(self.predators):
                if i == j or j in predators_to_remove: continue
                if prey.type in predator.stats['eats']:
                    if predator.pos.distance_to(prey.pos) < predator.size:
                        predators_to_remove.add(j)
                        events['eaten'].append(prey)
                        # Sound placeholder: # sfx_eat()
                        predator.size += 0.5 # Predator grows slightly
                        for _ in range(15):
                            angle = random.uniform(0, 2 * math.pi)
                            speed = random.uniform(1, 3)
                            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
                            self.particles.append(Particle(prey.pos, vel, prey.color, random.randint(1,3), 20))

        if predators_to_remove:
            self.predators = [p for i, p in enumerate(self.predators) if i not in predators_to_remove]
        
        return events

    def _calculate_reward(self, events, prev_balance):
        reward = 0
        
        # Event-based rewards
        for predator in events['trapped']:
            self.captured_counts[predator.type] += 1
            reward += predator.stats['reward']
        
        for _ in events['eaten']:
            reward -= 1 # Penalty for letting predators get eaten

        # Continuous balance reward
        new_balance = self._calculate_ecosystem_balance()
        reward += (new_balance - prev_balance) * 20

        # Kraken penalty
        self.kraken_timer -= 1
        reward -= 0.05
        
        # Penalty for inaction
        if len(self.last_action_history) == 5 and all(a == 0 for a in self.last_action_history):
            reward -= 0.1

        return reward

    def _calculate_ecosystem_balance(self):
        total_progress = 0
        num_types = len(self.TARGET_COUNTS)
        for p_type, target in self.TARGET_COUNTS.items():
            progress = min(1.0, self.captured_counts[p_type] / target)
            total_progress += progress
        return total_progress / num_types if num_types > 0 else 0
    
    def _is_win_condition_met(self):
        return self._calculate_ecosystem_balance() >= 1.0

    def _check_termination(self):
        if self._is_win_condition_met():
            return True
        if self.kraken_timer <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _spawn_predators(self):
        self.predators.clear()
        for p_type, count in self.INITIAL_PREDATORS.items():
            for _ in range(count):
                pos = (random.randint(50, self.WIDTH - 50), random.randint(50, self.HEIGHT - 50))
                self.predators.append(Predator(p_type, pos))
    
    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "kraken_timer": self.kraken_timer,
            "ecosystem_balance": self._calculate_ecosystem_balance(),
            "captured_counts": self.captured_counts,
        }

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_kraken()
        
        for p in self.particles: p.draw(self.screen)
        for lure in self.lures: lure.draw(self.screen, self)
        for predator in self.predators: predator.draw(self.screen, self)
        
        self._render_cursor()
        self._render_ui()

    def _render_background(self):
        # Draw some static, dark abyssal flora for depth
        for i in range(10):
            seed = i * 12345
            x = (seed * 17) % self.WIDTH
            y_base = self.HEIGHT
            height = 50 + (seed * 19) % 100
            width = 10 + (seed * 23) % 20
            color = (15, 25, 45)
            pygame.draw.rect(self.screen, color, (x, y_base - height, width, height))

    def _render_kraken(self):
        progress = 1.0 - max(0, self.kraken_timer / self.KRAKEN_START_TIME)
        if progress == 0: return

        alpha = int(progress * 150)
        scale = 0.5 + progress * 0.8

        # A simple polygon for a tentacle
        tentacle_points = [
            (0, 0), (20, -50), (30, -120), (10, -110), (0, -40), (-10, -60)
        ]
        
        kraken_color = (*self.COLOR_KRAKEN, alpha)
        eye_color = (*self.COLOR_KRAKEN_EYE, min(255, int(progress * 300)))

        # Draw a few tentacles rising from the bottom
        for i in range(3):
            base_x = self.WIDTH // 2 + (i - 1) * 200
            base_y = self.HEIGHT + 100
            
            scaled_points = []
            for x, y in tentacle_points:
                # Apply some sway based on time
                sway = math.sin(self.steps * 0.02 + i) * 20 * progress
                nx = base_x + (x + sway * (-y / 120)) * scale
                ny = base_y + y * scale
                scaled_points.append((nx, ny))
            
            if len(scaled_points) > 2:
                temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                pygame.draw.polygon(temp_surf, kraken_color, scaled_points)
                self.screen.blit(temp_surf, (0,0))
        
        # Eyes glowing in the dark
        if progress > 0.5:
            eye_y = self.HEIGHT - 20 - (progress - 0.5) * 50
            self._draw_glowing_circle(self.screen, eye_color, (self.WIDTH/2 - 40, eye_y), 8 * progress, 1.0)
            self._draw_glowing_circle(self.screen, eye_color, (self.WIDTH/2 + 40, eye_y), 8 * progress, 1.0)

    def _render_cursor(self):
        pos = self.cursor_pos
        size = 12
        line_width = 2
        # Glow
        self._draw_glowing_circle(self.screen, self.COLOR_CURSOR, pos, size * 1.5, 0.3)
        # Crosshair
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos.x - size, pos.y), (pos.x + size, pos.y), line_width)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos.x, pos.y - size), (pos.x, pos.y + size), line_width)
        # Cooldown indicator
        if self.lure_cooldown > 0:
            ratio = self.lure_cooldown / self.LURE_COOLDOWN_TIME
            pygame.draw.arc(self.screen, self.COLOR_CURSOR, (pos.x - size, pos.y - size, size*2, size*2), 0, 2 * math.pi * ratio, 2)


    def _render_ui(self):
        # Ecosystem Balance Bar
        bar_width, bar_height = 200, 15
        bar_x, bar_y = self.WIDTH // 2 - bar_width // 2, 10
        balance = self._calculate_ecosystem_balance()
        fill_width = int(bar_width * balance)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FILL, (bar_x, bar_y, fill_width, bar_height), border_radius=3)
        balance_text = self.font_small.render("ECO-BALANCE", True, self.COLOR_UI_TEXT)
        self.screen.blit(balance_text, (bar_x + bar_width // 2 - balance_text.get_width() // 2, bar_y + bar_height))

        # Captured Predator Counts
        start_x = 20
        for i, (p_type, stats) in enumerate(Predator.PREDATOR_TYPES.items()):
            x = start_x + i * 110
            y = 20
            # Icon
            dummy_pred = Predator(p_type, (0,0))
            dummy_pred.vel = pygame.math.Vector2(1,0) # Face right
            dummy_pred.pos = pygame.math.Vector2(x, y)
            dummy_pred.draw(self.screen, self)
            # Text
            count_text = f"{self.captured_counts[p_type]}/{self.TARGET_COUNTS[p_type]}"
            text_surf = self.font_large.render(count_text, True, self.COLOR_UI_TEXT)
            self.screen.blit(text_surf, (x + stats['size'] + 15, y - text_surf.get_height()//2))

        # Game Over Text
        if self.game_over:
            if self._is_win_condition_met():
                msg = "ECOSYSTEM STABILIZED"
                color = (150, 255, 150)
            else:
                msg = "THE ABYSS CONSUMES ALL"
                color = (255, 100, 100)
            
            text_surf = self.font_large.render(msg, True, color)
            pos = (self.WIDTH // 2 - text_surf.get_width() // 2, self.HEIGHT // 2 - text_surf.get_height() // 2)
            self.screen.blit(text_surf, pos)

    def _draw_glowing_circle(self, surface, color, pos, radius, max_alpha_factor):
        if radius <= 0: return
        max_alpha = int(255 * max_alpha_factor)
        glow_radius = int(radius * 2.5)
        
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        
        num_layers = 5
        for i in range(num_layers, 0, -1):
            alpha = int(max_alpha * (1 / (num_layers - i + 1))**2)
            layer_radius = radius + (glow_radius - radius) * (i / num_layers)
            layer_color = (*color, alpha)
            pygame.draw.circle(temp_surf, layer_color, (glow_radius, glow_radius), int(layer_radius))
        
        surface.blit(temp_surf, (int(pos.x - glow_radius), int(pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.draw.circle(surface, color, (int(pos.x), int(pos.y)), int(radius))

    def close(self):
        pygame.quit()
    
    def render(self):
        # This method is not used by the environment's step/reset logic, but can be useful for external rendering.
        # Since we use "rgb_array", _get_observation handles all rendering.
        return self._get_observation()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run, you'll need to `pip install pygame`
    # It's recommended to run this in a virtual environment.
    
    # Un-comment the line below to run with a visible display
    # os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    try:
        env = GameEnv()
        obs, info = env.reset()
        
        # Use a separate display for manual play if not running headless
        if os.environ.get("SDL_VIDEODRIVER", "dummy") != "dummy":
            manual_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
            pygame.display.set_caption("Deep Sea Terraformer")
        else:
            print("Running in headless mode. No window will be displayed.")
            print("To play manually, un-comment the SDL_VIDEODRIVER line in the __main__ block.")
            manual_screen = None

        running = True
        total_reward = 0
        
        # Map pygame keys to environment actions
        key_map = {
            pygame.K_UP: 1,
            pygame.K_DOWN: 2,
            pygame.K_LEFT: 3,
            pygame.K_RIGHT: 4,
        }

        clock = pygame.time.Clock()

        while running:
            movement_action = 0
            space_action = 0
            shift_action = 0

            # This event loop is for manual play, it won't run in a typical training loop.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            keys = pygame.key.get_pressed()
            
            for key, move_val in key_map.items():
                if keys[key]:
                    movement_action = move_val
                    break # Prioritize one movement direction
            
            if keys[pygame.K_SPACE]:
                space_action = 1
            
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift_action = 1

            action = [movement_action, space_action, shift_action]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if manual_screen:
                # Render the observation to the manual display
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                manual_screen.blit(surf, (0, 0))
                
                # Display score for manual play
                score_text = env.font_small.render(f"Total Reward: {total_reward:.2f}", True, (255, 255, 255))
                manual_screen.blit(score_text, (10, env.HEIGHT - 20))

                pygame.display.flip()

            if terminated or truncated:
                print(f"Episode finished. Total reward: {total_reward}")
                print("Press 'R' to reset or 'ESC' to quit.")
            
            clock.tick(30) # Run at 30 FPS
    
    finally:
        if 'env' in locals():
            env.close()