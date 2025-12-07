import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:05:08.358958
# Source Brief: brief_00799.md
# Brief Index: 799
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Classes for Game Entities ---

class Particle:
    """A simple class for a visual particle effect."""
    def __init__(self, pos, vel, radius, color, lifetime):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.radius = radius
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1
        self.radius -= 0.1

    def draw(self, surface):
        if self.lifetime > 0 and self.radius > 0:
            alpha = int(255 * (self.lifetime / self.max_lifetime))
            # Use gfxdraw for anti-aliased circles for a softer look
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), (*self.color, alpha))


class Herbivore:
    """Represents a single herbivore in the herd."""
    def __init__(self, pos, color):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)
        self.color = color
        self.size = 8
        self.max_speed = 1.8
        self.steer_force = 0.1
        self.trail_timer = 0

    def update(self, herding_point, herbivores, plain_center, plain_radius):
        # --- Boids-like behavior ---
        # 1. Cohesion: Steer towards the average position of the herd
        # 2. Separation: Steer away from neighbors to avoid crowding
        # 3. Alignment: Steer towards the average heading of the herd (simplified here)
        # 4. Seek: Steer towards the player's herding point

        avg_pos = pygame.Vector2(0, 0)
        avg_vel = pygame.Vector2(0, 0)
        separation_vec = pygame.Vector2(0, 0)
        neighbor_count = 0
        for other in herbivores:
            if other is not self:
                dist = self.pos.distance_to(other.pos)
                if 0 < dist < self.size * 4:
                    avg_pos += other.pos
                    avg_vel += other.vel
                    separation_vec += (self.pos - other.pos) / dist
                    neighbor_count += 1
        
        # Combine forces
        seek_vec = (herding_point - self.pos)
        if seek_vec.length() > 0:
            seek_vec.scale_to_length(self.max_speed)
        
        steering = seek_vec - self.vel
        
        if neighbor_count > 0:
            avg_pos /= neighbor_count
            cohesion_vec = (avg_pos - self.pos)
            if cohesion_vec.length() > 0:
                cohesion_vec.scale_to_length(self.max_speed)
            steering += (cohesion_vec - self.vel) * 0.5
            steering += separation_vec * 1.5

        if steering.length() > self.steer_force:
            steering.scale_to_length(self.steer_force)

        self.vel += steering
        if self.vel.length() > self.max_speed:
            self.vel.scale_to_length(self.max_speed)

        self.pos += self.vel
        
        # Keep within the shrinking plain
        to_center = self.pos - plain_center
        if to_center.length() > plain_radius - self.size:
            to_center.scale_to_length(plain_radius - self.size)
            self.pos = plain_center + to_center
            self.vel *= 0.8 # Dampen velocity at the edge

    def draw(self, surface, particles):
        # Main body
        if self.vel.length() > 0.1:
            angle = self.vel.angle_to(pygame.Vector2(1, 0))
            p1 = self.pos + pygame.Vector2(self.size, 0).rotate(-angle)
            p2 = self.pos + pygame.Vector2(-self.size / 2, self.size / 2).rotate(-angle)
            p3 = self.pos + pygame.Vector2(-self.size / 2, -self.size / 2).rotate(-angle)
            points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
            pygame.gfxdraw.filled_polygon(surface, points, self.color)
            pygame.gfxdraw.aapolygon(surface, points, self.color)
        else: # Draw as a circle if still
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.size/1.5), self.color)

        # Particle trail for movement feedback
        self.trail_timer -= 1
        if self.vel.length() > 1.0 and self.trail_timer <= 0:
            particles.append(Particle(self.pos, -self.vel * 0.1, 2, (180, 160, 140), 20))
            self.trail_timer = 5


class Predator:
    """Represents a predator that hunts the herd."""
    def __init__(self, pos, color):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()
        self.color = color
        self.size = 12
        self.max_speed = 1.2
        self.steer_force = 0.08
        self.state = 'PATROL' # PATROL or CHASE
        self.target_herbivore = None
        self.patrol_target = pygame.Vector2(0, 0)
        self.detection_radius = 150
        self.attack_radius = 10

    def update(self, herbivores, plain_center, plain_radius):
        # State transitions
        if self.state == 'PATROL':
            closest_dist = float('inf')
            closest_herbivore = None
            for herbivore in herbivores:
                dist = self.pos.distance_to(herbivore.pos)
                if dist < self.detection_radius and dist < closest_dist:
                    closest_dist = dist
                    closest_herbivore = herbivore
            if closest_herbivore:
                self.state = 'CHASE'
                self.target_herbivore = closest_herbivore
                self.max_speed = 1.9 # Speed up when chasing
                # sfx: predator_aggro
        
        elif self.state == 'CHASE':
            if not self.target_herbivore or self.target_herbivore not in herbivores or self.pos.distance_to(self.target_herbivore.pos) > self.detection_radius * 1.2:
                self.state = 'PATROL'
                self.target_herbivore = None
                self.max_speed = 1.2
                self.set_new_patrol_target(plain_center, plain_radius)

        # Movement logic based on state
        steering = pygame.Vector2(0, 0)
        if self.state == 'PATROL':
            if self.pos.distance_to(self.patrol_target) < 20 or self.patrol_target == pygame.Vector2(0,0):
                self.set_new_patrol_target(plain_center, plain_radius)
            
            seek_vec = (self.patrol_target - self.pos)
            if seek_vec.length() > 0:
                seek_vec.scale_to_length(self.max_speed)
            steering = seek_vec - self.vel
        
        elif self.state == 'CHASE' and self.target_herbivore:
            seek_vec = (self.target_herbivore.pos - self.pos)
            if seek_vec.length() > 0:
                seek_vec.scale_to_length(self.max_speed)
            steering = seek_vec - self.vel

        if steering.length() > self.steer_force:
            steering.scale_to_length(self.steer_force)
        
        self.vel += steering
        if self.vel.length() > self.max_speed:
            self.vel.scale_to_length(self.max_speed)
        
        self.pos += self.vel

        # Keep within the shrinking plain
        to_center = self.pos - plain_center
        if to_center.length() > plain_radius - self.size:
            to_center.scale_to_length(plain_radius - self.size)
            self.pos = plain_center + to_center

    def set_new_patrol_target(self, center, radius):
        angle = random.uniform(0, 2 * math.pi)
        self.patrol_target = center + pygame.Vector2(math.cos(angle), math.sin(angle)) * (radius * 0.9)

    def draw(self, surface):
        # Glow effect
        glow_color = (*self.color, 50)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.size * 1.5), glow_color)

        # Main body
        if self.vel.length() > 0.1:
            angle = self.vel.angle_to(pygame.Vector2(1, 0))
            p1 = self.pos + pygame.Vector2(self.size, 0).rotate(-angle)
            p2 = self.pos + pygame.Vector2(-self.size, self.size / 2).rotate(-angle)
            p3 = self.pos + pygame.Vector2(-self.size, -self.size / 2).rotate(-angle)
            points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
            pygame.gfxdraw.filled_polygon(surface, points, self.color)
            pygame.gfxdraw.aapolygon(surface, points, self.color)
        else: # Draw as circle if still
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.size), self.color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide a herd of herbivores to the safety of the valley while protecting them from predators. "
        "The plains shrink over time, so you must move quickly!"
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your herding point and guide the herd. "
        "Press space to trigger a protective shockwave that repels predators."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.INITIAL_HERD_SIZE = 30
        self.INITIAL_PREDATOR_COUNT = 2
        self.MIN_HERD_FOR_VICTORY = 10
        self.MAX_STEPS = 2000
        self.VALLEY_X_START = self.WIDTH - 40
        
        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_PLAIN = (80, 60, 40)
        self.COLOR_VALLEY = (60, 120, 180)
        self.COLOR_HERBIVORE = (50, 220, 50)
        self.COLOR_PREDATOR = (220, 50, 50)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_REACTION = (255, 200, 0)
        
        # Rewards
        self.REWARD_SURVIVAL_STEP = 0.01 # Changed from 0.1 to keep rewards in specified range
        self.REWARD_PREDATOR_KILL = 5.0
        self.REWARD_WIN = 100.0
        self.REWARD_LOSE = -100.0

        # Gameplay
        self.HERDING_POINT_SPEED = 5.0
        self.CHAIN_REACTION_RADIUS = 80
        self.CHAIN_REACTION_COOLDOWN_MAX = 90 # 3 seconds at 30fps
        self.CHAIN_REACTION_LIFETIME = 15

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.center = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        self.plain_radius = 0
        self.plain_shrink_rate = 0
        self.herding_point = pygame.Vector2(0, 0)
        self.herbivores = []
        self.predators = []
        self.particles = []
        self.active_chain_reaction = None
        self.chain_reaction_cooldown = 0
        self.prev_space_held = False
        self.predator_spawn_timer = 0
        self.max_predators = self.INITIAL_PREDATOR_COUNT
        self.bg_mountains = []
        
        # Initialize state by calling reset
        # self.reset() # Removed to avoid double-init, called by wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.plain_radius = self.HEIGHT * 0.9
        self.plain_shrink_rate = 0.05
        
        self.herding_point = self.center - pygame.Vector2(100, 0)
        
        self.herbivores = [Herbivore(self.herding_point + pygame.Vector2(random.uniform(-20, 20), random.uniform(-20, 20)), self.COLOR_HERBIVORE) for _ in range(self.INITIAL_HERD_SIZE)]
        
        self.predators = []
        for _ in range(self.INITIAL_PREDATOR_COUNT):
            angle = random.uniform(0, 2 * math.pi)
            pos = self.center + pygame.Vector2(math.cos(angle), math.sin(angle)) * self.plain_radius * 0.8
            self.predators.append(Predator(pos, self.COLOR_PREDATOR))
        
        self.particles = []
        self.active_chain_reaction = None
        self.chain_reaction_cooldown = 0
        self.prev_space_held = False
        self.predator_spawn_timer = 0
        self.max_predators = self.INITIAL_PREDATOR_COUNT

        # Generate procedural background
        self.bg_mountains = []
        for i in range(20):
            base_y = self.HEIGHT
            x = random.randint(-50, self.WIDTH + 50)
            h = random.randint(50, 150)
            w = random.randint(100, 300)
            color_val = random.randint(25, 45)
            color = (color_val, color_val + 5, color_val + 15)
            self.bg_mountains.append([(x - w//2, base_y), (x, base_y - h), (x + w//2, base_y), color])
        self.bg_mountains.sort(key=lambda m: m[1][1]) # Sort by peak height for correct layering

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = self.REWARD_SURVIVAL_STEP

        # --- Handle Input ---
        self._handle_input(action)
        
        # --- Update Game State ---
        self._update_herding_point(action[0])
        self._update_entities()
        self._update_plain_and_spawns()
        self._update_effects()

        # --- Handle Interactions & Rewards ---
        interaction_reward = self._handle_interactions()
        reward += interaction_reward

        # --- Check Termination ---
        terminated, win = self._check_termination()
        self.game_over = terminated
        
        if terminated:
            if win:
                # sfx: game_win
                win_bonus = self.REWARD_WIN * (len(self.herbivores) / self.INITIAL_HERD_SIZE)
                reward += win_bonus
                self.score += win_bonus
            else:
                # sfx: game_lose
                reward += self.REWARD_LOSE
                self.score += self.REWARD_LOSE
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        space_pressed = (action[1] == 1 and not self.prev_space_held)
        if space_pressed and self.chain_reaction_cooldown <= 0 and len(self.herbivores) > 0:
            # sfx: chain_reaction_trigger
            herd_center = sum((h.pos for h in self.herbivores), pygame.Vector2()) / len(self.herbivores)
            self.active_chain_reaction = {
                "pos": herd_center,
                "lifetime": self.CHAIN_REACTION_LIFETIME,
                "max_lifetime": self.CHAIN_REACTION_LIFETIME,
                "radius": self.CHAIN_REACTION_RADIUS,
            }
            self.chain_reaction_cooldown = self.CHAIN_REACTION_COOLDOWN_MAX
        self.prev_space_held = (action[1] == 1)

    def _update_herding_point(self, movement):
        if movement == 1: # Up
            self.herding_point.y -= self.HERDING_POINT_SPEED
        elif movement == 2: # Down
            self.herding_point.y += self.HERDING_POINT_SPEED
        elif movement == 3: # Left
            self.herding_point.x -= self.HERDING_POINT_SPEED
        elif movement == 4: # Right
            self.herding_point.x += self.HERDING_POINT_SPEED
        
        self.herding_point.x = np.clip(self.herding_point.x, 0, self.WIDTH)
        self.herding_point.y = np.clip(self.herding_point.y, 0, self.HEIGHT)

    def _update_entities(self):
        for h in self.herbivores:
            h.update(self.herding_point, self.herbivores, self.center, self.plain_radius)
        for p in self.predators:
            p.update(self.herbivores, self.center, self.plain_radius)

    def _update_plain_and_spawns(self):
        # Shrink plain
        self.plain_radius -= self.plain_shrink_rate
        if self.steps > 0 and self.steps % 200 == 0:
            self.plain_shrink_rate += 0.05
        
        # Spawn predators
        self.predator_spawn_timer += 1
        if self.steps > 0 and self.steps % 500 == 0:
            self.max_predators += 1

        if len(self.predators) < self.max_predators and self.predator_spawn_timer > 150:
            angle = random.uniform(0, 2 * math.pi)
            pos = self.center + pygame.Vector2(math.cos(angle), math.sin(angle)) * self.plain_radius * 0.95
            if pos.x < self.VALLEY_X_START: # Don't spawn in valley
                self.predators.append(Predator(pos, self.COLOR_PREDATOR))
                self.predator_spawn_timer = 0
                # sfx: predator_spawn
    
    def _update_effects(self):
        # Update particles
        self.particles = [p for p in self.particles if p.lifetime > 0 and p.radius > 0]
        for p in self.particles:
            p.update()
        
        # Update chain reaction
        if self.active_chain_reaction:
            self.active_chain_reaction["lifetime"] -= 1
            if self.active_chain_reaction["lifetime"] <= 0:
                self.active_chain_reaction = None
        
        if self.chain_reaction_cooldown > 0:
            self.chain_reaction_cooldown -= 1

    def _handle_interactions(self):
        reward = 0
        
        # Predator vs Herbivore
        for predator in self.predators:
            for herbivore in self.herbivores[:]:
                if predator.pos.distance_to(herbivore.pos) < predator.attack_radius:
                    # sfx: herbivore_death
                    for _ in range(15):
                        self.particles.append(Particle(herbivore.pos, pygame.Vector2(random.uniform(-2,2), random.uniform(-2,2)), 4, self.COLOR_HERBIVORE, 40))
                    self.herbivores.remove(herbivore)
                    predator.state = 'PATROL' # Reset after a kill
                    break

        # Chain reaction vs Predator
        if self.active_chain_reaction:
            reaction_pos = self.active_chain_reaction["pos"]
            reaction_radius = self.active_chain_reaction["radius"] * (1 - self.active_chain_reaction["lifetime"] / self.active_chain_reaction["max_lifetime"])
            for predator in self.predators[:]:
                if predator.pos.distance_to(reaction_pos) < reaction_radius:
                    # sfx: predator_death
                    reward += self.REWARD_PREDATOR_KILL
                    for _ in range(25):
                        self.particles.append(Particle(predator.pos, pygame.Vector2(random.uniform(-3,3), random.uniform(-3,3)), 5, self.COLOR_PREDATOR, 50))
                    self.predators.remove(predator)
        
        return reward

    def _check_termination(self):
        win = False
        num_in_valley = sum(1 for h in self.herbivores if h.pos.x > self.VALLEY_X_START)
        if num_in_valley >= self.MIN_HERD_FOR_VICTORY:
            win = True
            return True, win

        if not self.herbivores:
            return True, win
        if self.plain_radius <= 10:
            return True, win
        if self.steps >= self.MAX_STEPS:
            return True, win
            
        return False, win

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background mountains
        for points_and_color in self.bg_mountains:
            points, color = points_and_color[:-1], points_and_color[-1]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Valley (destination)
        valley_rect = (self.VALLEY_X_START, 0, self.WIDTH - self.VALLEY_X_START, self.HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_VALLEY, valley_rect)

        # Shrinking plain
        pygame.gfxdraw.filled_circle(self.screen, int(self.center.x), int(self.center.y), int(self.plain_radius), self.COLOR_PLAIN)
        pygame.gfxdraw.aacircle(self.screen, int(self.center.x), int(self.center.y), int(self.plain_radius), (0,0,0,50))

        # Particles
        for p in self.particles:
            p.draw(self.screen)

        # Herding point cursor
        pygame.gfxdraw.aacircle(self.screen, int(self.herding_point.x), int(self.herding_point.y), 10, (*self.COLOR_CURSOR, 150))
        pygame.gfxdraw.filled_circle(self.screen, int(self.herding_point.x), int(self.herding_point.y), 3, (*self.COLOR_CURSOR, 200))
        
        # Herbivores and Predators
        for h in self.herbivores:
            h.draw(self.screen, self.particles)
        for p in self.predators:
            p.draw(self.screen)

        # Chain reaction effect
        if self.active_chain_reaction:
            r = self.active_chain_reaction
            progress = (r["max_lifetime"] - r["lifetime"]) / r["max_lifetime"]
            current_radius = int(r["radius"] * progress)
            alpha = int(255 * (1 - progress**2))
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(r["pos"].x), int(r["pos"].y), current_radius, (*self.COLOR_REACTION, alpha))
                pygame.gfxdraw.aacircle(self.screen, int(r["pos"].x), int(r["pos"].y), current_radius, (*self.COLOR_REACTION, alpha+50 if alpha<200 else 255))

    def _render_ui(self):
        herd_text = self.font_main.render(f"Herd: {len(self.herbivores)}", True, self.COLOR_TEXT)
        self.screen.blit(herd_text, (10, 10))

        steps_text = self.font_small.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 35))

        # Cooldown indicator for chain reaction
        if self.chain_reaction_cooldown > 0:
            cooldown_percent = self.chain_reaction_cooldown / self.CHAIN_REACTION_COOLDOWN_MAX
            bar_width = 100
            fill_width = int(bar_width * cooldown_percent)
            pygame.draw.rect(self.screen, (100,100,100), (self.WIDTH - bar_width - 10, 10, bar_width, 15))
            pygame.draw.rect(self.screen, self.COLOR_REACTION, (self.WIDTH - bar_width - 10, 10, fill_width, 15))
            pygame.draw.rect(self.screen, self.COLOR_TEXT, (self.WIDTH - bar_width - 10, 10, bar_width, 15), 1)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "herd_size": len(self.herbivores),
            "predators": len(self.predators),
            "plain_radius": self.plain_radius
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a display, so we unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Herd Migration")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Remove the validation call from the main block as it's not needed for playtesting
    # and was part of the original template.
    # env.validate_implementation() 
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Herd Survived: {info['herd_size']}, Steps: {info['steps']}")
            # Wait for 'R' to reset
            while True:
                event_wait = pygame.event.wait()
                if event_wait.type == pygame.QUIT:
                    running = False
                    break
                if event_wait.type == pygame.KEYDOWN and event_wait.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    break
        
        clock.tick(30) # Cap FPS for consistent game speed

    env.close()