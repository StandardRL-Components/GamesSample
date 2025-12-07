import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game entities
class Player:
    def __init__(self, pos):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)
        self.angle = -90
        self.radius = 12
        self.base_speed = 2.5
        self.boost_speed = 4.5
        self.turn_speed = 4.5
        self.drag = 0.95
        self.invulnerable_timer = 0

    def update(self, movement, shift_held, bounds):
        is_moving = False
        # Turning
        if movement == 3:  # Left
            self.angle -= self.turn_speed
        if movement == 4:  # Right
            self.angle += self.turn_speed

        # Acceleration
        speed = self.boost_speed if shift_held else self.base_speed
        acceleration = pygame.Vector2(0, 0)
        if movement == 1:  # Up
            acceleration = pygame.Vector2(speed, 0).rotate(-self.angle)
            is_moving = True
        elif movement == 2:  # Down/Brake
            # Apply braking force opposite to velocity
            if self.vel.length() > 0.1:
                self.vel *= 0.90
            else:
                self.vel = pygame.Vector2(0, 0)
            is_moving = True
        
        # Drifting mechanics for turning while accelerating
        if shift_held and (movement == 3 or movement == 4):
             self.drag = 0.98 # Less drag when drifting
        else:
             self.drag = 0.95

        self.vel += acceleration * 0.1
        self.vel *= self.drag
        
        # Limit max speed
        max_speed = speed
        if self.vel.length() > max_speed:
            self.vel.scale_to_length(max_speed)

        self.pos += self.vel

        # Boundary checks
        self.pos.x = max(self.radius, min(self.pos.x, bounds[0] - self.radius))
        self.pos.y = max(self.radius, min(self.pos.y, bounds[1] - self.radius))
        
        if self.invulnerable_timer > 0:
            self.invulnerable_timer -= 1
            
        return is_moving, shift_held

    def draw(self, surface):
        # Flashing effect when invulnerable
        if self.invulnerable_timer > 0 and (self.invulnerable_timer // 3) % 2 == 0:
            return

        # Ship body
        color = (60, 180, 255) # Bright Blue
        
        # Create a triangular ship shape
        p1 = self.pos + pygame.Vector2(self.radius, 0).rotate(-self.angle)
        p2 = self.pos + pygame.Vector2(-self.radius, -self.radius * 0.7).rotate(-self.angle)
        p3 = self.pos + pygame.Vector2(-self.radius, self.radius * 0.7).rotate(-self.angle)
        points = [p1, p2, p3]
        
        pygame.gfxdraw.aapolygon(surface, [(int(p.x), int(p.y)) for p in points], color)
        pygame.gfxdraw.filled_polygon(surface, [(int(p.x), int(p.y)) for p in points], color)

        # Cockpit
        cockpit_pos = self.pos + pygame.Vector2(self.radius * 0.4, 0).rotate(-self.angle)
        pygame.gfxdraw.aacircle(surface, int(cockpit_pos.x), int(cockpit_pos.y), 3, (200, 255, 255))
        pygame.gfxdraw.filled_circle(surface, int(cockpit_pos.x), int(cockpit_pos.y), 3, (200, 255, 255))

class Asteroid:
    def __init__(self, pos, vel, size, ore, seed_rng):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.radius = size
        self.ore_amount = ore
        self.rotation = seed_rng.uniform(0, 360)
        self.rotation_speed = seed_rng.uniform(-1.5, 1.5)
        self.color = random.choice([(139, 125, 123), (160, 82, 45), (112, 128, 144)])
        
        # Generate a procedural shape
        num_points = seed_rng.integers(7, 12)
        self.shape_points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            dist = seed_rng.uniform(0.7, 1.0) * self.radius
            self.shape_points.append(pygame.Vector2(math.cos(angle) * dist, math.sin(angle) * dist))

    def update(self, bounds):
        self.pos += self.vel
        self.rotation += self.rotation_speed
        
        # Screen wrap
        if self.pos.x < -self.radius: self.pos.x = bounds[0] + self.radius
        if self.pos.x > bounds[0] + self.radius: self.pos.x = -self.radius
        if self.pos.y < -self.radius: self.pos.y = bounds[1] + self.radius
        if self.pos.y > bounds[1] + self.radius: self.pos.y = -self.radius

    def draw(self, surface):
        points = []
        for p in self.shape_points:
            rotated_p = p.rotate(self.rotation)
            points.append((int(self.pos.x + rotated_p.x), int(self.pos.y + rotated_p.y)))
        
        if len(points) > 2:
            pygame.gfxdraw.aapolygon(surface, points, self.color)
            pygame.gfxdraw.filled_polygon(surface, points, self.color)

class OreChunk:
    def __init__(self, pos):
        self.pos = pygame.Vector2(pos)
        self.radius = 5
        self.rotation = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-2, 2)
        self.value = 1
        self.spawn_time = pygame.time.get_ticks()

    def update(self):
        self.rotation += self.rotation_speed

    def draw(self, surface):
        color = (255, 215, 0) # Gold
        points = []
        for i in range(5):
            angle = self.rotation + i * (360 / 5)
            p = self.pos + pygame.Vector2(self.radius, 0).rotate(angle)
            points.append((int(p.x), int(p.y)))
        
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

class Particle:
    def __init__(self, pos, vel, life, color, radius_range):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.life = life
        self.max_life = life
        self.color = color
        self.radius_start, self.radius_end = radius_range
        
    def update(self):
        self.pos += self.vel
        self.life -= 1
        return self.life > 0

    def draw(self, surface):
        life_ratio = self.life / self.max_life
        current_radius = int(self.radius_start * life_ratio + self.radius_end * (1 - life_ratio))
        if current_radius > 0:
            # Fade out color
            alpha_color = (*self.color, int(255 * life_ratio))
            temp_surf = pygame.Surface((current_radius*2, current_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, alpha_color, (current_radius, current_radius), current_radius)
            surface.blit(temp_surf, (int(self.pos.x - current_radius), int(self.pos.y - current_radius)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: ↑↓←→ to move. Hold space to mine asteroids. Hold shift to boost."
    game_description = "Pilot a space miner through asteroid fields, collecting ore while dodging hazardous space rocks to reach a target yield."
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)
        
        # Game constants
        self.COLOR_BG = (10, 15, 30)
        self.WIN_ORE = 100
        self.MAX_LIVES = 3
        self.MAX_STEPS = 5000
        self.MIN_ASTEROIDS = 8
        self.MAX_ASTEROIDS = 12
        self.MINING_RANGE = 120
        self.MINING_COOLDOWN = 5 # frames
        
        # State variables are initialized in reset()
        self.player = None
        self.asteroids = []
        self.ore_chunks = []
        self.particles = []
        self.stars = []
        
        self.steps = 0
        self.score = 0
        self.ore_collected = 0
        self.lives = 0
        self.game_over = False
        self.mining_timer = 0
        self.last_difficulty_milestone = 0
        
        # self.validate_implementation() # Commented out for submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.ore_collected = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.last_difficulty_milestone = 0
        self.base_asteroid_speed = 0.5
        
        self.player = Player((self.width // 2, self.height // 2))
        
        self.asteroids = []
        self.ore_chunks = []
        self.particles = []
        
        # Create a static starfield
        if not self.stars:
            for _ in range(150):
                self.stars.append({
                    "pos": pygame.Vector2(self.np_random.uniform(0, self.width), self.np_random.uniform(0, self.height)),
                    "depth": self.np_random.uniform(0.2, 1.0),
                    "brightness": self.np_random.integers(50, 150)
                })

        for _ in range(self.MAX_ASTEROIDS):
            self._spawn_asteroid()
            
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = -0.01  # Time penalty
        
        # --- UPDATE GAME LOGIC ---
        is_moving, is_boosting = self.player.update(movement, shift_held, (self.width, self.height))
        
        # Engine particles
        if is_moving:
            self._create_engine_particles(is_boosting)

        # Update entities
        for asteroid in self.asteroids:
            asteroid.update((self.width, self.height))
        for ore in self.ore_chunks:
            ore.update()
        self.particles = [p for p in self.particles if p.update()]
        
        # Mining logic
        if self.mining_timer > 0: self.mining_timer -= 1
        
        target_asteroid = None
        if space_held and self.mining_timer == 0:
            target_asteroid = self._get_closest_asteroid()
            if target_asteroid:
                self.mining_timer = self.MINING_COOLDOWN
                # SFX: Mining laser hum
                self._create_mining_particles(target_asteroid)
                target_asteroid.ore_amount -= 1
                if target_asteroid.ore_amount <= 0:
                    self._create_explosion(target_asteroid.pos, 30, (100,100,100))
                    self.asteroids.remove(target_asteroid)
                elif self.np_random.random() < 0.15: # Chance to spawn ore
                    self.ore_chunks.append(OreChunk(target_asteroid.pos + pygame.Vector2(self.np_random.uniform(-20, 20), self.np_random.uniform(-20, 20))))
        
        # Collision detection
        # Player vs Asteroid
        if self.player.invulnerable_timer == 0:
            for asteroid in self.asteroids:
                if self.player.pos.distance_to(asteroid.pos) < self.player.radius + asteroid.radius * 0.8:
                    self.lives -= 1
                    reward -= 10
                    self.player.invulnerable_timer = 90 # 3 seconds at 30fps
                    self._create_explosion(self.player.pos, 50, (255, 100, 0))
                    # SFX: Player hit/explosion
                    if self.lives <= 0:
                        self.game_over = True
                    break
        
        # Player vs Ore
        collected_ore = []
        for ore in self.ore_chunks:
            if self.player.pos.distance_to(ore.pos) < self.player.radius + ore.radius:
                collected_ore.append(ore)
                self.ore_collected += ore.value
                self.score += ore.value
                reward += 1.1 # +1 for event, +0.1 continuous
                self._create_collection_effect(ore.pos)
                # SFX: Ore collect
        self.ore_chunks = [o for o in self.ore_chunks if o not in collected_ore]

        # --- HOUSEKEEPING ---
        self.steps += 1
        self._maintain_entity_counts()
        self._update_difficulty()
        
        # --- CHECK TERMINATION ---
        terminated = self.game_over or self.ore_collected >= self.WIN_ORE or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            if self.ore_collected >= self.WIN_ORE:
                reward += 100 # Victory bonus
            self.game_over = True

        return (
            self._get_observation(target_asteroid),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self, mining_target=None):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        
        # Sort entities by Y-position for pseudo-3D effect
        renderables = self.asteroids + self.ore_chunks
        if not (self.player.invulnerable_timer > 0 and (self.player.invulnerable_timer // 3) % 2 == 0):
            renderables.append(self.player)
        renderables.sort(key=lambda e: e.pos.y)

        for entity in renderables:
            entity.draw(self.screen)
        
        # Render mining laser on top
        if mining_target:
            pygame.draw.line(self.screen, (255, 255, 100), self.player.pos, mining_target.pos, 2)
            # SFX: Mining beam active
            
        for particle in self.particles:
            particle.draw(self.screen)
            
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ore_collected": self.ore_collected,
            "lives": self.lives,
        }

    # --- Helper methods for game logic ---
    def _spawn_asteroid(self):
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = (self.np_random.uniform(0, self.width), -30)
        elif edge == 1: # Bottom
            pos = (self.np_random.uniform(0, self.width), self.height + 30)
        elif edge == 2: # Left
            pos = (-30, self.np_random.uniform(0, self.height))
        else: # Right
            pos = (self.width + 30, self.np_random.uniform(0, self.height))

        angle = self.np_random.uniform(0, 360)
        speed = self.np_random.uniform(0.5, 1.5) * self.base_asteroid_speed
        vel = pygame.Vector2(speed, 0).rotate(angle)
        size = self.np_random.integers(15, 35)
        ore = self.np_random.integers(5, 15)
        self.asteroids.append(Asteroid(pos, vel, size, ore, self.np_random))

    def _maintain_entity_counts(self):
        if len(self.asteroids) < self.MIN_ASTEROIDS:
            self._spawn_asteroid()

    def _update_difficulty(self):
        current_milestone = self.ore_collected // 25
        if current_milestone > self.last_difficulty_milestone:
            self.last_difficulty_milestone = current_milestone
            self.base_asteroid_speed += 0.25
            for asteroid in self.asteroids:
                asteroid.vel.scale_to_length(asteroid.vel.length() + 0.25)

    def _get_closest_asteroid(self):
        closest = None
        min_dist = self.MINING_RANGE
        for asteroid in self.asteroids:
            dist = self.player.pos.distance_to(asteroid.pos)
            if dist < min_dist:
                min_dist = dist
                closest = asteroid
        return closest
        
    # --- Helper methods for effects ---
    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(speed, 0).rotate(angle)
            life = self.np_random.integers(20, 40)
            self.particles.append(Particle(pos, vel, life, color, (5, 1)))

    def _create_mining_particles(self, asteroid):
        for _ in range(2):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(0.5, 2)
            vel = pygame.Vector2(speed, 0).rotate(angle)
            life = self.np_random.integers(10, 20)
            self.particles.append(Particle(asteroid.pos, vel, life, (255, 200, 100), (3, 0)))

    def _create_engine_particles(self, is_boosting):
        # If the player is not moving, we cannot normalize the velocity vector.
        if self.player.vel.length() == 0:
            return

        num_particles = 3 if is_boosting else 1
        # Pre-calculate normalized velocity to avoid doing it twice and for safety.
        normalized_vel = self.player.vel.normalize()
        
        for _ in range(num_particles):
            base_vel = -normalized_vel * (3 if is_boosting else 2)
            angle_offset = self.np_random.uniform(-20, 20)
            vel = base_vel.rotate(angle_offset) + self.player.vel * 0.5
            pos = self.player.pos - normalized_vel * self.player.radius
            life = 30 if is_boosting else 15
            color = (255, 150, 50) if is_boosting else (150, 150, 255)
            radius = (4, 1) if is_boosting else (3, 0)
            self.particles.append(Particle(pos, vel, life, color, radius))
            
    def _create_collection_effect(self, pos):
        for _ in range(10):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(1, 3)
            vel = pygame.Vector2(speed, 0).rotate(angle)
            life = self.np_random.integers(15, 25)
            self.particles.append(Particle(pos, vel, life, (255, 215, 0), (4, 0)))
            
    # --- Helper methods for rendering ---
    def _render_background(self):
        for star in self.stars:
            p = star["pos"]
            d = star["depth"]
            b = star["brightness"]
            # Parallax effect
            offset_x = (self.player.pos.x - self.width/2) * (d-1) * 0.1
            offset_y = (self.player.pos.y - self.height/2) * (d-1) * 0.1
            x = (p.x + offset_x) % self.width
            y = (p.y + offset_y) % self.height
            size = int(d * 1.5)
            pygame.draw.circle(self.screen, (b,b,b), (int(x), int(y)), size)
            
    def _render_ui(self):
        ore_text = self.font_ui.render(f"ORE: {self.ore_collected}/{self.WIN_ORE}", True, (255, 255, 255))
        self.screen.blit(ore_text, (10, 10))
        
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, (255, 255, 255))
        self.screen.blit(steps_text, (self.width - steps_text.get_width() - 10, 10))
        
        # Lives display
        life_icon_surf = pygame.Surface((15, 15), pygame.SRCALPHA)
        pygame.gfxdraw.aapolygon(life_icon_surf, [(7,0), (0,15), (14,15)], (60, 180, 255))
        pygame.gfxdraw.filled_polygon(life_icon_surf, [(7,0), (0,15), (14,15)], (60, 180, 255))
        
        for i in range(self.lives):
            self.screen.blit(life_icon_surf, (self.width/2 - (self.lives*20)/2 + i*20, self.height - 25))
            
    def _render_game_over(self):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))

        if self.ore_collected >= self.WIN_ORE:
            msg = "MISSION COMPLETE"
            color = (100, 255, 100)
        else:
            msg = "GAME OVER"
            color = (255, 100, 100)
        
        text = self.font_game_over.render(msg, True, color)
        text_rect = text.get_rect(center=(self.width / 2, self.height / 2))
        self.screen.blit(text, text_rect)
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up a window to display the game
    pygame.display.set_caption("Space Miner")
    screen = pygame.display.set_mode((env.width, env.height))
    clock = pygame.time.Clock()
    
    # Game loop
    while not done:
        # --- Player Input ---
        keys = pygame.key.get_pressed()
        
        # Movement
        mov = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            mov = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            mov = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            mov = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            mov = 4
            
        # Handle turning while moving forward
        if (keys[pygame.K_UP] or keys[pygame.K_w]) and (keys[pygame.K_LEFT] or keys[pygame.K_a]):
             mov = 3
        if (keys[pygame.K_UP] or keys[pygame.K_w]) and (keys[pygame.K_RIGHT] or keys[pygame.K_d]):
             mov = 4

        # Buttons
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Rendering ---
        # The observation is the rendered frame, so we just need to show it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Run at 30 FPS

    env.close()