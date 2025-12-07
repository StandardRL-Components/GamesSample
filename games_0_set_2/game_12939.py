import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
import os
import pygame


# Set the SDL video driver to dummy to run Pygame headlessly
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
# Pygame's gfxdraw module is optional and might not be available on all systems.
# We'll use a fallback if it's not found.
try:
    import pygame.gfxdraw
    HAS_GFXDRAW = True
except ImportError:
    HAS_GFXDRAW = False


# Helper class for the player ships
class PlayerShip:
    def __init__(self, pos, color, size=12):
        self.pos = np.array(pos, dtype=np.float32)
        self.color = color
        self.base_color = color
        self.size = size
        self.angle = -90.0  # Pointing up
        self.speed_multiplier = 2.0  # Start at high speed
        self.base_speed = 2.5
        self.velocity = np.array([0.0, 0.0])

    def set_speed_mode(self, mode):
        # mode 'high' or 'low'
        self.speed_multiplier = 2.0 if mode == 'high' else 1.0
        # Update color based on speed
        if mode == 'high':
            self.color = (100, 200, 255) # Bright Blue
        else:
            self.color = (255, 100, 100) # Bright Red

    def move(self, dx, dy, bounds):
        move_vector = np.array([dx, dy], dtype=np.float32)
        if np.linalg.norm(move_vector) > 0:
            move_vector = move_vector / np.linalg.norm(move_vector)
        
        self.velocity = move_vector * self.base_speed * self.speed_multiplier
        self.pos += self.velocity
        
        # Clamp position to stay within screen bounds
        self.pos[0] = np.clip(self.pos[0], self.size, bounds[0] - self.size)
        self.pos[1] = np.clip(self.pos[1], self.size, bounds[1] - self.size)

    def get_vertices(self):
        # Calculate vertices for an isosceles triangle
        p1 = [self.pos[0] + math.cos(math.radians(self.angle)) * self.size,
              self.pos[1] + math.sin(math.radians(self.angle)) * self.size]
        p2 = [self.pos[0] + math.cos(math.radians(self.angle + 140)) * self.size,
              self.pos[1] + math.sin(math.radians(self.angle + 140)) * self.size]
        p3 = [self.pos[0] + math.cos(math.radians(self.angle - 140)) * self.size,
              self.pos[1] + math.sin(math.radians(self.angle - 140)) * self.size]
        return [p1, p2, p3]

    def draw(self, surface):
        verts = self.get_vertices()
        int_verts = [(int(p[0]), int(p[1])) for p in verts]

        if HAS_GFXDRAW:
            # Draw glow effect
            glow_color = (*self.color, 50)
            pygame.gfxdraw.filled_polygon(surface, int_verts, glow_color)
            for i in range(1, 4):
                 pygame.gfxdraw.aapolygon(surface, [(v[0],v[1]+i) for v in int_verts], glow_color)
                 pygame.gfxdraw.aapolygon(surface, [(v[0],v[1]-i) for v in int_verts], glow_color)
                 pygame.gfxdraw.aapolygon(surface, [(v[0]+i,v[1]) for v in int_verts], glow_color)
                 pygame.gfxdraw.aapolygon(surface, [(v[0]-i,v[1]) for v in int_verts], glow_color)

            # Draw main ship body
            pygame.gfxdraw.aapolygon(surface, int_verts, self.color)
            pygame.gfxdraw.filled_polygon(surface, int_verts, self.color)
        else:
            pygame.draw.polygon(surface, self.color, int_verts)


# Helper class for asteroids
class Asteroid:
    def __init__(self, pos, vel, size, level):
        self.pos = np.array(pos, dtype=np.float32)
        self.vel = np.array(vel, dtype=np.float32)
        self.size = size
        self.level = level
        self.angle = 0
        self.rotation_speed = random.uniform(-1, 1)
        self.num_vertices = random.randint(5, 8)
        self.vertices = self._create_shape()

    def _create_shape(self):
        verts = []
        for i in range(self.num_vertices):
            angle = (2 * math.pi / self.num_vertices) * i
            dist = self.size * random.uniform(0.8, 1.2)
            verts.append([dist * math.cos(angle), dist * math.sin(angle)])
        return verts

    def move(self, bounds):
        self.pos += self.vel
        self.angle += self.rotation_speed
        # Screen wrap
        if self.pos[0] < -self.size: self.pos[0] = bounds[0] + self.size
        if self.pos[0] > bounds[0] + self.size: self.pos[0] = -self.size
        if self.pos[1] < -self.size: self.pos[1] = bounds[1] + self.size
        if self.pos[1] > bounds[1] + self.size: self.pos[1] = -self.size

    def get_rotated_vertices(self):
        rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rotated = []
        for v in self.vertices:
            x = v[0] * cos_a - v[1] * sin_a + self.pos[0]
            y = v[0] * sin_a + v[1] * cos_a + self.pos[1]
            rotated.append((int(x), int(y)))
        return rotated

    def draw(self, surface):
        verts = self.get_rotated_vertices()
        color = (160, 160, 170)
        fill_color = (100, 100, 110)
        if HAS_GFXDRAW:
            pygame.gfxdraw.aapolygon(surface, verts, color)
            pygame.gfxdraw.filled_polygon(surface, verts, fill_color)
        else:
            pygame.draw.polygon(surface, fill_color, verts)
            pygame.draw.polygon(surface, color, verts, 1)


    def split(self):
        if self.size < 15:
            return [] # Don't split if too small
        
        num_fragments = random.randint(2, 3)
        new_asteroids = []
        for _ in range(num_fragments):
            angle = random.uniform(0, 2 * math.pi)
            speed_multiplier = random.uniform(1.2, 1.8)
            new_vel = [math.cos(angle) * speed_multiplier, math.sin(angle) * speed_multiplier]
            new_size = self.size / 1.75
            # FIX: Convert new_vel to a numpy array for vector arithmetic
            new_pos = self.pos + np.array(new_vel) * 10 # Eject from center
            new_asteroids.append(Asteroid(new_pos, new_vel, new_size, self.level))
        return new_asteroids

# Helper class for particle effects
class Particle:
    def __init__(self, pos, color):
        self.pos = np.array(pos, dtype=np.float32)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
        self.life = random.randint(20, 40)
        self.radius = self.life / 6
        self.color = color

    def update(self):
        self.life -= 1
        self.pos += self.vel
        self.radius = max(0, self.life / 6)
        return self.life > 0

    def draw(self, surface):
        if self.radius > 0:
            pos = (int(self.pos[0]), int(self.pos[1]))
            radius = int(self.radius)
            if HAS_GFXDRAW:
                pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, self.color)
            else:
                pygame.draw.circle(surface, self.color, pos, radius)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Control two ships simultaneously to navigate through a dense asteroid field. "
        "Guide both ships to the exit zone to advance to the next level."
    )
    user_guide = (
        "Ship 1 (Blue): Use Arrow Keys to move. Ship 2 (Orange): Use WASD keys to move. "
        "Both ships must reach the green exit zone."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.screen_width = 640
        self.screen_height = 400

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24)
        
        # Game state persistent across resets
        self.level = 1
        
        # Game state reset each episode
        self.ship1 = None
        self.ship2 = None
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.exit_zone = None
        self.steps = 0
        self.score = 0.0
        self.timer = 0
        self.speed_toggle_timer = 0
        self.game_over = False
        
        # Constants
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_EXIT = (0, 255, 100, 60)
        self.COLOR_UI = (220, 220, 240)
        self.MAX_STEPS = 3600 # 60 seconds at 60fps
        self.SPEED_TOGGLE_INTERVAL = 300 # 5 seconds at 60fps

        self._create_starfield()
        # No need to call reset here, it will be called by the runner
        
    def _create_starfield(self):
        self.stars = []
        for _ in range(150):
            x = random.randint(0, self.screen_width)
            y = random.randint(0, self.screen_height)
            brightness = random.randint(50, 150)
            self.stars.append(((x, y), (brightness, brightness, brightness)))

    def _spawn_asteroids(self):
        self.asteroids = []
        num_asteroids = 2 + self.level
        
        for _ in range(num_asteroids):
            while True:
                pos = [random.uniform(0, self.screen_width), random.uniform(0, self.screen_height)]
                # Avoid spawning on ships or exit
                if (self.ship1 and self.ship2 and
                    math.dist(pos, self.ship1.pos) > 100 and
                    math.dist(pos, self.ship2.pos) > 100 and
                    not self.exit_zone.collidepoint(pos)):
                    break
            
            angle = random.uniform(0, 2 * math.pi)
            base_speed = 0.5
            if self.level >= 3:
                base_speed = random.uniform(1.0, 1.5)
            
            vel = [math.cos(angle) * base_speed, math.sin(angle) * base_speed]
            
            size = random.uniform(25, 40)
            if self.level == 2:
                size = random.uniform(40, 55)
            
            self.asteroids.append(Asteroid(pos, vel, size, self.level))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = self.MAX_STEPS
        self.speed_toggle_timer = self.SPEED_TOGGLE_INTERVAL
        
        self.ship1 = PlayerShip(pos=[100, 200], color=(0, 150, 255))
        self.ship2 = PlayerShip(pos=[self.screen_width - 100, 200], color=(255, 150, 0))
        self.ship1.set_speed_mode('high')
        self.ship2.set_speed_mode('high')
        
        self.exit_zone = pygame.Rect(self.screen_width - 60, self.screen_height/2 - 50, 50, 100)
        
        self.particles = []
        self._spawn_asteroids()
        
        return self._get_observation(), self._get_info()
    
    def _handle_input(self, action):
        # Unpack factorized action
        # action[0]: Ship 1 Movement (0=none, 1=up, 2=down, 3=left, 4=right)
        # action[1], action[2]: Ship 2 Movement
        # [0,0]=Up, [0,1]=Down, [1,0]=Left, [1,1]=Right
        
        # Ship 1
        movement1 = action[0]
        dx1, dy1 = 0, 0
        if movement1 == 1: dy1 = -1
        elif movement1 == 2: dy1 = 1
        elif movement1 == 3: dx1 = -1
        elif movement1 == 4: dx1 = 1
        if self.ship1: self.ship1.move(dx1, dy1, (self.screen_width, self.screen_height))
        
        # Ship 2
        space_held, shift_held = action[1], action[2]
        dx2, dy2 = 0, 0
        if space_held == 0 and shift_held == 0: dy2 = -1 # Up
        elif space_held == 0 and shift_held == 1: dy2 = 1  # Down
        elif space_held == 1 and shift_held == 0: dx2 = -1 # Left
        elif space_held == 1 and shift_held == 1: dx2 = 1  # Right
        if self.ship2: self.ship2.move(dx2, dy2, (self.screen_width, self.screen_height))

    def _create_collision_particles(self, pos, color):
        for _ in range(30):
            self.particles.append(Particle(pos, color))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.01  # Survival reward
        self.steps += 1
        self.timer -= 1
        self.speed_toggle_timer -= 1

        # Handle player input
        self._handle_input(action)

        # Update ship speed modes
        if self.speed_toggle_timer <= 0:
            self.speed_toggle_timer = self.SPEED_TOGGLE_INTERVAL
            if self.ship1 and self.ship2:
                current_mode = 'high' if self.ship1.speed_multiplier == 1.0 else 'low'
                self.ship1.set_speed_mode(current_mode)
                self.ship2.set_speed_mode(current_mode)

        # Update asteroids
        asteroids_to_add = []
        for asteroid in self.asteroids:
            asteroid.move((self.screen_width, self.screen_height))

        # Update particles
        self.particles = [p for p in self.particles if p.update()]

        # Check for collisions
        terminated = False
        ships = [self.ship1, self.ship2]
        for i, ship in enumerate(ships):
            if not ship: continue
            for asteroid in self.asteroids:
                if math.dist(ship.pos, asteroid.pos) < ship.size + asteroid.size:
                    self.game_over = True
                    terminated = True
                    reward = -100.0
                    self.score += reward
                    self._create_collision_particles(ship.pos, ship.color)
                    # Remove collided ship for rendering
                    if i == 0: self.ship1 = None
                    else: self.ship2 = None
                    
                    # Split the asteroid
                    asteroids_to_add.extend(asteroid.split())
                    self.asteroids.remove(asteroid)
                    break
            if terminated: break
        
        self.asteroids.extend(asteroids_to_add)

        # Check for win condition
        if not terminated:
            ship1_in_exit = self.ship1 and self.exit_zone.collidepoint(self.ship1.pos)
            ship2_in_exit = self.ship2 and self.exit_zone.collidepoint(self.ship2.pos)
            if ship1_in_exit and ship2_in_exit:
                self.game_over = True
                terminated = True
                reward = 100.0
                self.score += reward
                self.level += 1 # Progress to next level

        # Check for timeout
        truncated = self.timer <= 0 and not terminated
        if truncated:
            self.game_over = True
            reward = -100.0
            self.score += reward

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _render_game(self):
        # Draw starfield
        for pos, color in self.stars:
            self.screen.set_at(pos, color)
            
        # Draw exit zone
        exit_surface = pygame.Surface((self.exit_zone.width, self.exit_zone.height), pygame.SRCALPHA)
        exit_surface.fill(self.COLOR_EXIT)
        self.screen.blit(exit_surface, self.exit_zone.topleft)

        # Draw asteroids
        for asteroid in self.asteroids:
            asteroid.draw(self.screen)
        
        # Draw ships
        if self.ship1: self.ship1.draw(self.screen)
        if self.ship2: self.ship2.draw(self.screen)
            
        # Draw particles
        for particle in self.particles:
            particle.draw(self.screen)

    def _render_ui(self):
        # Timer
        time_text = f"TIME: {self.timer // 60:02d}"
        time_surface = self.font.render(time_text, True, self.COLOR_UI)
        self.screen.blit(time_surface, (10, 10))
        
        # Level
        level_text = f"LEVEL: {self.level}"
        level_surface = self.font.render(level_text, True, self.COLOR_UI)
        self.screen.blit(level_surface, (self.screen_width - level_surface.get_width() - 10, 10))
        
        # Speed mode indicator
        if self.ship1:
            speed_mode = "HIGH" if self.ship1.speed_multiplier == 2.0 else "LOW"
            speed_color = (100, 200, 255) if speed_mode == "HIGH" else (255, 100, 100)
            speed_text = f"SPEED: {speed_mode}"
            speed_surface = self.font.render(speed_text, True, speed_color)
            self.screen.blit(speed_surface, (self.screen_width // 2 - speed_surface.get_width() // 2, 10))

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
            "level": self.level,
            "timer": self.timer,
        }

    def close(self):
        pygame.quit()


# Example of how to run the environment for manual play
if __name__ == '__main__':
    # Unset the dummy driver to allow for display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.set_caption("Dual Ship Navigator")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    obs, info = env.reset()
    
    key_map_ship1 = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    running = True
    while running:
        action = [0, 0, 0] # [ship1_move, ship2_flag1, ship2_flag2]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Ship 1 controls (Arrow Keys)
        action[0] = 0
        for key_code, move_action in key_map_ship1.items():
            if keys[key_code]:
                action[0] = move_action
                break
        
        # Ship 2 controls (WASD) mapped to the two binary flags
        # [0,0]=W(Up), [0,1]=S(Down), [1,0]=A(Left), [1,1]=D(Right)
        if keys[pygame.K_w]:
            action[1], action[2] = 0, 0
        elif keys[pygame.K_s]:
            action[1], action[2] = 0, 1
        elif keys[pygame.K_a]:
            action[1], action[2] = 1, 0
        elif keys[pygame.K_d]:
            action[1], action[2] = 1, 1
        else: # Default no-op for ship 2 requires a specific combination
            # This action space design doesn't have a natural no-op for ship 2
            # We can leave the last state or pick one, e.g., 'up'
            action[1], action[2] = 0, 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']:.2f}, Level Reached: {info['level']}")
            obs, info = env.reset()

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(60)
        
    env.close()