import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:25:42.140528
# Source Brief: brief_00457.md
# Brief Index: 457
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Asteroid Survival: A retro arcade game where the player controls a ship
    and must survive an increasingly dense asteroid field for 120 seconds.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a ship and survive an increasingly dense asteroid field for 120 seconds."
    )
    user_guide = "Use the arrow keys (↑↓←→) to move your ship and dodge the asteroids."
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    GAME_DURATION_SECONDS = 120
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_SHIP = (220, 255, 255)
    COLOR_ASTEROID = (150, 160, 180)
    COLOR_PARTICLE_EXPLOSION = [(255, 60, 0), (255, 150, 0), (200, 200, 200)]
    COLOR_PARTICLE_THRUSTER = [(100, 150, 255), (180, 220, 255)]
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_UI_SHADOW = (20, 30, 60)

    # Ship
    SHIP_SPEED = 5.0
    SHIP_SIZE = 12
    SHIP_INVINCIBILITY_FRAMES = 60 # 1 second

    # Asteroids
    INITIAL_ASTEROIDS = 10
    ASTEROID_DENSITY_INCREASE_INTERVAL = 10 * FPS # every 10 seconds
    ASTEROID_DENSITY_INCREASE_FACTOR = 1.02
    ASTEROID_SPEED_INCREASE_INTERVAL = 30 * FPS # every 30 seconds
    ASTEROID_SPEED_INCREASE_AMOUNT = 0.1
    ASTEROID_MAX_SPEED = 5.0
    ASTEROID_BASE_SPEED_MIN = 1.0
    ASTEROID_BASE_SPEED_MAX = 3.0
    ASTEROID_SIZE_MIN = 15
    ASTEROID_SIZE_MAX = 40

    # Player
    MAX_COLLISIONS = 3

    # Rewards
    REWARD_SURVIVAL = 0.01
    REWARD_COLLISION = -5.0
    REWARD_WIN = 100.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # Game State Variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ship_pos = None
        self.ship_vel = None
        self.ship_angle = 0.0
        self.asteroids = []
        self.particles = []
        self.starfield = []
        self.collision_count = 0
        self.invincibility_timer = 0
        self.current_max_asteroid_speed = 0.0
        self.target_asteroid_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.collision_count = 0
        self.invincibility_timer = self.SHIP_INVINCIBILITY_FRAMES

        # Player state
        self.ship_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float64)
        self.ship_vel = np.array([0.0, 0.0], dtype=np.float64)

        # Difficulty state
        self.current_max_asteroid_speed = self.ASTEROID_BASE_SPEED_MAX
        self.target_asteroid_count = self.INITIAL_ASTEROIDS

        # Entities
        self.asteroids = []
        for _ in range(self.INITIAL_ASTEROIDS):
            self.asteroids.append(self._create_asteroid(spawn_at_edge=True))
        
        self.particles = []
        self._create_starfield()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        self.steps += 1
        reward = self.REWARD_SURVIVAL

        # --- UPDATE GAME LOGIC ---
        self._handle_input(action)
        self._update_ship()
        self._update_asteroids()
        self._update_particles()
        
        collisions_this_frame = self._check_collisions()
        if collisions_this_frame > 0:
            reward += self.REWARD_COLLISION * collisions_this_frame
            self.collision_count += collisions_this_frame
            # Sound placeholder: pygame.mixer.Sound.play(explosion_sound)

        self._update_difficulty()

        # --- CHECK TERMINATION ---
        terminated = (self.collision_count >= self.MAX_COLLISIONS) or (self.steps >= self.MAX_STEPS)
        truncated = False # This environment does not truncate based on time limits, it terminates.
        
        if terminated:
            self.game_over = True
            if self.steps >= self.MAX_STEPS:
                reward += self.REWARD_WIN # Victory bonus
                # Sound placeholder: pygame.mixer.Sound.play(win_sound)

        # Update score (for info dict)
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        
        move_vec = np.array([0, 0], dtype=np.float64)
        if movement == 1: # Up
            move_vec[1] = -1
        elif movement == 2: # Down
            move_vec[1] = 1
        elif movement == 3: # Left
            move_vec[0] = -1
        elif movement == 4: # Right
            move_vec[0] = 1

        if np.linalg.norm(move_vec) > 0:
            self.ship_vel = move_vec * self.SHIP_SPEED
            self.ship_angle = math.atan2(self.ship_vel[1], self.ship_vel[0])
            self._spawn_thruster_particles()
        else:
            self.ship_vel *= 0.95 # Apply some friction/drag

    def _update_ship(self):
        self.ship_pos += self.ship_vel

        # Screen wrapping
        self.ship_pos[0] %= self.SCREEN_WIDTH
        self.ship_pos[1] %= self.SCREEN_HEIGHT
        
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            asteroid['pos'][0] %= self.SCREEN_WIDTH
            asteroid['pos'][1] %= self.SCREEN_HEIGHT
            asteroid['angle'] += asteroid['rot_speed']

    def _update_particles(self):
        # Iterate backwards to allow safe removal
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.pop(i)

    def _check_collisions(self):
        if self.invincibility_timer > 0:
            return 0
        
        collisions = 0
        collided_asteroids = []

        for i, asteroid in enumerate(self.asteroids):
            dist = np.linalg.norm(self.ship_pos - asteroid['pos'])
            if dist < self.SHIP_SIZE + asteroid['size']:
                collisions += 1
                collided_asteroids.append(i)
                self._spawn_explosion_particles(asteroid['pos'])
                self.invincibility_timer = self.SHIP_INVINCIBILITY_FRAMES
                # Break after one collision per frame to avoid multiple penalties
                break 

        # Respawn collided asteroids
        for i in sorted(collided_asteroids, reverse=True):
            self.asteroids.pop(i)
            self.asteroids.append(self._create_asteroid(spawn_at_edge=True))

        return collisions

    def _update_difficulty(self):
        # Increase asteroid speed
        if self.steps > 0 and self.steps % self.ASTEROID_SPEED_INCREASE_INTERVAL == 0:
            new_max_speed = self.current_max_asteroid_speed + self.ASTEROID_SPEED_INCREASE_AMOUNT
            self.current_max_asteroid_speed = min(new_max_speed, self.ASTEROID_MAX_SPEED)

        # Increase asteroid density
        if self.steps > 0 and self.steps % self.ASTEROID_DENSITY_INCREASE_INTERVAL == 0:
            self.target_asteroid_count *= self.ASTEROID_DENSITY_INCREASE_FACTOR
        
        while len(self.asteroids) < int(self.target_asteroid_count):
            self.asteroids.append(self._create_asteroid(spawn_at_edge=True))

    def _get_observation(self):
        # --- RENDER GAME ---
        self.screen.fill(self.COLOR_BG)
        self._render_starfield()
        self._render_asteroids()
        self._render_ship()
        self._render_particles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": (self.MAX_STEPS - self.steps) / self.FPS,
            "collisions": self.collision_count,
            "asteroid_count": len(self.asteroids),
        }

    # --- ENTITY CREATION ---
    def _create_asteroid(self, spawn_at_edge=False):
        size = self.np_random.integers(self.ASTEROID_SIZE_MIN, self.ASTEROID_SIZE_MAX)
        
        if spawn_at_edge:
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), -size], dtype=np.float64)
            elif edge == 1: # Bottom
                pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + size], dtype=np.float64)
            elif edge == 2: # Left
                pos = np.array([-size, self.np_random.uniform(0, self.SCREEN_HEIGHT)], dtype=np.float64)
            else: # Right
                pos = np.array([self.SCREEN_WIDTH + size, self.np_random.uniform(0, self.SCREEN_HEIGHT)], dtype=np.float64)
        else:
            pos = self.np_random.uniform([0, 0], [self.SCREEN_WIDTH, self.SCREEN_HEIGHT])

        speed = self.np_random.uniform(self.ASTEROID_BASE_SPEED_MIN, self.current_max_asteroid_speed)
        angle = self.np_random.uniform(0, 2 * math.pi)
        vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * speed
        
        # Generate polygon shape
        num_vertices = self.np_random.integers(7, 12)
        points = []
        for i in range(num_vertices):
            angle_offset = self.np_random.uniform(-0.2, 0.2)
            radius_offset = self.np_random.uniform(0.8, 1.2)
            a = (2 * math.pi / num_vertices) * i + angle_offset
            r = size * radius_offset
            points.append((r * math.cos(a), r * math.sin(a)))
        
        return {
            "pos": pos,
            "vel": vel,
            "size": size,
            "shape": points,
            "angle": self.np_random.uniform(0, 2 * math.pi),
            "rot_speed": self.np_random.uniform(-0.02, 0.02)
        }

    def _create_starfield(self):
        self.starfield = []
        for _ in range(150): # Far stars
            self.starfield.append({'pos': self.np_random.uniform([0,0], [self.SCREEN_WIDTH, self.SCREEN_HEIGHT]), 'speed': 0.2, 'radius': 1, 'color': (60, 70, 90)})
        for _ in range(70): # Mid stars
            self.starfield.append({'pos': self.np_random.uniform([0,0], [self.SCREEN_WIDTH, self.SCREEN_HEIGHT]), 'speed': 0.5, 'radius': 1, 'color': (100, 110, 130)})
        for _ in range(30): # Near stars
            self.starfield.append({'pos': self.np_random.uniform([0,0], [self.SCREEN_WIDTH, self.SCREEN_HEIGHT]), 'speed': 1.0, 'radius': 2, 'color': (180, 190, 210)})

    def _spawn_explosion_particles(self, pos):
        for _ in range(40):
            speed = self.np_random.uniform(1, 5)
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            life = self.np_random.integers(20, 40)
            color = random.choice(self.COLOR_PARTICLE_EXPLOSION)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'color': color})

    def _spawn_thruster_particles(self):
        if self.steps % 2 == 0: # Spawn less frequently
            num_particles = 2
            for _ in range(num_particles):
                # Spawn particles behind the ship
                angle_offset = self.np_random.uniform(-0.3, 0.3)
                thruster_angle = self.ship_angle + math.pi + angle_offset
                
                speed = self.np_random.uniform(1, 3)
                vel = np.array([math.cos(thruster_angle), math.sin(thruster_angle)]) * speed
                
                start_pos_offset = np.array([math.cos(self.ship_angle + math.pi), math.sin(self.ship_angle + math.pi)]) * self.SHIP_SIZE
                start_pos = self.ship_pos + start_pos_offset

                life = self.np_random.integers(10, 20)
                color = random.choice(self.COLOR_PARTICLE_THRUSTER)
                self.particles.append({'pos': start_pos, 'vel': vel, 'life': life, 'color': color})


    # --- RENDERING ---
    def _render_starfield(self):
        for star in self.starfield:
            # Parallax effect based on ship velocity
            star_pos = star['pos'] - self.ship_vel * star['speed']
            star_pos[0] %= self.SCREEN_WIDTH
            star_pos[1] %= self.SCREEN_HEIGHT
            star['pos'] = star_pos
            pygame.draw.circle(self.screen, star['color'], (int(star_pos[0]), int(star_pos[1])), star['radius'])

    def _render_ship(self):
        # Blinking effect when invincible
        if self.invincibility_timer > 0 and (self.invincibility_timer // 6) % 2 == 0:
            return

        # Define ship points relative to origin
        p1 = (self.SHIP_SIZE, 0)
        p2 = (-self.SHIP_SIZE * 0.7, self.SHIP_SIZE * 0.8)
        p3 = (-self.SHIP_SIZE * 0.7, -self.SHIP_SIZE * 0.8)

        # Rotate points
        angle = self.ship_angle
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        
        def rotate(p):
            return (p[0] * cos_a - p[1] * sin_a, p[0] * sin_a + p[1] * cos_a)

        p1_rot, p2_rot, p3_rot = rotate(p1), rotate(p2), rotate(p3)

        # Translate points to ship position
        points = [
            (int(self.ship_pos[0] + p1_rot[0]), int(self.ship_pos[1] + p1_rot[1])),
            (int(self.ship_pos[0] + p2_rot[0]), int(self.ship_pos[1] + p2_rot[1])),
            (int(self.ship_pos[0] + p3_rot[0]), int(self.ship_pos[1] + p3_rot[1])),
        ]

        # Draw ship with antialiasing
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_SHIP)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_SHIP)
    
    def _render_asteroids(self):
        for asteroid in self.asteroids:
            # Rotate and translate vertices
            cos_a, sin_a = math.cos(asteroid['angle']), math.sin(asteroid['angle'])
            points = []
            for p in asteroid['shape']:
                x_rot = p[0] * cos_a - p[1] * sin_a
                y_rot = p[0] * sin_a + p[1] * cos_a
                points.append((int(asteroid['pos'][0] + x_rot), int(asteroid['pos'][1] + y_rot)))
            
            # Draw asteroid with antialiasing
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20.0))))
            color_with_alpha = p['color'] + (alpha,)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (1, 1), 1)
            self.screen.blit(temp_surf, (int(p['pos'][0] - 1), int(p['pos'][1] - 1)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Time remaining
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = f"TIME: {time_left:.1f}"
        
        # Collisions (lives remaining)
        lives_left = max(0, self.MAX_COLLISIONS - self.collision_count)
        collision_text = f"HULL: {'I' * lives_left}"

        # Render with shadow
        self._draw_text(time_text, (12, 10), self.font_large)
        self._draw_text(collision_text, (self.SCREEN_WIDTH - 12, 10), self.font_large, align="right")
        
        if self.game_over:
            if self.steps >= self.MAX_STEPS:
                end_text = "SURVIVAL COMPLETE"
            else:
                end_text = "GAME OVER"
            self._draw_text(end_text, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2), self.font_large, align="center")

    def _draw_text(self, text, pos, font, align="left"):
        text_surf = font.render(text, True, self.COLOR_UI_TEXT)
        shadow_surf = font.render(text, True, self.COLOR_UI_SHADOW)
        
        text_rect = text_surf.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "right":
            text_rect.topright = pos
        else:
            text_rect.topleft = pos
            
        shadow_rect = text_rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2

        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

# Example usage to run and visualize the game
if __name__ == '__main__':
    # This block will not run in the test environment, but is useful for human play.
    # It requires a display, so we unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame window for human play ---
    pygame.display.set_caption("Asteroid Survival")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    print("R: Reset")
    
    while not done:
        # --- Human Input ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # Space and Shift not used

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    print("--- Game Reset ---")

        # --- Rendering ---
        # The observation is the rendered frame, so we just blit it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Check for termination from env ---
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            
        clock.tick(GameEnv.FPS)

    env.close()