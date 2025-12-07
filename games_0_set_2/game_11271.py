import gymnasium as gym
import os
import pygame
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Particle:
    """A simple class for particles used in effects."""
    def __init__(self, pos, vel, radius, color, lifetime):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.radius = radius
        self.color = color
        self.lifetime = lifetime
        self.initial_lifetime = lifetime

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1
        self.radius = max(0, self.radius * 0.98)

    def draw(self, surface):
        if self.lifetime > 0:
            # Fade out effect
            alpha = int(255 * (self.lifetime / self.initial_lifetime))
            current_color = (*self.color, alpha)
            
            temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, current_color, (self.radius, self.radius), self.radius)
            surface.blit(temp_surf, (int(self.pos.x - self.radius), int(self.pos.y - self.radius)), special_flags=pygame.BLEND_RGBA_ADD)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Dodge waves of incoming asteroids in your nimble spacecraft. The longer you survive, the faster they get!"
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to move your ship and avoid the asteroids."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 10, 42) # #0a0a2a
    COLOR_PLAYER = (255, 255, 255)
    COLOR_ASTEROID = (255, 68, 68) # #ff4444
    COLOR_ASTEROID_GLOW = (128, 0, 0)
    COLOR_TRAIL = (0, 170, 255) # #00aaff
    COLOR_THRUSTER = (255, 200, 0)
    COLOR_UI_TEXT = (200, 200, 220)
    
    # Player
    PLAYER_RADIUS = 12
    PLAYER_THRUST = 0.6
    PLAYER_DRAG = 0.96
    PLAYER_MAX_SPEED = 10.0

    # Asteroids
    ASTEROID_MAX_COUNT = 8
    ASTEROID_SPAWN_CHANCE = 0.05
    ASTEROID_MIN_RADIUS = 15
    ASTEROID_MAX_RADIUS = 35

    # Game
    WIN_CONDITION_AVOIDED = 10
    MAX_EPISODE_STEPS = 1000
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        
        # State variables
        self.player_pos = None
        self.player_vel = None
        self.player_speed_multiplier = None
        self.asteroids = []
        self.stars = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.asteroids_avoided = 0
        self.consecutive_dodges = 0
        self.asteroid_base_speed = 0
        self.game_over = False
        
        # self.reset() is called by the wrapper, but we can call it here for initialization
        # The seed will be properly handled by the environment wrapper
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset player state
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.8)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_speed_multiplier = 1.0

        # Reset game state
        self.steps = 0
        self.score = 0.0
        self.asteroids_avoided = 0
        self.consecutive_dodges = 0
        self.game_over = False
        
        # Reset entities
        self.asteroids = []
        self.particles = []
        self.asteroid_base_speed = 2.0
        
        # Generate starfield
        self.stars = []
        for _ in range(150):
            self.stars.append({
                "pos": pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)),
                "size": self.np_random.uniform(0.5, 2.0),
                "speed_mult": self.np_random.uniform(0.1, 0.5)
            })
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1 # Survival reward

        # --- UPDATE GAME LOGIC ---
        self._handle_input(action)
        self._update_player()
        reward += self._update_asteroids()
        self._update_particles()
        
        # --- CHECK COLLISIONS & TERMINATION ---
        collision = self._check_collisions()
        terminated = self.game_over or collision
        truncated = False
        
        if collision:
            reward = -100.0
            self.score += reward
            self._create_explosion(self.player_pos, self.COLOR_PLAYER, 50)
            self.game_over = True
            terminated = True
        elif self.asteroids_avoided >= self.WIN_CONDITION_AVOIDED:
            reward = 100.0
            self.score += reward
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True
            self.game_over = True

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
        thrust_direction = pygame.Vector2(0, 0)
        
        if movement == 1: thrust_direction.y = -1 # Up
        elif movement == 2: thrust_direction.y = 1  # Down
        elif movement == 3: thrust_direction.x = -1 # Left
        elif movement == 4: thrust_direction.x = 1  # Right

        if thrust_direction.length() > 0:
            self.player_vel += thrust_direction.normalize() * self.PLAYER_THRUST * self.player_speed_multiplier
            self._create_thruster_effect(thrust_direction)

    def _update_player(self):
        # Apply drag
        self.player_vel *= self.PLAYER_DRAG
        
        # Limit speed
        if self.player_vel.length() > self.PLAYER_MAX_SPEED * self.player_speed_multiplier:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED * self.player_speed_multiplier)
        
        # Update position
        self.player_pos += self.player_vel
        
        # Clamp to screen boundaries
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

    def _update_asteroids(self):
        reward = 0
        
        # Move and despawn asteroids
        for asteroid in self.asteroids[:]:
            asteroid["pos"].y += self.asteroid_base_speed
            if asteroid["pos"].y > self.SCREEN_HEIGHT + asteroid["radius"]:
                self.asteroids.remove(asteroid)
                
                # Reward for avoiding
                reward += 1.0
                self.asteroids_avoided += 1
                self.consecutive_dodges += 1
                
                # Scale difficulty
                if self.asteroids_avoided > 0 and self.asteroids_avoided % 2 == 0:
                    self.asteroid_base_speed += 0.1
                
                # Increase speed boost
                if self.consecutive_dodges > 0 and self.consecutive_dodges % 3 == 0:
                    self.player_speed_multiplier += 0.15

        # Spawn new asteroids
        if len(self.asteroids) < self.ASTEROID_MAX_COUNT and self.np_random.random() < self.ASTEROID_SPAWN_CHANCE:
            self._spawn_asteroid()
            
        return reward

    def _spawn_asteroid(self):
        radius = self.np_random.uniform(self.ASTEROID_MIN_RADIUS, self.ASTEROID_MAX_RADIUS)
        x_pos = self.np_random.uniform(radius, self.SCREEN_WIDTH - radius)
        pos = pygame.Vector2(x_pos, -radius)
        
        # Anti-softlock: Ensure it doesn't spawn too close to another new asteroid
        can_spawn = True
        for ast in self.asteroids:
            if ast["pos"].y < 0 and abs(ast["pos"].x - pos.x) < radius * 2 + ast["radius"]:
                can_spawn = False
                break
        
        if can_spawn:
            self.asteroids.append({"pos": pos, "radius": radius})

    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.lifetime <= 0:
                self.particles.remove(p)

    def _check_collisions(self):
        for asteroid in self.asteroids:
            distance = self.player_pos.distance_to(asteroid["pos"])
            if distance < self.PLAYER_RADIUS + asteroid["radius"]:
                return True
        return False

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_effects()
        self._render_game_objects()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.stars:
            star["pos"].y += star["speed_mult"] * self.asteroid_base_speed * 0.5
            if star["pos"].y > self.SCREEN_HEIGHT:
                star["pos"].y = 0
                star["pos"].x = self.np_random.uniform(0, self.SCREEN_WIDTH)
            
            size = int(star["size"])
            color_val = int(100 + star["speed_mult"] * 155)
            pygame.draw.rect(self.screen, (color_val, color_val, color_val), (int(star["pos"].x), int(star["pos"].y), size, size))

    def _render_effects(self):
        # Player speed trail
        if self.player_speed_multiplier > 1.1:
            trail_pos = self.player_pos.copy()
            trail_vel = -self.player_vel.normalize() * 2 if self.player_vel.length() > 0 else pygame.Vector2(0, 1)
            p = Particle(trail_pos, trail_vel, self.PLAYER_RADIUS * 0.7, self.COLOR_TRAIL, 20)
            self.particles.append(p)

        # Draw all particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_game_objects(self):
        # Draw asteroids with glow
        for asteroid in self.asteroids:
            pos = (int(asteroid["pos"].x), int(asteroid["pos"].y))
            radius = int(asteroid["radius"])
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius * 1.5), self.COLOR_ASTEROID_GLOW)
            # Main body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)
            
        # Draw player if not game over (but not if just collided)
        if not (self.game_over and self._check_collisions()):
            pos = (int(self.player_pos.x), int(self.player_pos.y))
            # Main body
            pygame.draw.circle(self.screen, self.COLOR_PLAYER, pos, self.PLAYER_RADIUS)
            # "Cockpit"
            cockpit_dir = self.player_vel.normalize() if self.player_vel.length() > 0.1 else pygame.Vector2(0, -1)
            cockpit_pos = self.player_pos + cockpit_dir * (self.PLAYER_RADIUS * 0.5)
            pygame.draw.circle(self.screen, self.COLOR_TRAIL, (int(cockpit_pos.x), int(cockpit_pos.y)), 3)

    def _render_ui(self):
        # Speed display
        speed_text = f"SPEED: {self.player_speed_multiplier:.2f}x"
        speed_surf = self.font.render(speed_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_surf, (10, 10))

        # Avoided count display
        avoid_text = f"AVOIDED: {self.asteroids_avoided}/{self.WIN_CONDITION_AVOIDED}"
        avoid_surf = self.font.render(avoid_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(avoid_surf, (self.SCREEN_WIDTH - avoid_surf.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "asteroids_avoided": self.asteroids_avoided,
            "player_speed_multiplier": self.player_speed_multiplier,
            "asteroid_speed": self.asteroid_base_speed,
        }

    def _create_thruster_effect(self, direction):
        for _ in range(3):
            # Eject particles opposite to thrust direction
            vel = -direction.normalize() * self.np_random.uniform(2, 4)
            vel = vel.rotate(self.np_random.uniform(-20, 20))
            
            start_pos = self.player_pos - direction.normalize() * self.PLAYER_RADIUS
            
            p = Particle(
                start_pos,
                vel,
                radius=self.np_random.uniform(2, 5),
                color=self.COLOR_THRUSTER,
                lifetime=self.np_random.integers(10, 21) # randint(10, 20) is inclusive
            )
            self.particles.append(p)

    def _create_explosion(self, position, color, num_particles):
        for _ in range(num_particles):
            vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
            if vel.length() > 0:
                vel = vel.normalize() * self.np_random.uniform(1, 8)
            
            p = Particle(
                position,
                vel,
                radius=self.np_random.uniform(1, 4),
                color=color,
                lifetime=self.np_random.integers(20, 51) # randint(20, 50) is inclusive
            )
            self.particles.append(p)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It is not part of the Gymnasium environment API
    # You can run this file directly to play
    
    # Un-dummy the video driver for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for manual play
    pygame.display.set_caption("Neon Asteroid Dodger")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        # The other action dimensions are not used in this game's logic
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(60) # Run at 60 FPS for smooth manual play
        
    env.close()