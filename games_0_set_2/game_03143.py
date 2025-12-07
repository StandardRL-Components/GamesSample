import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your ship. Hold space to mine the nearest asteroid."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Mine asteroids in a procedurally generated field. Collect ore to win, but watch out for collisions!"
    )

    # Frames auto-advance at 30fps.
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PLAYER_SIZE = 10
    PLAYER_SPEED = 5.0
    INITIAL_HEALTH = 100
    LASER_RANGE = 150
    LASER_POWER = 0.5
    COLLISION_DAMAGE = 25
    
    WIN_ORE = 500
    MAX_STEPS = 3000 # Increased to allow more time for the ore goal
    INITIAL_ASTEROIDS = 15
    MAX_ASTEROIDS = 20

    # --- Colors ---
    COLOR_BG = (15, 18, 32)
    COLOR_PLAYER = (0, 255, 127) # Bright green
    COLOR_PLAYER_GLOW = (0, 255, 127, 30)
    COLOR_ASTEROID_BASE = (90, 90, 110)
    COLOR_ASTEROID_HIGH = (200, 200, 220)
    COLOR_LASER = (255, 20, 50)
    COLOR_EXPLOSION = (255, 165, 0)
    COLOR_IMPACT_SPARK = (255, 255, 100)
    COLOR_TEXT = (240, 240, 255)
    COLOR_HEALTH_BAR = (220, 40, 50)
    COLOR_HEALTH_BAR_BG = (80, 20, 25)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        self.player_pos = pygame.Vector2(0, 0)
        self.player_health = 0
        self.ore_collected = 0
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.game_over = False
        self.mining_target = None
        self.mining_beam_active = False
        self.rng = None
        
        # This is not part of the official API but is useful for the example
        self.render_mode = render_mode
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = self.INITIAL_HEALTH
        self.ore_collected = 0
        self.steps = 0
        self.game_over = False
        self.mining_target = None
        self.mining_beam_active = False

        self.asteroids = []
        for _ in range(self.INITIAL_ASTEROIDS):
            self._spawn_asteroid()

        self.particles = []
        self.stars = [
            (
                self.rng.integers(0, self.WIDTH),
                self.rng.integers(0, self.HEIGHT),
                self.rng.random() * 1.5
            ) for _ in range(100)
        ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Cost of living per step
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self._handle_movement(movement)
        mine_reward = self._handle_mining(space_held)
        reward += mine_reward
        
        self._update_asteroids()
        self._update_particles()
        
        collision_reward = self._check_collisions()
        reward += collision_reward

        if self.rng.random() < 0.1 and len(self.asteroids) < self.MAX_ASTEROIDS:
            self._spawn_asteroid()

        terminated = False
        truncated = False
        if self.player_health <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.ore_collected >= self.WIN_ORE:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            self.game_over = True
            
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_movement(self, movement):
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y -= 1
        elif movement == 2: move_vec.y += 1
        elif movement == 3: move_vec.x -= 1
        elif movement == 4: move_vec.x += 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED
        
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _handle_mining(self, space_held):
        self.mining_beam_active = False
        self.mining_target = None
        reward = 0

        if space_held:
            closest_asteroid = None
            min_dist_sq = self.LASER_RANGE ** 2
            
            for asteroid in self.asteroids:
                # FIX: Use length_squared() on the difference vector instead of the non-existent distance_to_squared()
                dist_sq = (self.player_pos - asteroid['pos']).length_squared()
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_asteroid = asteroid
            
            if closest_asteroid:
                # Sound effect: Laser hum
                self.mining_beam_active = True
                self.mining_target = closest_asteroid
                
                mined_amount = min(closest_asteroid['ore'], self.LASER_POWER)
                closest_asteroid['ore'] -= mined_amount
                self.ore_collected += mined_amount
                reward += mined_amount * 0.1
                
                closest_asteroid['radius'] = max(5, closest_asteroid['initial_radius'] * (closest_asteroid['ore'] / closest_asteroid['initial_ore']))
                
                if self.rng.random() < 0.7:
                    self._create_particles(closest_asteroid['pos'], 1, self.COLOR_IMPACT_SPARK, 1, 10, 2)

        return reward

    def _update_asteroids(self):
        destroyed_asteroids = []
        for asteroid in self.asteroids:
            if asteroid['ore'] <= 0:
                destroyed_asteroids.append(asteroid)
        
        for asteroid in destroyed_asteroids:
            # Sound effect: Explosion
            self._create_particles(asteroid['pos'], 20, self.COLOR_EXPLOSION, 2, 25, 4)
            self.asteroids.remove(asteroid)

    def _check_collisions(self):
        reward = 0
        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid['pos'])
            if dist < self.PLAYER_SIZE + asteroid['radius']:
                # Sound effect: Hull impact
                self.player_health -= self.COLLISION_DAMAGE
                reward -= 1 # Small penalty for collision in addition to health loss
                self._create_particles(self.player_pos, 5, self.COLOR_EXPLOSION, 1, 15, 3)
                
                # Push player away from asteroid
                push_vec = self.player_pos - asteroid['pos']
                if push_vec.length() > 0:
                    self.player_pos += push_vec.normalize() * (self.PLAYER_SIZE + asteroid['radius'] - dist)
                
        return reward

    def _spawn_asteroid(self):
        # Difficulty scaling: higher ore content as game progresses
        ore_multiplier = 1 + (self.ore_collected // 50) * 0.1
        ore = self.rng.integers(5, 20) * ore_multiplier
        radius = 5 + int(ore * 0.5)
        
        # Spawn away from player
        while True:
            pos = pygame.Vector2(
                self.rng.integers(radius, self.WIDTH - radius),
                self.rng.integers(radius, self.HEIGHT - radius)
            )
            if pos.distance_to(self.player_pos) > self.LASER_RANGE * 1.2:
                break
        
        # Generate a procedural shape
        num_points = self.rng.integers(7, 12)
        shape = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            rad = radius * self.rng.uniform(0.8, 1.2)
            shape.append((math.cos(angle) * rad, math.sin(angle) * rad))

        self.asteroids.append({
            'pos': pos,
            'radius': radius,
            'ore': ore,
            'initial_radius': radius,
            'initial_ore': ore,
            'shape': shape,
            'angle': self.rng.random() * 2 * math.pi
        })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_asteroids()
        self._render_player()
        self._render_laser()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_background(self):
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, (200, 200, 220), (x, y), size)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            ore_ratio = np.clip(asteroid['ore'] / (asteroid['initial_ore'] * 1.2), 0, 1)
            color = tuple(int(c1 + (c2 - c1) * ore_ratio) for c1, c2 in zip(self.COLOR_ASTEROID_BASE, self.COLOR_ASTEROID_HIGH))
            
            points = [(p[0] + asteroid['pos'].x, p[1] + asteroid['pos'].y) for p in asteroid['shape']]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_SIZE + 5, self.COLOR_PLAYER_GLOW)
        
        # Ship body
        points = [
            (pos[0], pos[1] - self.PLAYER_SIZE),
            (pos[0] - self.PLAYER_SIZE * 0.7, pos[1] + self.PLAYER_SIZE * 0.7),
            (pos[0] + self.PLAYER_SIZE * 0.7, pos[1] + self.PLAYER_SIZE * 0.7)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_laser(self):
        if self.mining_beam_active and self.mining_target:
            start_pos = (int(self.player_pos.x), int(self.player_pos.y))
            end_pos = (int(self.mining_target['pos'].x), int(self.mining_target['pos'].y))
            
            # Pulsating width
            width = int(2 + math.sin(pygame.time.get_ticks() * 0.05) * 1.5)
            
            # Draw multiple lines for a glow effect
            pygame.draw.line(self.screen, (255, 100, 100, 100), start_pos, end_pos, width + 4)
            pygame.draw.line(self.screen, self.COLOR_LASER, start_pos, end_pos, width)

    def _render_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] > 0:
                alpha = int(255 * (p['life'] / p['max_life']))
                color = p['color'] + (alpha,)
                size = int(p['size'] * (p['life'] / p['max_life']))
                if size > 0:
                    temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, color, (size, size), size)
                    self.screen.blit(temp_surf, (p['pos'].x - size, p['pos'].y - size), special_flags=pygame.BLEND_RGBA_ADD)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _create_particles(self, pos, count, color, speed, lifespan, size):
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.rng.uniform(0.5, 1.0) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.rng.integers(lifespan//2, lifespan),
                'max_life': lifespan,
                'color': color,
                'size': self.rng.uniform(size//2, size)
            })

    def _render_ui(self):
        # Ore count
        ore_text = self.font_small.render(f"ORE: {int(self.ore_collected)} / {self.WIN_ORE}", True, self.COLOR_TEXT)
        self.screen.blit(ore_text, (10, 10))
        
        # Health bar
        health_ratio = np.clip(self.player_health / self.INITIAL_HEALTH, 0, 1)
        bar_width = 150
        bar_height = 15
        bar_x = self.WIDTH - bar_width - 10
        bar_y = 10
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, int(bar_width * health_ratio), bar_height))
        
        # Game over / Victory message
        if self.game_over:
            if self.ore_collected >= self.WIN_ORE:
                msg = "VICTORY!"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_HEALTH_BAR
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.ore_collected,
            "steps": self.steps,
            "health": self.player_health,
            "asteroids": len(self.asteroids)
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == "__main__":
    # To run with a display window, you would change the render_mode
    # and add a `render` method.
    # For this script, we'll stick to the default 'rgb_array' mode.
    # Example: env = GameEnv(render_mode='human')
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # --- Manual Play ---
    # This example demonstrates headless interaction. For 'human' mode,
    # you would need to handle rendering to a display window.
    
    # Create a minimal window to capture key presses
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    
    print("\n" + GameEnv.user_guide)
    print("Running headless demo. Focus the Pygame window to control.")
    
    terminated = False
    truncated = False
    total_reward = 0
    
    for step_count in range(1, 2001):
        # Map keyboard to MultiDiscrete action
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step_count % 100 == 0:
            print(f"Step {info['steps']}: Score={info['score']:.1f}, Health={info['health']:.0f}, Total Reward={total_reward:.2f}")

        if terminated or truncated:
            reason = "Terminated" if terminated else "Truncated"
            print(f"Episode finished after {info['steps']} steps ({reason}). Final Score: {info['score']:.1f}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset(seed=42)
            total_reward = 0

    env.close()