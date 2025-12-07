
# Generated: 2025-08-28T00:15:20.507823
# Source Brief: brief_01551.md
# Brief Index: 1551

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 100
        self.INITIAL_HEALTH = 3
        self.INITIAL_ASTEROIDS = 8
        self.MAX_ASTEROIDS = 15

        # Player physics
        self.PLAYER_ACCELERATION = 0.3
        self.PLAYER_TURN_SPEED = 5 # degrees
        self.PLAYER_BRAKE_FORCE = 0.2
        self.PLAYER_FRICTION = 0.985
        self.PLAYER_MAX_SPEED = 8
        self.PLAYER_SIZE = 12

        # Asteroid properties
        self.ASTEROID_MIN_SPEED = 0.5
        self.ASTEROID_MAX_SPEED = 2.0
        self.ASTEROID_MIN_SIZE = 15
        self.ASTEROID_MAX_SIZE = 40

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_DRIFT = (50, 255, 255)
        self.COLOR_ASTEROID = (120, 120, 130)
        self.COLOR_ORE = (255, 223, 0)
        self.COLOR_LASER = (200, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_ICON = (50, 255, 50)
        self.COLOR_HEALTH_ICON_LOST = (60, 60, 60)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.health = None
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_asteroid_speed = self.ASTEROID_MIN_SPEED

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.health = self.INITIAL_HEALTH
        self.current_asteroid_speed = self.ASTEROID_MIN_SPEED

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        self.player_angle = -90.0 # Pointing up

        self.asteroids = []
        for _ in range(self.INITIAL_ASTEROIDS):
            self._spawn_asteroid(on_screen=True)

        self.particles = []
        self.stars = [
            {
                "pos": np.array([random.uniform(0, self.WIDTH), random.uniform(0, self.HEIGHT)]),
                "depth": random.uniform(0.1, 0.8),
            }
            for _ in range(150)
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        if self.game_over:
            return self._get_observation(), reward, True, False, self._get_info()
            
        self.steps += 1
        
        self._handle_input(action)
        self._update_player()
        self._update_asteroids()
        self._update_particles()
        self._update_stars()

        reward += self._handle_mining(action)
        collision_reward, collision_occurred = self._handle_collisions()
        reward += collision_reward

        if not collision_occurred:
            # Small reward for surviving
            reward += 0.001

        self._spawn_asteroids_if_needed()
        self._update_difficulty()

        terminated = self.health <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100.0  # Victory reward
            elif self.health <= 0:
                reward -= 50.0  # Defeat penalty
            # No specific penalty for timeout, the lack of victory reward is sufficient

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Turning
        if movement == 3: # Left
            self.player_angle -= self.PLAYER_TURN_SPEED
        if movement == 4: # Right
            self.player_angle += self.PLAYER_TURN_SPEED

        # Thrust / Brake
        if not shift_held: # Can't thrust while drifting
            if movement == 1: # Up
                rad_angle = math.radians(self.player_angle)
                acceleration = np.array([math.cos(rad_angle), math.sin(rad_angle)]) * self.PLAYER_ACCELERATION
                self.player_vel += acceleration
                # Thrust particles
                if self.steps % 2 == 0:
                    self._create_thrust_particles()
            elif movement == 2: # Down
                self.player_vel *= (1.0 - self.PLAYER_BRAKE_FORCE)

    def _update_player(self):
        # Apply friction
        self.player_vel *= self.PLAYER_FRICTION
        
        # Clamp speed
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel / speed * self.PLAYER_MAX_SPEED

        # Update position
        self.player_pos += self.player_vel

        # Screen wrap
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid["pos"] += asteroid["vel"]
            asteroid["pos"][0] %= self.WIDTH
            asteroid["pos"][1] %= self.HEIGHT
            asteroid["angle"] += asteroid["rot_speed"]

    def _update_particles(self):
        # Iterate backwards to allow safe removal
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.pop(i)

    def _update_stars(self):
        for star in self.stars:
            # Parallax effect: stars move opposite to player, scaled by depth
            star["pos"] -= self.player_vel * star["depth"]
            star["pos"][0] %= self.WIDTH
            star["pos"][1] %= self.HEIGHT

    def _handle_mining(self, action):
        _, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        if not space_held:
            return reward

        rad_angle = math.radians(self.player_angle)
        laser_dir = np.array([math.cos(rad_angle), math.sin(rad_angle)])
        laser_end = self.player_pos + laser_dir * self.WIDTH # Effectively infinite length

        mined_asteroids = []
        for i, asteroid in enumerate(self.asteroids):
            # Simple line-circle intersection
            dist_to_asteroid = np.linalg.norm(asteroid["pos"] - self.player_pos)
            if dist_to_asteroid > 300: continue # Optimization

            p_to_a = asteroid["pos"] - self.player_pos
            proj = np.dot(p_to_a, laser_dir)
            if proj < 0: continue

            dist_sq = np.dot(p_to_a, p_to_a) - proj * proj
            if dist_sq < asteroid["size"] ** 2:
                # Hit!
                asteroid["health"] -= 1
                hit_pos = self.player_pos + laser_dir * proj
                self._create_hit_sparks(hit_pos)
                # sfx: mining_hit.wav

                if asteroid["health"] <= 0:
                    mined_asteroids.append(i)
                    ore_gained = int(asteroid["size"] / 5)
                    self.score = min(self.WIN_SCORE, self.score + ore_gained)
                    reward += ore_gained * 0.1
                    self._create_ore_particles(asteroid["pos"], ore_gained)
                    # sfx: asteroid_destroyed.wav
                break # Laser hits only one asteroid at a time

        # Remove destroyed asteroids
        for i in sorted(mined_asteroids, reverse=True):
            del self.asteroids[i]
        
        return reward

    def _handle_collisions(self):
        reward = 0.0
        collision_occurred = False
        collided_asteroids = []
        for i, asteroid in enumerate(self.asteroids):
            dist = np.linalg.norm(self.player_pos - asteroid["pos"])
            if dist < self.PLAYER_SIZE + asteroid["size"]:
                collided_asteroids.append(i)
        
        if collided_asteroids:
            self.health -= 1
            reward -= 1.0 # Collision penalty
            collision_occurred = True
            self._create_explosion(self.player_pos)
            # sfx: player_hit.wav
            
            # Reset player to center to give a moment of recovery
            self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
            self.player_vel = np.array([0.0, 0.0], dtype=float)

            # Remove collided asteroids
            for i in sorted(collided_asteroids, reverse=True):
                self._create_explosion(self.asteroids[i]["pos"])
                del self.asteroids[i]
        
        return reward, collision_occurred

    def _spawn_asteroid(self, on_screen=False):
        if on_screen:
            pos = np.array([random.uniform(0, self.WIDTH), random.uniform(0, self.HEIGHT)])
        else:
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                pos = np.array([random.uniform(0, self.WIDTH), -self.ASTEROID_MAX_SIZE])
            elif edge == 'bottom':
                pos = np.array([random.uniform(0, self.WIDTH), self.HEIGHT + self.ASTEROID_MAX_SIZE])
            elif edge == 'left':
                pos = np.array([-self.ASTEROID_MAX_SIZE, random.uniform(0, self.HEIGHT)])
            else: # right
                pos = np.array([self.WIDTH + self.ASTEROID_MAX_SIZE, random.uniform(0, self.HEIGHT)])

        # Ensure it doesn't spawn on the player
        while np.linalg.norm(pos - self.player_pos) < self.ASTEROID_MAX_SIZE * 3:
            pos = np.array([random.uniform(0, self.WIDTH), random.uniform(0, self.HEIGHT)])

        angle = random.uniform(0, 360)
        rad_angle = math.radians(angle)
        speed = random.uniform(self.current_asteroid_speed, self.current_asteroid_speed + 1.0)
        vel = np.array([math.cos(rad_angle), math.sin(rad_angle)]) * speed
        size = random.uniform(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE)
        
        self.asteroids.append({
            "pos": pos,
            "vel": vel,
            "size": size,
            "health": int(size / 5),
            "angle": random.uniform(0, 360),
            "rot_speed": random.uniform(-1, 1),
            "shape": self._create_asteroid_shape(size)
        })

    def _create_asteroid_shape(self, size):
        points = []
        num_vertices = random.randint(7, 12)
        for i in range(num_vertices):
            angle = i * (2 * math.pi / num_vertices)
            dist = random.uniform(size * 0.7, size)
            points.append((dist * math.cos(angle), dist * math.sin(angle)))
        return points

    def _spawn_asteroids_if_needed(self):
        if len(self.asteroids) < self.MAX_ASTEROIDS and self.steps % 30 == 0:
            self._spawn_asteroid()

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 100 == 0:
            self.current_asteroid_speed = min(self.ASTEROID_MAX_SPEED, self.current_asteroid_speed + 0.05)

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.health}

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_particles()
        self._render_asteroids()
        self._render_player()

    def _render_stars(self):
        for star in self.stars:
            pos = star["pos"]
            # Darker stars for more depth
            brightness = int(200 * star["depth"])
            color = (brightness, brightness, brightness)
            size = int(star["depth"] * 2)
            pygame.draw.rect(self.screen, color, (int(pos[0]), int(pos[1]), size, size))

    def _render_player(self):
        _, _, shift_held = self.action_space.sample() if len(self.last_action) == 0 else self.last_action
        shift_held = shift_held == 1
        
        color = self.COLOR_PLAYER_DRIFT if shift_held else self.COLOR_PLAYER
        rad_angle = math.radians(self.player_angle)
        
        p1 = (
            self.player_pos[0] + self.PLAYER_SIZE * math.cos(rad_angle),
            self.player_pos[1] + self.PLAYER_SIZE * math.sin(rad_angle)
        )
        p2 = (
            self.player_pos[0] + self.PLAYER_SIZE * math.cos(rad_angle + math.radians(140)),
            self.player_pos[1] + self.PLAYER_SIZE * math.sin(rad_angle + math.radians(140))
        )
        p3 = (
            self.player_pos[0] + self.PLAYER_SIZE * math.cos(rad_angle - math.radians(140)),
            self.player_pos[1] + self.PLAYER_SIZE * math.sin(rad_angle - math.radians(140))
        )
        points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3]]
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Mining laser
        _, space_held, _ = self.last_action if hasattr(self, 'last_action') else (0,0,0)
        if space_held == 1:
            end_pos = self.player_pos + np.array([math.cos(rad_angle), math.sin(rad_angle)]) * self.WIDTH
            pygame.draw.aaline(self.screen, self.COLOR_LASER, (int(p1[0]), int(p1[1])), (int(end_pos[0]), int(end_pos[1])), 2)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            pos = asteroid["pos"]
            rad_angle = math.radians(asteroid["angle"])
            
            points = []
            for p in asteroid["shape"]:
                x = p[0] * math.cos(rad_angle) - p[1] * math.sin(rad_angle) + pos[0]
                y = p[0] * math.sin(rad_angle) + p[1] * math.cos(rad_angle) + pos[1]
                points.append((int(x), int(y)))

            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            radius = int(p["radius"] * (p["life"] / p["max_life"]))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p["color"])

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"ORE: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Health
        for i in range(self.INITIAL_HEALTH):
            color = self.COLOR_HEALTH_ICON if i < self.health else self.COLOR_HEALTH_ICON_LOST
            self._draw_health_icon(self.WIDTH - 20 - (i * 25), 20, color)
        
        # Game Over Message
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "VICTORY!"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = (255, 50, 50)
            
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def _draw_health_icon(self, x, y, color):
        points = [(x, y - 8), (x - 6, y + 6), (x + 6, y + 6)]
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _create_explosion(self, pos):
        # sfx: explosion.wav
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": random.randint(2, 5),
                "color": random.choice([(255, 50, 50), (255, 150, 0), (255, 255, 100)]),
                "life": random.randint(20, 40),
                "max_life": 40
            })

    def _create_thrust_particles(self):
        rad_angle = math.radians(self.player_angle + 180) # Opposite direction
        for _ in range(2):
            offset_angle = rad_angle + random.uniform(-0.3, 0.3)
            vel = np.array([math.cos(offset_angle), math.sin(offset_angle)]) * 2 + self.player_vel
            pos_offset = np.array([math.cos(rad_angle), math.sin(rad_angle)]) * self.PLAYER_SIZE
            self.particles.append({
                "pos": self.player_pos + pos_offset,
                "vel": vel,
                "radius": random.randint(1, 3),
                "color": (255, 200, 150),
                "life": random.randint(10, 20),
                "max_life": 20
            })

    def _create_hit_sparks(self, pos):
        for _ in range(5):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": random.randint(1, 2),
                "color": self.COLOR_LASER,
                "life": random.randint(5, 10),
                "max_life": 10
            })

    def _create_ore_particles(self, pos, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": random.randint(2, 4),
                "color": self.COLOR_ORE,
                "life": random.randint(25, 50),
                "max_life": 50
            })

    def close(self):
        pygame.quit()

    def render(self):
        return self._get_observation()

    def _get_observation(self):
        if not hasattr(self, 'last_action'):
             self.last_action = self.action_space.sample()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def step(self, action):
        self.last_action = action
        reward = 0.0
        
        if self.game_over:
            return self._get_observation(), reward, True, False, self._get_info()
            
        self.steps += 1
        
        self._handle_input(action)
        self._update_player()
        self._update_asteroids()
        self._update_particles()
        self._update_stars()

        reward += self._handle_mining(action)
        collision_reward, collision_occurred = self._handle_collisions()
        reward += collision_reward

        if not collision_occurred:
            reward += 0.001

        self._spawn_asteroids_if_needed()
        self._update_difficulty()

        terminated = self.health <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100.0
            elif self.health <= 0:
                reward -= 50.0

        return self._get_observation(), reward, terminated, False, self._get_info()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # To display the game, we need to create a window
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = [0, 0, 0] # No-op
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                done = True

        # Human controls
        keys = pygame.key.get_pressed()
        
        # Movement
        mov = 0 # none
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        # Buttons
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for human play

    print(f"Game Over. Final Info: {info}")
    env.close()