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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold space to mine nearby asteroids."
    )

    game_description = (
        "Pilot a mining ship through an asteroid field, collecting ore before time runs out. "
        "Collide with an asteroid and it's game over."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.WIN_SCORE = 100
        self.MAX_STEPS = 6000  # 60 seconds at 100 steps/sec as per brief
        self.NUM_ASTEROIDS = 15
        self.NUM_STARS = 150

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_STAR = (150, 150, 170)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_THRUSTER = (255, 180, 50)
        self.COLOR_ASTEROID = (120, 120, 120)
        self.COLOR_ORE = (255, 223, 0)
        self.COLOR_BEAM = (100, 200, 255, 100)
        self.COLOR_EXPLOSION = [(255, 50, 50), (255, 150, 50)]
        self.COLOR_TEXT = (255, 255, 255)
        
        # Physics constants
        self.PLAYER_THRUST = 0.15
        self.PLAYER_ROTATION_SPEED = 3.5
        self.PLAYER_DRAG = 0.98
        self.PLAYER_BRAKE_DRAG = 0.92
        self.PLAYER_MAX_SPEED = 5
        self.PLAYER_RADIUS = 10
        self.ASTEROID_MIN_RADIUS = 15
        self.ASTEROID_MAX_RADIUS = 35
        self.ORE_SPEED = 2.5
        self.MINING_RANGE = 120
        self.MINING_ANGLE = 30 # degrees
        self.MINING_RATE = 2 # health per step

        # EXACT spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.mining_target = None
        
        # Initialize state
        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_angle = -90.0 # Pointing up
        
        self.asteroids = []
        for _ in range(self.NUM_ASTEROIDS):
            self.asteroids.append(self._spawn_asteroid(on_screen=True))

        self.particles = []
        self.stars = [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
            for _ in range(self.NUM_STARS)
        ]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Time penalty
        self.steps += 1
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # 1. Update game logic
        self._handle_input(movement)
        self._update_player()
        
        mining_reward = self._handle_mining(space_held)
        reward += mining_reward

        self._update_asteroids()
        
        collection_reward = self._update_particles()
        reward += collection_reward

        # 2. Check for collisions
        if self._check_collisions():
            self.game_over = True
            reward -= 10
            self._create_explosion(self.player_pos)
            # sfx: player_explosion
        
        # 3. Check termination conditions
        terminated = self.game_over
        truncated = False
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Per brief, this is a termination condition
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement):
        # Rotate
        if movement == 3:  # Left
            self.player_angle -= self.PLAYER_ROTATION_SPEED
        if movement == 4:  # Right
            self.player_angle += self.PLAYER_ROTATION_SPEED

        # Thrust
        if movement == 1:  # Up
            rad_angle = math.radians(self.player_angle)
            self.player_vel.x += self.PLAYER_THRUST * math.cos(rad_angle)
            self.player_vel.y += self.PLAYER_THRUST * math.sin(rad_angle)
            
            # Thruster particles
            if self.steps % 2 == 0:
                self._create_thruster_particle()

        # Brake
        drag = self.PLAYER_BRAKE_DRAG if movement == 2 else self.PLAYER_DRAG
        self.player_vel *= drag
    
    def _update_player(self):
        if self.player_vel.length() > self.PLAYER_MAX_SPEED:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)
        
        self.player_pos += self.player_vel
        
        # World wrapping
        self.player_pos.x %= self.WIDTH
        self.player_pos.y %= self.HEIGHT
        
    def _handle_mining(self, space_held):
        self.mining_target = None
        reward = 0
        if not space_held:
            return reward

        # Find closest asteroid in front of the ship
        best_target = None
        min_dist = self.MINING_RANGE + 1

        for asteroid in self.asteroids:
            vec_to_asteroid = asteroid["pos"] - self.player_pos
            dist = vec_to_asteroid.length()

            if 0 < dist < min_dist:
                if vec_to_asteroid.length() > 0:
                    # Normalize angle to be within [-180, 180]
                    angle_diff = (math.degrees(math.atan2(vec_to_asteroid.y, vec_to_asteroid.x)) - self.player_angle + 540) % 360 - 180

                    if abs(angle_diff) < self.MINING_ANGLE:
                        min_dist = dist
                        best_target = asteroid
        
        if best_target:
            self.mining_target = best_target
            best_target["health"] -= self.MINING_RATE
            # sfx: mining_beam_loop
            
            # Spawn ore particles
            if self.np_random.random() < 0.5:
                self._create_ore_particle(best_target["pos"])

            if best_target["health"] <= 0:
                reward += 1 # Bonus for fully mining an asteroid
                # sfx: asteroid_break
                self.asteroids.remove(best_target)
                self.asteroids.append(self._spawn_asteroid(on_screen=False))
                self.mining_target = None
        
        return reward

    def _update_asteroids(self):
        base_speed = 1.0 + (self.steps / 1000) * 0.01
        for asteroid in self.asteroids:
            asteroid["pos"] += asteroid["vel"] * base_speed
            asteroid["pos"].x %= self.WIDTH
            asteroid["pos"].y %= self.HEIGHT
    
    def _update_particles(self):
        reward = 0
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
                continue
            
            if p["type"] == "ore":
                # Magnetism towards player
                vec_to_player = self.player_pos - p["pos"]
                dist_to_player = vec_to_player.length()
                if dist_to_player < self.PLAYER_RADIUS + 50:
                    p["vel"] += vec_to_player.normalize() * 0.5
                    if p["vel"].length() > self.ORE_SPEED * 1.5:
                        p["vel"].scale_to_length(self.ORE_SPEED * 1.5)
                
                if dist_to_player < self.PLAYER_RADIUS:
                    self.score += 1
                    reward += 0.1
                    self.particles.remove(p)
                    # sfx: ore_collect
        return reward
        
    def _check_collisions(self):
        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid["pos"])
            if dist < self.PLAYER_RADIUS + asteroid["radius"]:
                return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw stars
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, self.COLOR_STAR, (x, y, size, size))

        # Draw particles
        for p in self.particles:
            if p["type"] == "ore":
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), p["radius"], self.COLOR_ORE)
            elif p["type"] == "thruster":
                alpha = int(255 * (p["life"] / p["max_life"]))
                color = (*self.COLOR_THRUSTER, alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), p["radius"], color)
            elif p["type"] == "explosion":
                alpha = int(255 * (p["life"] / p["max_life"]))
                color = (*p["color"], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"] * (p["life"] / p["max_life"])), color)

        # Draw mining beam
        if self.mining_target and not self.game_over:
            start_pos = self.player_pos
            end_pos = self.mining_target["pos"]
            beam_poly = self._get_beam_polygon(start_pos, end_pos)
            pygame.gfxdraw.filled_polygon(self.screen, beam_poly, self.COLOR_BEAM)
            pygame.gfxdraw.aapolygon(self.screen, beam_poly, self.COLOR_BEAM)

        # Draw asteroids
        for asteroid in self.asteroids:
            points = [(v + asteroid["pos"]) for v in asteroid["vertices"]]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
            outline_color = tuple(int(c * 0.8) for c in self.COLOR_ASTEROID)
            pygame.gfxdraw.aapolygon(self.screen, points, outline_color)

        # Draw player
        if not self.game_over:
            self._render_player()

    def _render_player(self):
        rad_angle = math.radians(self.player_angle)
        cos_a, sin_a = math.cos(rad_angle), math.sin(rad_angle)

        # Player ship shape (triangle)
        p1 = (self.player_pos.x + 15 * cos_a, self.player_pos.y + 15 * sin_a)
        p2 = (self.player_pos.x - 10 * cos_a + 8 * sin_a, self.player_pos.y - 10 * sin_a - 8 * cos_a)
        p3 = (self.player_pos.x - 10 * cos_a - 8 * sin_a, self.player_pos.y - 10 * sin_a + 8 * cos_a)
        
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

    def _render_ui(self):
        # Ore count
        score_text = self.font.render(f"ORE: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / 100) # Assuming 100 steps/sec
        timer_text = self.font.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        # Game Over / Win message
        if self.game_over:
            msg = "GAME OVER"
            color = self.COLOR_EXPLOSION[0]
            if self.score >= self.WIN_SCORE:
                msg = "MISSION COMPLETE!"
                color = self.COLOR_PLAYER
            
            end_text = self.font.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    # Helper methods for game object creation and effects
    def _spawn_asteroid(self, on_screen=False):
        radius = self.np_random.integers(self.ASTEROID_MIN_RADIUS, self.ASTEROID_MAX_RADIUS + 1)
        if on_screen:
            pos = pygame.Vector2(self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT))
        else: # Spawn off-screen
            edge = self.np_random.integers(0, 4)
            if edge == 0: pos = pygame.Vector2(-radius, self.np_random.integers(0, self.HEIGHT))
            elif edge == 1: pos = pygame.Vector2(self.WIDTH + radius, self.np_random.integers(0, self.HEIGHT))
            elif edge == 2: pos = pygame.Vector2(self.np_random.integers(0, self.WIDTH), -radius)
            else: pos = pygame.Vector2(self.np_random.integers(0, self.WIDTH), self.HEIGHT + radius)

        angle = self.np_random.uniform(0, 2 * math.pi)
        vel = pygame.Vector2(math.cos(angle), math.sin(angle))
        
        num_vertices = self.np_random.integers(7, 12)
        vertices = []
        for i in range(num_vertices):
            a = (i / num_vertices) * 2 * math.pi
            r = radius + self.np_random.uniform(-0.2, 0.2) * radius
            vertices.append(pygame.Vector2(r * math.cos(a), r * math.sin(a)))

        return {
            "pos": pos, "vel": vel, "radius": radius, 
            "health": radius * 3, "vertices": vertices
        }

    def _create_ore_particle(self, pos):
        angle = self.np_random.uniform(0, 2 * math.pi)
        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.np_random.uniform(0.5, 1.5)
        self.particles.append({
            "type": "ore", "pos": pos.copy(), "vel": vel, "life": 150, "radius": 3
        })

    def _create_thruster_particle(self):
        rad_angle = math.radians(self.player_angle + 180) # Opposite direction
        offset = pygame.Vector2(math.cos(rad_angle), math.sin(rad_angle)) * 12
        pos = self.player_pos + offset
        
        vel_angle = rad_angle + self.np_random.uniform(-0.3, 0.3)
        vel = pygame.Vector2(math.cos(vel_angle), math.sin(vel_angle)) * 1.5 + self.player_vel * 0.5
        
        self.particles.append({
            "type": "thruster", "pos": pos, "vel": vel, 
            "life": 20, "max_life": 20, "radius": self.np_random.integers(2, 5)
        })

    def _create_explosion(self, pos):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "type": "explosion", "pos": pos.copy(), "vel": vel,
                "life": 40, "max_life": 40, "radius": self.np_random.integers(2, 6),
                "color": random.choice(self.COLOR_EXPLOSION)
            })

    def _get_beam_polygon(self, start_pos, end_pos):
        vec = end_pos - start_pos
        if vec.length() == 0: return []
        
        perp_vec = vec.rotate(90).normalize()
        
        p1 = start_pos + perp_vec * 3
        p2 = start_pos - perp_vec * 3
        p3 = end_pos - perp_vec * 5
        p4 = end_pos + perp_vec * 5
        
        return [p1, p4, p3, p2]

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    # It will create a window, which is fine for local testing.
    # The environment itself is headless when used by the framework.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Asteroid Miner")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Get player input
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(60) # Limit human play to 60 FPS
        
    env.close()