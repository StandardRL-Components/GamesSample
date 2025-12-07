
# Generated: 2025-08-28T00:20:04.760619
# Source Brief: brief_03762.md
# Brief Index: 3762

        
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

    user_guide = (
        "Controls: ↑↓←→ to move. Press space to fire your weapon."
    )

    game_description = (
        "Pilot a spaceship in a top-down arena, blasting asteroids to smithereens for high scores before they deplete your shields."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.INITIAL_ASTEROIDS = 20
        self.SHIP_SPEED = 4.0
        self.LASER_SPEED = 8.0
        self.FIRE_COOLDOWN = 10  # frames
        self.INVULNERABILITY_DURATION = 90 # frames

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_SHIP = (100, 255, 100)
        self.COLOR_ASTEROID = (200, 100, 100)
        self.COLOR_LASER = (255, 255, 150)
        self.COLOR_SHIELD = (100, 150, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_STAR = (200, 200, 220)
        
        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.ship_pos = None
        self.ship_lives = None
        self.ship_facing_dir = None
        self.asteroids_destroyed = None
        self.asteroid_speed = None
        self.asteroids = None
        self.lasers = None
        self.particles = None
        self.fire_cooldown_timer = 0
        self.invulnerability_timer = 0
        self.stars = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.ship_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float64)
        self.ship_lives = 3
        self.ship_facing_dir = np.array([0, -1], dtype=np.float64) # Start facing up
        
        self.asteroids_destroyed = 0
        self.asteroid_speed = 1.0
        
        self.asteroids = []
        for _ in range(self.INITIAL_ASTEROIDS):
            self._create_asteroid(size=3)
            
        self.lasers = []
        self.particles = []
        self.fire_cooldown_timer = 0
        self.invulnerability_timer = 0

        if not self.stars:
            for _ in range(150):
                self.stars.append((
                    self.np_random.integers(0, self.WIDTH),
                    self.np_random.integers(0, self.HEIGHT),
                    self.np_random.integers(1, 3)
                ))
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Small penalty for each step to encourage efficiency

        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()
            reward += self._check_collisions()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            if self.win:
                reward += 100
            elif self.ship_lives <= 0:
                reward -= 100
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
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
            self.ship_pos += move_vec * self.SHIP_SPEED
            self.ship_facing_dir = move_vec
        
        self.ship_pos[0] = np.clip(self.ship_pos[0], 0, self.WIDTH)
        self.ship_pos[1] = np.clip(self.ship_pos[1], 0, self.HEIGHT)

        if space_held and self.fire_cooldown_timer <= 0:
            self._fire_laser()
            self.fire_cooldown_timer = self.FIRE_COOLDOWN
    
    def _fire_laser(self):
        # Sound: Laser fire
        start_pos = self.ship_pos + self.ship_facing_dir * 15
        laser = {
            "pos": start_pos,
            "vel": self.ship_facing_dir * self.LASER_SPEED
        }
        self.lasers.append(laser)

    def _update_game_state(self):
        if self.fire_cooldown_timer > 0:
            self.fire_cooldown_timer -= 1
        if self.invulnerability_timer > 0:
            self.invulnerability_timer -= 1

        # Update asteroids
        for asteroid in self.asteroids:
            asteroid["pos"] += asteroid["vel"] * self.asteroid_speed
            asteroid["angle"] += asteroid["rot_speed"]
            # Screen wrap
            asteroid["pos"][0] %= self.WIDTH
            asteroid["pos"][1] %= self.HEIGHT
        
        # Update lasers
        self.lasers = [laser for laser in self.lasers if self._is_on_screen(laser["pos"])]
        for laser in self.lasers:
            laser["pos"] += laser["vel"]

        # Update particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

    def _check_collisions(self):
        reward = 0
        
        # Laser-Asteroid collisions
        lasers_to_remove = []
        asteroids_to_remove = []
        for i, laser in enumerate(self.lasers):
            for j, asteroid in enumerate(self.asteroids):
                if j in asteroids_to_remove: continue
                dist = np.linalg.norm(laser["pos"] - asteroid["pos"])
                if dist < asteroid["radius"]:
                    lasers_to_remove.append(i)
                    asteroids_to_remove.append(j)
                    self._create_explosion(asteroid["pos"], asteroid["size"])
                    # Sound: Explosion
                    
                    reward += 1
                    self.score += 1
                    self.asteroids_destroyed += 1
                    
                    # Spawn smaller asteroids if size > 1
                    if asteroid["size"] > 1:
                        for _ in range(2):
                            self._create_asteroid(size=asteroid["size"] - 1, pos=asteroid["pos"].copy())
                    
                    # Increase difficulty
                    if self.asteroids_destroyed > 0 and self.asteroids_destroyed % 5 == 0:
                        self.asteroid_speed += 0.05

                    break
        
        self.lasers = [l for i, l in enumerate(self.lasers) if i not in lasers_to_remove]
        self.asteroids = [a for j, a in enumerate(self.asteroids) if j not in asteroids_to_remove]

        # Ship-Asteroid collisions
        if self.invulnerability_timer <= 0:
            for asteroid in self.asteroids:
                dist = np.linalg.norm(self.ship_pos - asteroid["pos"])
                if dist < asteroid["radius"] + 10: # 10 is ship radius
                    self.ship_lives -= 1
                    reward -= 1
                    self.invulnerability_timer = self.INVULNERABILITY_DURATION
                    self._create_explosion(self.ship_pos, 2) # Small explosion for hit
                    # Sound: Ship hit
                    break
        
        return reward

    def _check_termination(self):
        if self.ship_lives <= 0:
            return True
        if not self.asteroids:
            self.win = True
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_asteroids()
        self._render_lasers()
        self._render_ship()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.ship_lives,
            "asteroids_remaining": len(self.asteroids),
        }

    # --- Helper methods for state management ---

    def _create_asteroid(self, size, pos=None):
        if pos is None:
            # Spawn away from the center
            while True:
                x = self.np_random.uniform(0, self.WIDTH)
                y = self.np_random.uniform(0, self.HEIGHT)
                if np.linalg.norm(np.array([x, y]) - self.ship_pos) > 100:
                    pos = np.array([x, y], dtype=np.float64)
                    break
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
        
        num_vertices = self.np_random.integers(7, 13)
        radius = size * 10
        base_shape = []
        for i in range(num_vertices):
            a = 2 * math.pi * i / num_vertices
            r = radius * self.np_random.uniform(0.7, 1.3)
            base_shape.append((r * math.cos(a), r * math.sin(a)))

        asteroid = {
            "pos": pos,
            "vel": vel,
            "angle": self.np_random.uniform(0, 2 * math.pi),
            "rot_speed": self.np_random.uniform(-0.03, 0.03),
            "size": size,
            "radius": radius,
            "base_shape": base_shape,
        }
        self.asteroids.append(asteroid)

    def _create_explosion(self, pos, size):
        num_particles = 10 * size
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(15, 30)
            color = random.choice([(255, 255, 255), (255, 255, 100), (255, 150, 50)])
            self.particles.append({"pos": pos.copy(), "vel": vel, "life": life, "color": color})

    def _is_on_screen(self, pos):
        return 0 <= pos[0] <= self.WIDTH and 0 <= pos[1] <= self.HEIGHT

    # --- Helper methods for rendering ---

    def _render_background(self):
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (x, y), size)
    
    def _render_asteroids(self):
        for asteroid in self.asteroids:
            points = []
            for x, y in asteroid["base_shape"]:
                rot_x = x * math.cos(asteroid["angle"]) - y * math.sin(asteroid["angle"])
                rot_y = x * math.sin(asteroid["angle"]) + y * math.cos(asteroid["angle"])
                points.append((rot_x + asteroid["pos"][0], rot_y + asteroid["pos"][1]))
            
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_lasers(self):
        for laser in self.lasers:
            start_pos = laser["pos"]
            end_pos = laser["pos"] - laser["vel"] * 2 # Create a tail
            pygame.draw.line(self.screen, self.COLOR_LASER, start_pos, end_pos, 3)

    def _render_ship(self):
        if self.ship_lives <= 0: return

        # Draw shield if invulnerable
        if self.invulnerability_timer > 0:
            alpha = 100 * (self.invulnerability_timer / self.INVULNERABILITY_DURATION)
            radius = 15 + (1 - (self.invulnerability_timer / self.INVULNERABILITY_DURATION)) * 5
            temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(temp_surf, int(self.ship_pos[0]), int(self.ship_pos[1]), int(radius), (*self.COLOR_SHIELD, int(alpha)))
            pygame.gfxdraw.filled_circle(temp_surf, int(self.ship_pos[0]), int(self.ship_pos[1]), int(radius), (*self.COLOR_SHIELD, int(alpha/2)))
            self.screen.blit(temp_surf, (0,0))
        
        # Draw ship as a triangle
        angle = math.atan2(self.ship_facing_dir[1], self.ship_facing_dir[0]) + math.pi / 2
        s = 10
        p1 = (self.ship_pos[0] + s * math.cos(angle), self.ship_pos[1] + s * math.sin(angle))
        p2 = (self.ship_pos[0] + s * math.cos(angle + 2.5), self.ship_pos[1] + s * math.sin(angle + 2.5))
        p3 = (self.ship_pos[0] + s * math.cos(angle - 2.5), self.ship_pos[1] + s * math.sin(angle - 2.5))
        
        points = [p1, p2, p3]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_SHIP)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_SHIP)

    def _render_particles(self):
        for p in self.particles:
            size = max(1, int(p["life"] / 6))
            pygame.draw.circle(self.screen, p["color"], p["pos"], size)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Asteroids remaining
        ast_text = self.font_small.render(f"ASTEROIDS: {len(self.asteroids)}", True, self.COLOR_TEXT)
        self.screen.blit(ast_text, (self.WIDTH - ast_text.get_width() - 150, 10))

        # Lives
        for i in range(self.ship_lives):
            p1 = (self.WIDTH - 100 + i * 20, 25)
            p2 = (self.WIDTH - 105 + i * 20, 10)
            p3 = (self.WIDTH - 95 + i * 20, 10)
            pygame.draw.polygon(self.screen, self.COLOR_SHIP, [p1, p2, p3])
        
        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_SHIP if self.win else self.COLOR_ASTEROID
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # Example of how to run the environment
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame display for human play
    pygame.display.set_caption(GameEnv.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    print(env.user_guide)

    # Main game loop for human play
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        else:
            # Allow restarting the game
            if keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False
                total_reward = 0

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS

    pygame.quit()
    print(f"Game Over. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")