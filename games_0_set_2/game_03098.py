
# Generated: 2025-08-27T22:21:17.218509
# Source Brief: brief_03098.md
# Brief Index: 3098

        
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
    """
    An arcade-style top-down spaceship shooter environment. The player must
    destroy all asteroids while avoiding collisions.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the ship. Press space to fire projectiles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship in a top-down arena, blasting asteroids to survive and achieve a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    INITIAL_ASTEROIDS = 20
    INITIAL_LIVES = 3

    # Colors
    COLOR_BG = (15, 15, 30)
    COLOR_STAR = (200, 200, 220)
    COLOR_SHIP = (50, 255, 50)
    COLOR_SHIP_INVINCIBLE = (150, 255, 150)
    COLOR_PROJECTILE = (255, 255, 100)
    COLOR_ASTEROID = (180, 180, 180)
    COLOR_TEXT = (255, 255, 255)
    COLOR_EXPLOSION = [(255, 50, 50), (255, 150, 50), (255, 255, 50)]

    # Game Physics
    SHIP_ACCELERATION = 0.4
    SHIP_MAX_SPEED = 5.0
    SHIP_FRICTION = 0.97
    SHIP_RADIUS = 10
    PROJECTILE_SPEED = 8.0
    PROJECTILE_COOLDOWN = 6  # frames
    ASTEROID_SPEED = 2.0
    ASTEROID_SIZES = {"small": 10, "medium": 20, "large": 30}
    INVINCIBILITY_DURATION = 90 # frames (3 seconds at 30fps)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)

        self.stars = []
        self.asteroid_shapes = {}
        self._generate_procedural_assets()

        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.ship_pos = np.zeros(2)
        self.ship_vel = np.zeros(2)
        self.ship_angle = 0.0
        self.last_move_dir = np.array([0.0, -1.0])
        self.projectiles = []
        self.asteroids = []
        self.particles = []
        self.fire_cooldown = 0
        self.invincibility_timer = 0
        
        self.reset()
        self.validate_implementation()

    def _generate_procedural_assets(self):
        # Generate a static starfield
        for _ in range(150):
            self.stars.append(
                (
                    random.randint(0, self.SCREEN_WIDTH),
                    random.randint(0, self.SCREEN_HEIGHT),
                    random.choice([1, 2]),
                )
            )
        # Generate base shapes for asteroids
        for size_name, radius in self.ASTEROID_SIZES.items():
            points = []
            num_vertices = random.randint(7, 11)
            for i in range(num_vertices):
                angle = 2 * math.pi * i / num_vertices
                dist = random.uniform(radius * 0.7, radius * 1.0)
                points.append((dist * math.cos(angle), dist * math.sin(angle)))
            self.asteroid_shapes[size_name] = points

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False

        self.ship_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        self.ship_vel = np.array([0.0, 0.0], dtype=float)
        self.ship_angle = -90.0
        self.last_move_dir = np.array([0.0, -1.0])

        self.projectiles = []
        self.asteroids = []
        self.particles = []
        self.fire_cooldown = 0
        self.invincibility_timer = self.INVINCIBILITY_DURATION # Start with invincibility

        # Spawn asteroids, ensuring they are not too close to the player's start
        for _ in range(self.INITIAL_ASTEROIDS):
            while True:
                pos = self.np_random.uniform(low=0, high=[self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
                if np.linalg.norm(pos - self.ship_pos) > 100:
                    break
            
            size_name = self.np_random.choice(list(self.ASTEROID_SIZES.keys()))
            radius = self.ASTEROID_SIZES[size_name]
            angle = self.np_random.uniform(0, 360)
            vel = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))]) * self.ASTEROID_SPEED
            
            self.asteroids.append({
                "pos": pos,
                "vel": vel,
                "radius": radius,
                "size_name": size_name,
                "angle": self.np_random.uniform(0, 360),
                "rot_speed": self.np_random.uniform(-2.0, 2.0)
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for each step to encourage efficiency
        terminated = False

        if not self.game_over:
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

            self._handle_input(movement, space_held)
            self._update_ship()
            self._update_projectiles()
            self._update_asteroids()
            
            collision_reward = self._handle_collisions()
            reward += collision_reward

        self._update_particles()
        
        self.steps += 1
        if self.lives <= 0 or len(self.asteroids) == 0 or self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            if self.lives <= 0:
                reward -= 100 # Terminal penalty for losing
            elif len(self.asteroids) == 0:
                reward += 100 # Terminal reward for winning

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held):
        # Movement
        move_dir = np.array([0.0, 0.0])
        if movement == 1: move_dir[1] = -1 # Up
        elif movement == 2: move_dir[1] = 1 # Down
        elif movement == 3: move_dir[0] = -1 # Left
        elif movement == 4: move_dir[0] = 1 # Right
        
        if np.any(move_dir):
            self.ship_vel += move_dir * self.SHIP_ACCELERATION
            self.last_move_dir = move_dir
        
        # Firing
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1

        if space_held and self.fire_cooldown <= 0:
            # SFX: Laser fire
            projectile_vel = self.last_move_dir * self.PROJECTILE_SPEED
            self.projectiles.append({
                "pos": self.ship_pos.copy(),
                "vel": projectile_vel
            })
            self.fire_cooldown = self.PROJECTILE_COOLDOWN

    def _update_ship(self):
        # Apply friction
        self.ship_vel *= self.SHIP_FRICTION
        # Clamp speed
        speed = np.linalg.norm(self.ship_vel)
        if speed > self.SHIP_MAX_SPEED:
            self.ship_vel = (self.ship_vel / speed) * self.SHIP_MAX_SPEED
        
        # Update position
        self.ship_pos += self.ship_vel

        # Clamp to screen bounds
        self.ship_pos[0] = np.clip(self.ship_pos[0], self.SHIP_RADIUS, self.SCREEN_WIDTH - self.SHIP_RADIUS)
        self.ship_pos[1] = np.clip(self.ship_pos[1], self.SHIP_RADIUS, self.SCREEN_HEIGHT - self.SHIP_RADIUS)

        # Update angle for rendering
        self.ship_angle = math.degrees(math.atan2(self.last_move_dir[1], self.last_move_dir[0])) + 90
        
        # Update invincibility
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1

    def _update_projectiles(self):
        projectiles_to_keep = []
        for p in self.projectiles:
            p["pos"] += p["vel"]
            if 0 < p["pos"][0] < self.SCREEN_WIDTH and 0 < p["pos"][1] < self.SCREEN_HEIGHT:
                projectiles_to_keep.append(p)
        self.projectiles = projectiles_to_keep

    def _update_asteroids(self):
        for a in self.asteroids:
            a["pos"] += a["vel"]
            a["angle"] = (a["angle"] + a["rot_speed"]) % 360
            # Screen wrapping
            if a["pos"][0] < -a["radius"]: a["pos"][0] = self.SCREEN_WIDTH + a["radius"]
            if a["pos"][0] > self.SCREEN_WIDTH + a["radius"]: a["pos"][0] = -a["radius"]
            if a["pos"][1] < -a["radius"]: a["pos"][1] = self.SCREEN_HEIGHT + a["radius"]
            if a["pos"][1] > self.SCREEN_HEIGHT + a["radius"]: a["pos"][1] = -a["radius"]

    def _update_particles(self):
        particles_to_keep = []
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] > 0:
                particles_to_keep.append(p)
        self.particles = particles_to_keep

    def _handle_collisions(self):
        reward = 0
        
        # Projectile-Asteroid collisions
        projectiles_to_remove = set()
        asteroids_to_remove = set()
        for i, p in enumerate(self.projectiles):
            for j, a in enumerate(self.asteroids):
                if np.linalg.norm(p["pos"] - a["pos"]) < a["radius"]:
                    projectiles_to_remove.add(i)
                    asteroids_to_remove.add(j)
                    reward += 10
                    self.score += 1
                    self._create_explosion(a["pos"], a["radius"])
                    # SFX: Asteroid explosion
                    break # Projectile can only hit one asteroid
        
        if projectiles_to_remove or asteroids_to_remove:
            self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]
            self.asteroids = [a for i, a in enumerate(self.asteroids) if i not in asteroids_to_remove]

        # Ship-Asteroid collisions
        if self.invincibility_timer <= 0:
            for a in self.asteroids:
                if np.linalg.norm(self.ship_pos - a["pos"]) < self.SHIP_RADIUS + a["radius"]:
                    self.lives -= 1
                    reward -= 10
                    self.invincibility_timer = self.INVINCIBILITY_DURATION
                    self._create_explosion(self.ship_pos, self.SHIP_RADIUS * 2)
                    # SFX: Ship hit/explosion
                    if self.lives > 0:
                        self.ship_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
                        self.ship_vel = np.array([0.0, 0.0], dtype=float)
                    break
        return reward

    def _create_explosion(self, position, size):
        num_particles = int(size * 1.5)
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifespan = self.np_random.integers(15, 40)
            self.particles.append({
                "pos": position.copy(),
                "vel": vel,
                "lifespan": lifespan,
                "max_lifespan": lifespan,
                "color": self.np_random.choice(self.COLOR_EXPLOSION)
            })

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

        # Draw asteroids
        for a in self.asteroids:
            self._draw_rotated_polygon(a["pos"], self.asteroid_shapes[a["size_name"]], a["angle"], self.COLOR_ASTEROID)

        # Draw projectiles
        for p in self.projectiles:
            start_pos = (int(p["pos"][0]), int(p["pos"][1]))
            end_pos = (int(p["pos"][0] - p["vel"][0]*0.5), int(p["pos"][1] - p["vel"][1]*0.5))
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, start_pos, end_pos, 2)
            pygame.gfxdraw.aacircle(self.screen, int(p["pos"][0]), int(p["pos"][1]), 2, self.COLOR_PROJECTILE)

        # Draw ship
        if self.lives > 0:
            is_invincible = self.invincibility_timer > 0
            is_visible = not (is_invincible and (self.invincibility_timer // 4) % 2 == 0)
            if is_visible:
                ship_color = self.COLOR_SHIP_INVINCIBLE if is_invincible else self.COLOR_SHIP
                ship_points = [(-8, 10), (0, -12), (8, 10)]
                self._draw_rotated_polygon(self.ship_pos, ship_points, self.ship_angle, ship_color)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            color = p["color"]
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), 2, (*color, alpha))

    def _draw_rotated_polygon(self, center_pos, points, angle, color):
        angle_rad = math.radians(angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        rotated_points = []
        for x, y in points:
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            rotated_points.append((int(center_pos[0] + x_rot), int(center_pos[1] + y_rot)))
        
        if len(rotated_points) > 1:
            pygame.gfxdraw.aapolygon(self.screen, rotated_points, color)
            pygame.gfxdraw.filled_polygon(self.screen, rotated_points, color)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_ui.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 150, 10))
        for i in range(self.lives):
             self._draw_rotated_polygon(
                (self.SCREEN_WIDTH - 70 + i * 25, 20),
                [(-6, 7), (0, -9), (6, 7)], -90, self.COLOR_SHIP
            )

        # Game Over message
        if self.game_over:
            msg = "YOU WIN!" if len(self.asteroids) == 0 else "GAME OVER"
            color = self.COLOR_SHIP if len(self.asteroids) == 0 else self.COLOR_EXPLOSION[0]
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "asteroids_left": len(self.asteroids)
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        assert info['asteroids_left'] == self.INITIAL_ASTEROIDS
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to run and display the game
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Override screen for display
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroid Arena")

    terminated = False
    running = True
    while running:
        # Human input mapping
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        if terminated:
            if keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False
            action = env.action_space.sample() * 0 # No-op

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Rendering to the display window
        display_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        env.screen.blit(display_surface, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        env.clock.tick(30) # Run at 30 FPS

    env.close()