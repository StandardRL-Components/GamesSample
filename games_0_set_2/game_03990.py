
# Generated: 2025-08-28T01:04:32.678586
# Source Brief: brief_03990.md
# Brief Index: 3990

        
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
        "Controls: Arrow keys to move. Hold Space to fire. Survive the asteroid waves!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down arcade space shooter. Survive three waves of increasingly "
        "difficult asteroids by dodging and shooting."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # --- Game Constants ---
        self.PLAYER_SPEED = 5
        self.PLAYER_RADIUS = 12
        self.PROJECTILE_SPEED = 10
        self.PROJECTILE_RADIUS = 3
        self.FIRE_RATE = 8  # Cooldown in frames
        self.INVULNERABILITY_DURATION = 90  # Frames (3 seconds at 30fps)
        self.MAX_STEPS = 2000

        self.ASTEROID_SIZES = {
            "small": {"radius": 10, "health": 1, "score": 10, "reward": 0.1},
            "medium": {"radius": 20, "health": 2, "score": 20, "reward": 0.2},
            "large": {"radius": 35, "health": 3, "score": 30, "reward": 0.3},
        }
        self.BASE_ASTEROID_SPEED = 1.0

        # --- Colors ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_THRUSTER = (255, 180, 0)
        self.COLOR_PROJECTILE = (255, 255, 0)
        self.COLOR_ASTEROID = (160, 160, 160)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_EXPLOSION = [(255, 60, 0), (255, 150, 0), (255, 220, 0)]

        # --- State Variables ---
        self.player_pos = None
        self.player_health = None
        self.player_invulnerability_timer = None
        self.fire_cooldown = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.victory = None
        self.current_wave = None
        self.asteroids = []
        self.projectiles = []
        self.particles = []
        self.stars = []
        self.np_random = None
        self.step_action = self.action_space.sample() # To store last action
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.player_pos = [self.screen_width / 2, self.screen_height / 2]
        self.player_health = 3
        self.player_invulnerability_timer = self.INVULNERABILITY_DURATION
        self.fire_cooldown = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.victory = False
        self.current_wave = 0

        self.asteroids = []
        self.projectiles = []
        self.particles = []

        self._generate_stars(200)
        self._start_next_wave()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.clock.tick(30)
        self.steps += 1
        self.step_action = action # Store for rendering
        reward = -0.01  # Small penalty for time passing

        if not self.game_over and not self.victory:
            self._update_player(action)
            self._update_projectiles()
            self._update_asteroids()
            self._update_particles()
            reward += self._handle_collisions()

            if not self.asteroids and self.current_wave > 0:
                reward += 1.0
                self._start_next_wave()
        
        terminated = self._check_termination()
        
        if self.player_health <= 0 and not self.game_over:
            self.game_over = True
            terminated = True
            reward += -10.0
            self._create_explosion(self.player_pos, 80, self.COLOR_EXPLOSION)
            # SFX: player_death_explosion.wav

        if self.victory and not terminated: # First frame of victory
            terminated = True
            reward += 10.0
            
        if self.steps >= self.MAX_STEPS:
            terminated = True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_player(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Timers
        if self.fire_cooldown > 0: self.fire_cooldown -= 1
        if self.player_invulnerability_timer > 0: self.player_invulnerability_timer -= 1

        # Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED

        # Clamp position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.screen_width - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.screen_height - self.PLAYER_RADIUS)

        # Firing
        if space_held and self.fire_cooldown == 0:
            projectile = {
                "pos": list(self.player_pos),
                "vel": [0, -self.PROJECTILE_SPEED] # Always fire "up"
            }
            self.projectiles.append(projectile)
            self.fire_cooldown = self.FIRE_RATE
            # SFX: player_shoot.wav

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            if not (0 < p["pos"][0] < self.screen_width and 0 < p["pos"][1] < self.screen_height):
                self.projectiles.remove(p)

    def _update_asteroids(self):
        for a in self.asteroids[:]:
            a["pos"][0] += a["vel"][0]
            a["pos"][1] += a["vel"][1]
            a["angle"] += a["rot_speed"]
            
            # Despawn if off-screen by a margin
            margin = a["radius"] * 2
            if not (-margin < a["pos"][0] < self.screen_width + margin and -margin < a["pos"][1] < self.screen_height + margin):
                 self.asteroids.remove(a)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        
        # Projectiles vs Asteroids
        for p in self.projectiles[:]:
            for a in self.asteroids[:]:
                dist = math.hypot(p["pos"][0] - a["pos"][0], p["pos"][1] - a["pos"][1])
                if dist < a["radius"] + self.PROJECTILE_RADIUS:
                    a["health"] -= 1
                    if p in self.projectiles: self.projectiles.remove(p)
                    
                    if a["health"] <= 0:
                        reward += self.ASTEROID_SIZES[a["size"]]["reward"]
                        self.score += self.ASTEROID_SIZES[a["size"]]["score"]
                        self._create_explosion(a["pos"], int(a["radius"]), self.COLOR_EXPLOSION)
                        if a in self.asteroids: self.asteroids.remove(a)
                        # SFX: asteroid_explosion.wav
                    else:
                        self._create_explosion(p["pos"], 5, [self.COLOR_ASTEROID])
                        # SFX: asteroid_hit.wav
                    break 

        # Player vs Asteroids
        if self.player_invulnerability_timer == 0:
            for a in self.asteroids[:]:
                dist = math.hypot(self.player_pos[0] - a["pos"][0], self.player_pos[1] - a["pos"][1])
                if dist < a["radius"] + self.PLAYER_RADIUS:
                    self.player_health -= 1
                    self.player_invulnerability_timer = self.INVULNERABILITY_DURATION
                    self._create_explosion(a["pos"], int(a["radius"]), self.COLOR_EXPLOSION)
                    if a in self.asteroids: self.asteroids.remove(a)
                    # SFX: player_hit.wav
                    break
        return reward

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > 3:
            self.victory = True
            return

        wave_configs = {1: (20, 0.0), 2: (30, 0.2), 3: (40, 0.4)}
        count, speed_mod = wave_configs[self.current_wave]

        for _ in range(count):
            self._spawn_asteroid(self.BASE_ASTEROID_SPEED + speed_mod)
            
    def _spawn_asteroid(self, speed):
        size = self.np_random.choice(list(self.ASTEROID_SIZES.keys()))
        radius = self.ASTEROID_SIZES[size]["radius"]
        
        # Spawn on an edge
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = [self.np_random.uniform(0, self.screen_width), -radius]
        elif edge == 1: # Bottom
            pos = [self.np_random.uniform(0, self.screen_width), self.screen_height + radius]
        elif edge == 2: # Left
            pos = [-radius, self.np_random.uniform(0, self.screen_height)]
        else: # Right
            pos = [self.screen_width + radius, self.np_random.uniform(0, self.screen_height)]

        angle = math.atan2(self.screen_height/2 - pos[1], self.screen_width/2 - pos[0])
        angle += self.np_random.uniform(-math.pi / 4, math.pi / 4) # Add variance
        vel = [math.cos(angle) * speed, math.sin(angle) * speed]

        # Generate irregular shape
        num_vertices = self.np_random.integers(7, 12)
        shape_points = []
        for i in range(num_vertices):
            angle_offset = 2 * math.pi * i / num_vertices
            dist = self.np_random.uniform(radius * 0.7, radius)
            shape_points.append((dist * math.cos(angle_offset), dist * math.sin(angle_offset)))

        self.asteroids.append({
            "pos": pos,
            "vel": vel,
            "radius": radius,
            "health": self.ASTEROID_SIZES[size]["health"],
            "size": size,
            "shape": shape_points,
            "angle": 0,
            "rot_speed": self.np_random.uniform(-0.05, 0.05)
        })

    def _check_termination(self):
        return self.player_health <= 0 or self.victory or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_asteroids()
        self._render_projectiles()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _generate_stars(self, count):
        self.stars = []
        for _ in range(count):
            self.stars.append(
                (
                    self.np_random.integers(0, self.screen_width),
                    self.np_random.integers(0, self.screen_height),
                    self.np_random.uniform(0.5, 1.5) # for size/brightness
                )
            )

    def _render_stars(self):
        for x, y, size in self.stars:
            brightness = int(100 * size)
            color = (brightness, brightness, brightness)
            pygame.draw.rect(self.screen, color, (x, y, int(size), int(size)))

    def _render_player(self):
        if self.player_health <= 0: return

        # Blinking effect when invulnerable
        if self.player_invulnerability_timer > 0 and self.steps % 10 < 5:
            return

        x, y = self.player_pos
        points = [(x, y - 15), (x - 10, y + 10), (x + 10, y + 10)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Thruster - movement == 1 is 'up'
        if self.step_action[0] == 1:
            flicker = self.np_random.uniform(0.8, 1.2)
            thruster_points = [
                (x - 6, y + 12), (x + 6, y + 12), (x, y + 12 + 12 * flicker)
            ]
            pygame.gfxdraw.aapolygon(self.screen, thruster_points, self.COLOR_THRUSTER)
            pygame.gfxdraw.filled_polygon(self.screen, thruster_points, self.COLOR_THRUSTER)

    def _render_asteroids(self):
        for a in self.asteroids:
            x, y = a["pos"]
            angle = a["angle"]
            
            rotated_points = []
            for px, py in a["shape"]:
                rx = px * math.cos(angle) - py * math.sin(angle)
                ry = px * math.sin(angle) + py * math.cos(angle)
                rotated_points.append((x + rx, y + ry))

            if len(rotated_points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_ASTEROID)
    
    def _render_projectiles(self):
        for p in self.projectiles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p["lifespan"] / p["max_lifespan"]))
            color = p["color"] + (alpha,)
            size = int(p["size"] * (p["lifespan"] / p["max_lifespan"]))
            if size > 0:
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (int(p["pos"][0] - size), int(p["pos"][1] - size)))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_str = f"WAVE: {self.current_wave}" if not self.victory else "VICTORY!"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_UI)
        self.screen.blit(wave_text, (self.screen_width - wave_text.get_width() - 10, 10))

        # Health
        health_text = self.font_small.render(f"HEALTH: {max(0, self.player_health)}", True, self.COLOR_UI)
        self.screen.blit(health_text, (self.screen_width / 2 - health_text.get_width() / 2, self.screen_height - 30))

        # Game Over / Victory Text
        if self.game_over:
            end_text = self.font_large.render("GAME OVER", True, (255, 50, 50))
            self.screen.blit(end_text, (self.screen_width / 2 - end_text.get_width() / 2, self.screen_height / 2 - end_text.get_height() / 2))
        elif self.victory:
            end_text = self.font_large.render("VICTORY", True, (50, 255, 50))
            self.screen.blit(end_text, (self.screen_width / 2 - end_text.get_width() / 2, self.screen_height / 2 - end_text.get_height() / 2))
            
    def _create_explosion(self, pos, num_particles, colors):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifespan": lifespan,
                "max_lifespan": lifespan,
                "color": self.np_random.choice(colors, p=[0.1, 0.4, 0.5] if len(colors) == 3 else None),
                "size": self.np_random.uniform(2, 5)
            })

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # To run the game with manual controls
    pygame.display.set_caption("Asteroid Waves")
    render_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    total_reward = 0
    
    # Store last action to pass to step
    action = env.action_space.sample() 
    action.fill(0) # Start with no-op

    running = True
    while running:
        # --- Human Controls ---
        movement = 0 # none
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = np.array([movement, space, shift])
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False
                total_reward = 0
        
        if not done:
            # --- Gym Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # --- Render to screen ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(surf, (0, 0))
            pygame.display.flip()
        else:
            # Keep displaying final screen
            env.clock.tick(30)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    pygame.quit()