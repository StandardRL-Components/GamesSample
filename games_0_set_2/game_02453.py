
# Generated: 2025-08-28T04:51:58.410949
# Source Brief: brief_02453.md
# Brief Index: 2453

        
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
    Gymnasium environment for a fast-paced arcade space shooter.
    The player must survive a 60-second asteroid onslaught.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to move. Hold Shift to drift. Press Space to fire."
    )

    # User-facing description of the game
    game_description = (
        "Survive a 60-second asteroid field by dodging and destroying "
        "incoming asteroids in this top-down retro space shooter."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_THRUSTER = (255, 150, 50)
        self.COLOR_LASER = (255, 50, 50)
        self.COLOR_ASTEROID = (180, 180, 190)
        self.COLOR_PARTICLE = (255, 180, 80)
        self.COLOR_TEXT = (240, 240, 255)
        self.COLOR_TIMER_WARN = (255, 100, 100)

        # Player settings
        self.PLAYER_SIZE = 12
        self.PLAYER_ACCEL = 0.3
        self.PLAYER_FRICTION_NORMAL = 0.95
        self.PLAYER_FRICTION_DRIFT = 0.99
        self.PLAYER_MAX_SPEED = 6

        # Laser settings
        self.LASER_SPEED = 8
        self.LASER_COOLDOWN = 10  # frames

        # Asteroid settings
        self.ASTEROID_MIN_SPEED = 0.5
        self.ASTEROID_MAX_SPEED = 2.0
        self.ASTEROID_MIN_SIZE = 15
        self.ASTEROID_MAX_SIZE = 40
        self.ASTEROID_SPAWN_RATE_START = 1.0  # per second
        self.ASTEROID_SPAWN_RATE_END = 4.0   # per second

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        self.font_win = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        self.game_won = False
        
        self.player_pos = None
        self.player_vel = None
        self.player_angle = 0.0

        self.asteroids = []
        self.lasers = []
        self.particles = []
        self.stars = []

        self.last_space_held = False
        self.laser_cooldown_timer = 0
        self.asteroid_spawn_timer = 0

        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.time_remaining = self.MAX_STEPS
        self.game_over = False
        self.game_won = False

        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_angle = -90 # Pointing up

        self.asteroids = []
        self.lasers = []
        self.particles = []
        
        self.stars = [
            (
                self.np_random.integers(0, self.WIDTH),
                self.np_random.integers(0, self.HEIGHT),
                self.np_random.uniform(0.5, 1.5),
            )
            for _ in range(150)
        ]

        self.last_space_held = False
        self.laser_cooldown_timer = 0
        self.asteroid_spawn_timer = 0

        # Initial asteroid spawn
        for _ in range(5):
            self._spawn_asteroid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return (
                self._get_observation(),
                0,
                True,
                False,
                self._get_info()
            )

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.1  # Survival reward per frame

        self._handle_input(movement, space_held, shift_held)
        self._update_player(shift_held)
        self._update_lasers()
        self._update_asteroids()
        self._update_particles()
        
        reward += self._handle_collisions()
        self._spawn_asteroids_periodically()

        self.steps += 1
        self.time_remaining -= 1
        
        terminated = self.time_remaining <= 0 or self.game_over
        if terminated:
            if self.time_remaining <= 0 and not self.game_over:
                self.game_won = True
                reward += 100.0 # Victory bonus
            else: # Hit by asteroid
                reward -= 100.0 # Penalty for losing
            self.game_over = True


        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, movement, space_held, shift_held):
        # --- Movement ---
        accel = pygame.math.Vector2(0, 0)
        if movement == 1:  # Up
            accel.y = -self.PLAYER_ACCEL
        elif movement == 2:  # Down
            accel.y = self.PLAYER_ACCEL
        if movement == 3:  # Left
            accel.x = -self.PLAYER_ACCEL
        elif movement == 4:  # Right
            accel.x = self.PLAYER_ACCEL
        
        if accel.length() > 0:
            accel.scale_to_length(self.PLAYER_ACCEL)
            self.player_angle = accel.angle_to(pygame.math.Vector2(1, 0))
        
        self.player_vel += accel

        # --- Firing ---
        if self.laser_cooldown_timer > 0:
            self.laser_cooldown_timer -= 1

        if space_held and not self.last_space_held and self.laser_cooldown_timer == 0:
            self._fire_laser()
            self.laser_cooldown_timer = self.LASER_COOLDOWN
            # sfx: player_shoot.wav
        
        self.last_space_held = space_held

    def _fire_laser(self):
        direction = pygame.math.Vector2(1, 0).rotate(-self.player_angle)
        start_pos = self.player_pos + direction * self.PLAYER_SIZE
        laser_vel = direction * self.LASER_SPEED
        self.lasers.append({"pos": start_pos, "vel": laser_vel})

    def _update_player(self, shift_held):
        friction = self.PLAYER_FRICTION_DRIFT if shift_held else self.PLAYER_FRICTION_NORMAL
        self.player_vel *= friction

        if self.player_vel.length() > self.PLAYER_MAX_SPEED:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)

        self.player_pos += self.player_vel

        # Clamp player to screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _update_lasers(self):
        for laser in self.lasers[:]:
            laser["pos"] += laser["vel"]
            if not (0 < laser["pos"].x < self.WIDTH and 0 < laser["pos"].y < self.HEIGHT):
                self.lasers.remove(laser)

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid["pos"] += asteroid["vel"]
            # Screen wrapping
            if asteroid["pos"].x < -asteroid["size"]: asteroid["pos"].x = self.WIDTH + asteroid["size"]
            if asteroid["pos"].x > self.WIDTH + asteroid["size"]: asteroid["pos"].x = -asteroid["size"]
            if asteroid["pos"].y < -asteroid["size"]: asteroid["pos"].y = self.HEIGHT + asteroid["size"]
            if asteroid["pos"].y > self.HEIGHT + asteroid["size"]: asteroid["pos"].y = -asteroid["size"]

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _spawn_asteroid(self):
        size = self.np_random.uniform(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE)
        edge = self.np_random.integers(0, 4)
        if edge == 0:  # Top
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), -size)
        elif edge == 1:  # Bottom
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + size)
        elif edge == 2:  # Left
            pos = pygame.math.Vector2(-size, self.np_random.uniform(0, self.HEIGHT))
        else:  # Right
            pos = pygame.math.Vector2(self.WIDTH + size, self.np_random.uniform(0, self.HEIGHT))

        # Difficulty scaling for speed
        progress = self.steps / self.MAX_STEPS
        speed_multiplier = 1.0 + 0.5 * progress
        speed = self.np_random.uniform(self.ASTEROID_MIN_SPEED, self.ASTEROID_MAX_SPEED) * speed_multiplier
        
        angle = self.np_random.uniform(0, 360)
        vel = pygame.math.Vector2(speed, 0).rotate(angle)
        
        # Generate procedural shape
        num_vertices = self.np_random.integers(7, 12)
        shape = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            radius = size * self.np_random.uniform(0.7, 1.1)
            shape.append((radius * math.cos(angle), radius * math.sin(angle)))

        self.asteroids.append({"pos": pos, "vel": vel, "size": size, "shape": shape})

    def _spawn_asteroids_periodically(self):
        progress = self.steps / self.MAX_STEPS
        current_spawn_rate = self.ASTEROID_SPAWN_RATE_START + (self.ASTEROID_SPAWN_RATE_END - self.ASTEROID_SPAWN_RATE_START) * progress
        spawn_interval = self.FPS / current_spawn_rate

        self.asteroid_spawn_timer += 1
        if self.asteroid_spawn_timer >= spawn_interval:
            self.asteroid_spawn_timer = 0
            self._spawn_asteroid()

    def _handle_collisions(self):
        reward = 0
        # Lasers vs Asteroids
        for laser in self.lasers[:]:
            for asteroid in self.asteroids[:]:
                if laser["pos"].distance_to(asteroid["pos"]) < asteroid["size"]:
                    self._create_explosion(asteroid["pos"], asteroid["size"])
                    self.asteroids.remove(asteroid)
                    if laser in self.lasers: self.lasers.remove(laser)
                    self.score += 10
                    reward += 1.0
                    # sfx: explosion.wav
                    break
        
        # Player vs Asteroids
        for asteroid in self.asteroids:
            if self.player_pos.distance_to(asteroid["pos"]) < asteroid["size"] + self.PLAYER_SIZE * 0.5:
                self.game_over = True
                self._create_explosion(self.player_pos, self.PLAYER_SIZE * 2)
                # sfx: player_death.wav
                break
        
        return reward

    def _create_explosion(self, pos, size):
        num_particles = int(size)
        for _ in range(num_particles):
            vel = pygame.math.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3))
            life = self.np_random.integers(15, 40)
            self.particles.append({"pos": pos.copy(), "vel": vel, "life": life})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_end_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, (200, 200, 220), (x, y), size)

    def _render_game(self):
        # Render asteroids
        for asteroid in self.asteroids:
            points = [(p[0] + asteroid["pos"].x, p[1] + asteroid["pos"].y) for p in asteroid["shape"]]
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

        # Render lasers
        for laser in self.lasers:
            start_pos = (int(laser["pos"].x), int(laser["pos"].y))
            end_pos = (int(laser["pos"].x - laser["vel"].x), int(laser["pos"].y - laser["vel"].y))
            pygame.draw.aaline(self.screen, self.COLOR_LASER, start_pos, end_pos, 2)
        
        # Render player
        if not (self.game_over and not self.game_won):
            self._render_player()

        # Render particles
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / 40))
            color = (*self.COLOR_PARTICLE, alpha)
            size = max(1, p["life"] / 10)
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p["pos"].x - size), int(p["pos"].y - size)))

    def _render_player(self):
        angle_rad = math.radians(-self.player_angle)
        
        # Main ship body
        p1 = (
            self.player_pos.x + self.PLAYER_SIZE * math.cos(angle_rad),
            self.player_pos.y + self.PLAYER_SIZE * math.sin(angle_rad)
        )
        p2 = (
            self.player_pos.x + self.PLAYER_SIZE * 0.7 * math.cos(angle_rad + 2.5),
            self.player_pos.y + self.PLAYER_SIZE * 0.7 * math.sin(angle_rad + 2.5)
        )
        p3 = (
            self.player_pos.x + self.PLAYER_SIZE * 0.7 * math.cos(angle_rad - 2.5),
            self.player_pos.y + self.PLAYER_SIZE * 0.7 * math.sin(angle_rad - 2.5)
        )
        ship_points = [p1, p2, p3]
        pygame.gfxdraw.aapolygon(self.screen, ship_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, ship_points, self.COLOR_PLAYER)

        # Thruster flame when moving
        if self.player_vel.length() > 0.5:
            flame_len = self.PLAYER_SIZE * (0.8 + 0.4 * math.sin(self.steps * 0.8))
            f1 = (
                self.player_pos.x - self.PLAYER_SIZE * 0.5 * math.cos(angle_rad),
                self.player_pos.y - self.PLAYER_SIZE * 0.5 * math.sin(angle_rad)
            )
            f2 = (
                self.player_pos.x - flame_len * math.cos(angle_rad),
                self.player_pos.y - flame_len * math.sin(angle_rad)
            )
            pygame.draw.aaline(self.screen, self.COLOR_THRUSTER, f1, f2, 4)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_sec = self.time_remaining / self.FPS
        timer_color = self.COLOR_TIMER_WARN if time_sec < 10 else self.COLOR_TEXT
        timer_text = self.font_ui.render(f"TIME: {max(0, time_sec):.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

    def _render_end_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        if self.game_won:
            end_text = self.font_win.render("VICTORY!", True, self.COLOR_PLAYER)
        else:
            end_text = self.font_game_over.render("GAME OVER", True, self.COLOR_LASER)
        
        text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "asteroids_on_screen": len(self.asteroids),
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Asteroid Survival")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()
                total_reward = 0

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            
            # Reset loop for playing again
            waiting_for_reset = True
            while waiting_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        print("Resetting game...")
                        obs, info = env.reset()
                        total_reward = 0
                        waiting_for_reset = False

        clock.tick(env.FPS)

    env.close()