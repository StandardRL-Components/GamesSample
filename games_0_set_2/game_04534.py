
# Generated: 2025-08-28T02:41:31.494822
# Source Brief: brief_04534.md
# Brief Index: 4534

        
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
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold space to activate the mining beam."
    )

    game_description = (
        "Pilot a mining ship through a dense asteroid field. Collect valuable ore by mining asteroids, "
        "but be careful to avoid collisions which will damage your ship."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.WIN_SCORE = 100
        self.STARTING_LIVES = 3
        self.NUM_ASTEROIDS = 12
        self.NUM_STARS = 150

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (50, 150, 255, 50)
        self.COLOR_THRUSTER = (255, 180, 50)
        self.COLOR_ASTEROID = (120, 110, 100)
        self.COLOR_ORE = (255, 220, 0)
        self.COLOR_BEAM = (100, 255, 100, 100)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_GAMEOVER_OVERLAY = (0, 0, 0, 150)

        # Physics & Gameplay
        self.PLAYER_THRUST = 0.25
        self.PLAYER_TURN_SPEED = 4.5
        self.PLAYER_DRAG = 0.985
        self.PLAYER_BRAKE_DRAG = 0.95
        self.PLAYER_RADIUS = 10
        self.MINING_RANGE = 80
        self.MINING_RATE = 0.5
        self.ASTEROID_BASE_SPEED = 0.5

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.SysFont("monospace", 20, bold=True)
        self.msg_font = pygame.font.SysFont("monospace", 40, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.win = False
        self.np_random = None
        self.player_pos = None
        self.player_vel = None
        self.player_angle = 0
        self.is_thrusting = False
        self.is_mining = False
        self.asteroids = []
        self.particles = []
        self.stars = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.STARTING_LIVES
        self.game_over = False
        self.win = False
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_angle = -90

        self.asteroids = []
        for _ in range(self.NUM_ASTEROIDS):
            self._spawn_asteroid()

        self.particles = []
        self.stars = [
            (
                self.np_random.integers(0, self.WIDTH),
                self.np_random.integers(0, self.HEIGHT),
                self.np_random.uniform(0.5, 1.5),
                self.np_random.integers(50, 150)
            )
            for _ in range(self.NUM_STARS)
        ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        ore_this_step = 0
        self.is_mining = False
        self.is_thrusting = False

        if not self.game_over:
            self._handle_player_input(movement)
            self._update_player()
            self._update_asteroids()
            self._update_particles()
            
            mined, destroyed_asteroid = self._handle_mining(space_held)
            ore_this_step += mined
            
            if self._handle_collisions():
                reward -= 10 # Penalty for a single hit
                # Sound: Ship hit/explosion
                if self.lives <= 0:
                    self.game_over = True
                    reward -= 50 # Terminal penalty for losing all lives

            if ore_this_step > 0:
                reward += ore_this_step * 0.1
                # Sound: Ore collection ping
            else:
                reward -= 0.005 # Small penalty for inactivity

            if destroyed_asteroid:
                reward += 1
                # Sound: Asteroid destroyed success

        self.score += ore_this_step
        self.steps += 1
        
        terminated = False
        if self.score >= self.WIN_SCORE and not self.game_over:
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100
        
        if self.lives <= 0:
            self.game_over = True
            terminated = True

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_input(self, movement):
        # Adapt MultiDiscrete to arcade steering
        if movement == 1: # Up: Thrust
            self.is_thrusting = True
            thrust_vec = pygame.Vector2(1, 0).rotate(self.player_angle) * self.PLAYER_THRUST
            self.player_vel += thrust_vec
        elif movement == 2: # Down: Brake
            self.player_vel *= self.PLAYER_BRAKE_DRAG
        elif movement == 3: # Left: Turn
            self.player_angle -= self.PLAYER_TURN_SPEED
        elif movement == 4: # Right: Turn
            self.player_angle += self.PLAYER_TURN_SPEED

    def _update_player(self):
        self.player_vel *= self.PLAYER_DRAG
        self.player_pos += self.player_vel

        # Toroidal world wrapping
        self.player_pos.x %= self.WIDTH
        self.player_pos.y %= self.HEIGHT
        
        if self.is_thrusting:
            self._create_thrust_particles()

    def _update_asteroids(self):
        current_speed = self.ASTEROID_BASE_SPEED + (self.steps // 2000) * 0.02
        for asteroid in self.asteroids:
            asteroid["pos"] += asteroid["vel"] * current_speed
            asteroid["rot"] = (asteroid["rot"] + asteroid["rot_speed"]) % 360
            asteroid["pos"].x %= self.WIDTH
            asteroid["pos"].y %= self.HEIGHT

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["life"] -= 1
            p["pos"] += p["vel"]

    def _handle_mining(self, space_held):
        if not space_held:
            return 0, False
        
        self.is_mining = True
        mined_total = 0
        destroyed = False

        closest_asteroid = None
        min_dist = float('inf')

        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid["pos"])
            if dist < min_dist:
                min_dist = dist
                closest_asteroid = asteroid
        
        if closest_asteroid and min_dist < self.MINING_RANGE + closest_asteroid["radius"]:
            # Sound: Mining beam active
            mined_amount = min(closest_asteroid["ore"], self.MINING_RATE)
            if mined_amount > 0:
                closest_asteroid["ore"] -= mined_amount
                mined_total += mined_amount
                self._create_ore_particles(closest_asteroid)
                if closest_asteroid["ore"] <= 0:
                    self._spawn_asteroid(from_asteroid=closest_asteroid)
                    destroyed = True
        
        return mined_total, destroyed

    def _handle_collisions(self):
        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid["pos"])
            if dist < self.PLAYER_RADIUS + asteroid["radius"]:
                self.lives -= 1
                self._create_explosion(self.player_pos)
                self.asteroids.remove(asteroid)
                self._spawn_asteroid()
                self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
                self.player_vel = pygame.Vector2(0, 0)
                return True
        return False

    def _spawn_asteroid(self, position=None, from_asteroid=None):
        if from_asteroid is not None:
            self.asteroids.remove(from_asteroid)

        # Ensure new asteroid doesn't spawn on player
        is_safe = False
        while not is_safe:
            edge = self.np_random.integers(4)
            if edge == 0: pos = pygame.Vector2(self.np_random.uniform(-10, self.WIDTH+10), -30)
            elif edge == 1: pos = pygame.Vector2(self.np_random.uniform(-10, self.WIDTH+10), self.HEIGHT + 30)
            elif edge == 2: pos = pygame.Vector2(-30, self.np_random.uniform(-10, self.HEIGHT+10))
            else: pos = pygame.Vector2(self.WIDTH + 30, self.np_random.uniform(-10, self.HEIGHT+10))
            if pos.distance_to(self.player_pos) > 150:
                is_safe = True

        center_dir = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2) - pos
        vel = center_dir.normalize() * self.np_random.uniform(0.5, 1.5)
        
        radius = self.np_random.uniform(15, 30)
        num_vertices = self.np_random.integers(7, 13)
        points = []
        for i in range(num_vertices):
            angle = i * (360 / num_vertices)
            r = radius + self.np_random.uniform(-radius * 0.3, radius * 0.3)
            points.append(pygame.Vector2(r, 0).rotate(angle))
        
        self.asteroids.append({
            "pos": pos,
            "vel": vel,
            "radius": radius,
            "rot": self.np_random.uniform(0, 360),
            "rot_speed": self.np_random.uniform(-1.5, 1.5),
            "ore": radius * 2,
            "points": points
        })

    def _create_explosion(self, pos):
        for _ in range(50):
            vel = pygame.Vector2(1, 0).rotate(self.np_random.uniform(0, 360)) * self.np_random.uniform(1, 6)
            self.particles.append({
                "pos": pos.copy(), "vel": vel,
                "life": self.np_random.integers(20, 40),
                "color": random.choice([(255, 50, 50), (255, 150, 50), (200, 200, 200)])
            })

    def _create_ore_particles(self, asteroid):
        for _ in range(2):
            start_pos = asteroid["pos"] + pygame.Vector2(self.np_random.uniform(-5, 5), self.np_random.uniform(-5, 5))
            vel_to_player = (self.player_pos - start_pos).normalize() * 2.0
            self.particles.append({
                "pos": start_pos, "vel": vel_to_player,
                "life": self.np_random.integers(30, 50), "color": self.COLOR_ORE
            })

    def _create_thrust_particles(self):
        if self.np_random.random() < 0.7:
            pos = self.player_pos - pygame.Vector2(self.PLAYER_RADIUS, 0).rotate(self.player_angle)
            vel = -pygame.Vector2(1, 0).rotate(self.player_angle) * self.np_random.uniform(1, 3) + self.player_vel
            self.particles.append({
                "pos": pos, "vel": vel,
                "life": self.np_random.integers(10, 20), "color": self.COLOR_THRUSTER
            })

    def _get_observation(self):
        self._render_background()
        self._render_asteroids()
        self._render_particles()
        if self.is_mining: self._render_mining_beam()
        self._render_player()
        self._render_ui()
        if self.game_over: self._render_game_over_message()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for x, y, size, brightness in self.stars:
            c = max(0, min(255, brightness))
            pygame.draw.circle(self.screen, (c, c, c), (int(x), int(y)), size)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            points = [p.rotate(asteroid["rot"]) + asteroid["pos"] for p in asteroid["points"]]
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, [(int(p.x), int(p.y)) for p in points], self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, [(int(p.x), int(p.y)) for p in points], self.COLOR_ASTEROID)

    def _render_player(self):
        # Glow
        glow_radius = int(self.PLAYER_RADIUS * 2.5)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (int(self.player_pos.x - glow_radius), int(self.player_pos.y - glow_radius)))

        # Ship body
        p1 = self.player_pos + pygame.Vector2(self.PLAYER_RADIUS, 0).rotate(self.player_angle)
        p2 = self.player_pos + pygame.Vector2(-self.PLAYER_RADIUS, -self.PLAYER_RADIUS * 0.7).rotate(self.player_angle)
        p3 = self.player_pos + pygame.Vector2(-self.PLAYER_RADIUS, self.PLAYER_RADIUS * 0.7).rotate(self.player_angle)
        points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_mining_beam(self):
        closest_asteroid, min_dist = None, float('inf')
        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid["pos"])
            if dist < min_dist:
                min_dist = dist
                closest_asteroid = asteroid
        
        if closest_asteroid and min_dist < self.MINING_RANGE + closest_asteroid["radius"]:
            start_pos = (int(self.player_pos.x), int(self.player_pos.y))
            end_pos = (int(closest_asteroid["pos"].x), int(closest_asteroid["pos"].y))
            
            # Pulsating width
            width = int(3 + math.sin(self.steps * 0.5) * 2)
            if width > 1:
                pygame.draw.line(self.screen, self.COLOR_BEAM, start_pos, end_pos, width)

    def _render_particles(self):
        for p in self.particles:
            size = max(1, int(p["life"] * 0.1))
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"].x), int(p["pos"].y)), size)

    def _render_ui(self):
        # Score
        score_text = self.ui_font.render(f"ORE: {int(self.score)}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.lives):
            pos = pygame.Vector2(self.WIDTH - 30 - i * 25, 25)
            p1 = pos + pygame.Vector2(8, 0).rotate(-90)
            p2 = pos + pygame.Vector2(-8, -6).rotate(-90)
            p3 = pos + pygame.Vector2(-8, 6).rotate(-90)
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1, p2, p3], 2)

    def _render_game_over_message(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill(self.COLOR_GAMEOVER_OVERLAY)
        self.screen.blit(overlay, (0, 0))

        message = "YOU WIN!" if self.win else "GAME OVER"
        text = self.msg_font.render(message, True, self.COLOR_TEXT)
        text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def close(self):
        pygame.quit()

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
    
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no movement, no mining
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            # Wait for 'R' to reset
            waiting_for_reset = True
            while waiting_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        waiting_for_reset = False

        env.clock.tick(30) # Maintain 30 FPS
        
    env.close()