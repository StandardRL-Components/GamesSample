
# Generated: 2025-08-28T05:50:47.541973
# Source Brief: brief_02755.md
# Brief Index: 2755

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ←→ to move, Space to jump. Hold ↑ while jumping for a boost."
    )

    game_description = (
        "Guide a hopping spaceship through a dynamic asteroid field, collecting valuable coins to score points. Reach the portal at the end to win, but beware: a single collision with an asteroid is fatal!"
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 19, 41)
    COLOR_PLAYER = (61, 191, 255)
    COLOR_PLAYER_GLOW = (180, 230, 255)
    COLOR_ASTEROID = (255, 87, 87)
    COLOR_ASTEROID_GLOW = (255, 150, 150)
    COLOR_COIN = (255, 223, 0)
    COLOR_COIN_GLOW = (255, 240, 150)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_GOAL = (220, 100, 255)

    # Screen dimensions
    WIDTH, HEIGHT = 640, 400

    # Game parameters
    GRAVITY = 0.5
    MAX_FALL_SPEED = 10
    PLAYER_SPEED = 4.0
    PLAYER_JUMP_STRENGTH = 10.0
    PLAYER_JUMP_BOOST = 3.0
    PLAYER_RADIUS = 12
    ASTEROID_RADIUS_MIN, ASTEROID_RADIUS_MAX = 15, 30
    COIN_RADIUS = 8
    GOAL_X = 580
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Persistent state (resets only on __init__)
        self.asteroid_base_speed = 3.0

        # Used for RNG
        self.np_random = None

        self.reset()
        
        # This is a placeholder for a more robust validation if needed
        # self.validate_implementation() 

    def _generate_stars(self, count):
        return [[self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.uniform(0.2, 0.8)] for _ in range(count)]

    def _generate_asteroid(self, x_pos=None):
        if x_pos is None:
            x_pos = self.np_random.integers(self.WIDTH, self.WIDTH * 1.5)
        y_pos = self.np_random.integers(50, self.HEIGHT - 50)
        radius = self.np_random.integers(self.ASTEROID_RADIUS_MIN, self.ASTEROID_RADIUS_MAX + 1)
        
        num_points = self.np_random.integers(7, 12)
        angles = sorted([self.np_random.uniform(0, 2 * math.pi) for _ in range(num_points)])
        shape = [(math.cos(a) * radius * self.np_random.uniform(0.8, 1.2), 
                  math.sin(a) * radius * self.np_random.uniform(0.8, 1.2)) for a in angles]
        
        return {
            "pos": pygame.Vector2(x_pos, y_pos),
            "radius": radius,
            "shape": shape,
            "angle": 0,
            "rot_speed": self.np_random.uniform(-0.02, 0.02)
        }

    def _generate_coin(self):
        return {
            "pos": pygame.Vector2(self.np_random.integers(50, self.GOAL_X - 50), self.np_random.integers(50, self.HEIGHT - 50)),
            "angle": self.np_random.uniform(0, 2 * math.pi)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = pygame.Vector2(50, self.HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = True
        self.space_was_held = False

        self.asteroids = [self._generate_asteroid(x) for x in self.np_random.integers(200, 550, size=3)]
        self.coins = [self._generate_coin() for _ in range(10)]
        self.particles = []
        self.stars = self._generate_stars(150)
        
        self.asteroid_spawn_timer = 250
        self.coin_respawn_timer = 100
        
        self.asteroid_speed = self.asteroid_base_speed

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward += 0.01  # Small survival reward

        # 1. Handle Input
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal movement
        player_input_x = 0
        if movement == 3:  # Left
            player_input_x = -1
            reward -= 0.02 # Small penalty for backtracking
        elif movement == 4:  # Right
            player_input_x = 1
        
        self.player_vel.x = player_input_x * self.PLAYER_SPEED

        # Jump
        jump_pressed = space_held and not self.space_was_held
        if jump_pressed and self.on_ground:
            jump_power = self.PLAYER_JUMP_STRENGTH
            if movement == 1:  # Up
                jump_power += self.PLAYER_JUMP_BOOST
            self.player_vel.y = -jump_power
            self.on_ground = False
            # sfx: player_jump.wav
            self._create_particles(self.player_pos + (0, self.PLAYER_RADIUS), 20, self.COLOR_PLAYER_GLOW, count=15, direction=-math.pi/2, spread=math.pi/4)

        self.space_was_held = space_held

        # 2. Update Player Physics
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(self.player_vel.y, self.MAX_FALL_SPEED)
        self.player_pos += self.player_vel

        # Player boundaries
        if self.player_pos.y >= self.HEIGHT - self.PLAYER_RADIUS:
            self.player_pos.y = self.HEIGHT - self.PLAYER_RADIUS
            self.player_vel.y = 0
            if not self.on_ground:
                # sfx: player_land.wav
                self._create_particles(self.player_pos + (0, self.PLAYER_RADIUS), 5, (200,200,200), count=5, direction=-math.pi/2, spread=math.pi/2)
            self.on_ground = True
        else:
            self.on_ground = False
            
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)

        # 3. Update Game Objects
        # Asteroids
        self.asteroid_spawn_timer -= 1
        if self.asteroid_spawn_timer <= 0:
            self.asteroids.append(self._generate_asteroid())
            self.asteroid_spawn_timer = self.np_random.integers(150, 250)

        for asteroid in self.asteroids:
            asteroid["pos"].x -= self.asteroid_speed
            asteroid["angle"] += asteroid["rot_speed"]
        self.asteroids = [a for a in self.asteroids if a["pos"].x > -a["radius"]]

        # Coins
        self.coin_respawn_timer -= 1
        if self.coin_respawn_timer <= 0 and len(self.coins) < 15:
            self.coins.append(self._generate_coin())
            self.coin_respawn_timer = 100
            
        for coin in self.coins:
            coin["angle"] += 0.05
            
        # Particles
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

        # 4. Handle Collisions & Interactions
        # Player vs Coins
        collected_coins = []
        for coin in self.coins:
            if self.player_pos.distance_to(coin["pos"]) < self.PLAYER_RADIUS + self.COIN_RADIUS:
                collected_coins.append(coin)
                self.score += 1
                reward += 10.0
                # sfx: coin_collect.wav
                self._create_particles(coin["pos"], 15, self.COLOR_COIN_GLOW, count=20)
        self.coins = [c for c in self.coins if c not in collected_coins]

        # Player vs Asteroids
        for asteroid in self.asteroids:
            if self.player_pos.distance_to(asteroid["pos"]) < self.PLAYER_RADIUS + asteroid["radius"] * 0.9: # 0.9 for forgiveness
                terminated = True
                self.game_over = True
                reward -= 50.0
                # sfx: explosion.wav
                self._create_particles(self.player_pos, 30, self.COLOR_ASTEROID, count=50, speed_mult=2.0)
                break

        # 5. Check Win/Loss Conditions
        if self.player_pos.x > self.GOAL_X:
            terminated = True
            self.game_over = True
            reward += 100.0
            self.score += 50 # Bonus for winning
            # sfx: level_complete.wav
            self._create_particles(self.player_pos, 50, self.COLOR_GOAL, count=100, speed_mult=3.0)
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        # 6. Difficulty Scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.asteroid_speed += 0.1
            self.asteroid_base_speed += 0.1 # Persists for next reset

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particles(self, pos, num_particles, color, count=10, speed_mult=1.0, direction=None, spread=2*math.pi):
        for _ in range(count):
            if direction is None:
                angle = self.np_random.uniform(0, 2 * math.pi)
            else:
                angle = direction + self.np_random.uniform(-spread/2, spread/2)
            
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(15, 30)
            radius = self.np_random.uniform(1, 4)
            self.particles.append({"pos": pos.copy(), "vel": vel, "lifespan": lifespan, "color": color, "radius": radius})
    
    def _render_game(self):
        # Stars with parallax
        for x, y, depth in self.stars:
            screen_x = (x - self.player_pos.x * (1 - depth) * 0.1) % self.WIDTH
            pygame.draw.circle(self.screen, (255, 255, 255), (int(screen_x), int(y)), max(1, int(depth * 2)))

        # Goal Line
        for i in range(self.HEIGHT // 20):
            alpha = 100 + 155 * (math.sin(self.steps * 0.1 + i) * 0.5 + 0.5)
            color = (*self.COLOR_GOAL, int(alpha))
            start_pos = (self.GOAL_X, i * 20)
            end_pos = (self.GOAL_X, i * 20 + 10)
            pygame.draw.line(self.screen, color, start_pos, end_pos, 3)

        # Coins
        for coin in self.coins:
            size_mod = (math.sin(coin["angle"]) * 0.5 + 0.5) * 0.8 + 0.2
            w = self.COIN_RADIUS * 2 * size_mod
            h = self.COIN_RADIUS * 2
            if w > 0:
                rect = pygame.Rect(coin["pos"].x - w/2, coin["pos"].y - h/2, w, h)
                pygame.gfxdraw.ellipse(self.screen, int(rect.centerx), int(rect.centery), int(rect.width/2), int(rect.height/2), (*self.COLOR_COIN_GLOW, 100))
                pygame.gfxdraw.filled_ellipse(self.screen, int(rect.centerx), int(rect.centery), int(rect.width/2), int(rect.height/2), self.COLOR_COIN)

        # Asteroids
        for asteroid in self.asteroids:
            points = []
            for px, py in asteroid["shape"]:
                rotated_x = px * math.cos(asteroid["angle"]) - py * math.sin(asteroid["angle"])
                rotated_y = px * math.sin(asteroid["angle"]) + py * math.cos(asteroid["angle"])
                points.append((int(asteroid["pos"].x + rotated_x), int(asteroid["pos"].y + rotated_y)))
            
            if len(points) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID_GLOW)

        # Player
        if not self.game_over:
            # Hopping animation
            bob_offset = math.sin(self.steps * 0.3) * 2 if self.on_ground else 0
            render_pos = (int(self.player_pos.x), int(self.player_pos.y + bob_offset))

            # Jetpack flame
            if not self.on_ground:
                flame_length = self.np_random.uniform(10, 20)
                flame_width = self.np_random.uniform(4, 8)
                flame_points = [
                    (render_pos[0] - flame_width, render_pos[1] + self.PLAYER_RADIUS),
                    (render_pos[0] + flame_width, render_pos[1] + self.PLAYER_RADIUS),
                    (render_pos[0], render_pos[1] + self.PLAYER_RADIUS + flame_length)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, flame_points, self.COLOR_PLAYER_GLOW)
                pygame.gfxdraw.aapolygon(self.screen, flame_points, self.COLOR_PLAYER_GLOW)

            # Body
            pygame.gfxdraw.filled_circle(self.screen, render_pos[0], render_pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, render_pos[0], render_pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER_GLOW)
            # Cockpit
            pygame.gfxdraw.filled_circle(self.screen, render_pos[0], render_pos[1]-2, 5, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.aacircle(self.screen, render_pos[0], render_pos[1]-2, 5, (255,255,255))
            
        # Particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30.0))
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), color)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.small_font.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "LEVEL COMPLETE!" if self.player_pos.x > self.GOAL_X else "GAME OVER"
            end_text = self.font.render(end_text_str, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": (self.player_pos.x, self.player_pos.y),
            "asteroid_count": len(self.asteroids),
            "asteroid_speed": self.asteroid_speed
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
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


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Asteroid Hopper")
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default no-op action
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        elif keys[pygame.K_UP]:
            action[0] = 1

        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            # Wait for 'r' to reset
            pass

    env.close()