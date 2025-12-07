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

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to move your ship. Hold Space to mine nearby asteroids. Avoid colliding with asteroids and enemies!"
    )

    # User-facing description of the game
    game_description = (
        "Pilot a mining ship through a dangerous asteroid field. Collect 50 ore to win, but watch out for hostile ships and collisions!"
    )

    # Frames auto-advance for smooth, real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WORLD_WIDTH = 2000
    WORLD_HEIGHT = 2000
    FPS = 30
    
    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (60, 180, 255)
    COLOR_PLAYER_THRUSTER = (255, 200, 100)
    COLOR_ENEMY_1 = (255, 50, 50)
    COLOR_ENEMY_2 = (255, 100, 50)
    COLOR_ENEMY_3 = (255, 150, 50)
    COLOR_ASTEROID = (120, 100, 90)
    COLOR_ASTEROID_GOLD = (200, 150, 50)
    COLOR_ORE = (255, 220, 0)
    COLOR_PROJECTILE = (255, 100, 0)
    COLOR_BEAM = (100, 255, 255, 150) # RGBA
    COLOR_WHITE = (240, 240, 240)
    
    # Game parameters
    PLAYER_SPEED = 6
    PLAYER_DRAG = 0.92
    PLAYER_LIVES = 3
    PLAYER_INVINCIBILITY_FRAMES = 60 # 2 seconds
    PLAYER_MINING_RANGE = 100

    WIN_SCORE = 50
    MAX_STEPS = 5000

    INITIAL_ASTEROIDS = 25
    ASTEROID_MIN_ORE = 5
    ASTEROID_MAX_ORE = 15
    ASTEROID_RESPAWN_TIME_BASE = 10 * FPS # 10 seconds
    
    INITIAL_ENEMIES = 1

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.render_mode = render_mode
        self.np_random = None

        # This will be populated in reset()
        self.player = {}
        self.asteroids = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.stars = []
        
        self.steps = 0
        self.score = 0
        self.game_over_message = ""

        # self.validate_implementation() # Commented out for submission, but useful for local testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)
        if seed is not None:
             self.np_random = np.random.default_rng(seed)


        self.player = {
            "pos": pygame.math.Vector2(self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2),
            "vel": pygame.math.Vector2(0, 0),
            "angle": -90,
            "lives": self.PLAYER_LIVES,
            "invincible_timer": 0,
            "is_mining": False,
            "mining_target": None
        }

        self.steps = 0
        self.score = 0
        self.game_over_message = ""
        
        self.asteroids = [self._create_asteroid() for _ in range(self.INITIAL_ASTEROIDS)]
        self.enemies = [self._create_enemy(1) for _ in range(self.INITIAL_ENEMIES)]
        self.projectiles = []
        self.particles = []
        self.stars = [
            (
                self.np_random.integers(0, self.WORLD_WIDTH), 
                self.np_random.integers(0, self.WORLD_HEIGHT), 
                self.np_random.integers(1, 4)
            ) for _ in range(200)
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.01  # Small penalty for each step
        terminated = False
        truncated = False

        # --- Player Logic ---
        self._handle_player_input(movement)
        self.player["vel"] *= self.PLAYER_DRAG
        self.player["pos"] += self.player["vel"]
        self._wrap_around_world(self.player["pos"])
        
        if self.player["invincible_timer"] > 0:
            self.player["invincible_timer"] -= 1

        # Mining logic
        self.player["is_mining"] = space_held
        self.player["mining_target"] = None
        if self.player["is_mining"]:
            ore_collected_this_step, reward_from_mining = self._handle_mining()
            self.score += ore_collected_this_step
            reward += reward_from_mining

        # --- Update Entities ---
        self._update_enemies()
        self._update_projectiles()
        self._update_asteroids()
        self._update_particles()
        
        # --- Collision Detection ---
        reward += self._handle_collisions()

        # --- Difficulty Scaling ---
        self._update_difficulty()
        
        self.steps += 1

        # --- Termination Check ---
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward += 100
            self.game_over_message = "TARGET YIELD REACHED!"
        elif self.player["lives"] <= 0:
            terminated = True
            reward -= 100
            self.game_over_message = "SHIP DESTROYED"
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            terminated = True # Gymnasium standard is to set both
            self.game_over_message = "MISSION TIMEOUT"

        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    # --- Entity Creation ---
    def _create_asteroid(self, pos=None):
        if pos is None:
            pos = pygame.math.Vector2(
                self.np_random.integers(0, self.WORLD_WIDTH),
                self.np_random.integers(0, self.WORLD_HEIGHT)
            )
        size = self.np_random.integers(15, 40)
        num_points = self.np_random.integers(7, 12)
        shape = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            dist = size + self.np_random.uniform(-size * 0.2, size * 0.2)
            shape.append(pygame.math.Vector2(dist * math.cos(angle), dist * math.sin(angle)))
        
        return {
            "pos": pos,
            "size": size,
            "shape": shape,
            "angle": self.np_random.uniform(0, 360),
            "rot_speed": self.np_random.uniform(-0.5, 0.5),
            "ore": self.np_random.integers(self.ASTEROID_MIN_ORE, self.ASTEROID_MAX_ORE + 1),
            "depleted_timer": 0
        }

    def _create_enemy(self, enemy_type):
        # Spawn away from player
        angle = self.np_random.uniform(0, 2 * math.pi)
        dist = self.np_random.uniform(self.SCREEN_WIDTH / 2, self.SCREEN_WIDTH)
        pos = self.player["pos"] + pygame.math.Vector2(dist * math.cos(angle), dist * math.sin(angle))
        self._wrap_around_world(pos)

        base_speed = 1.0 + self.score / 100.0
        base_fire_rate = 1.0 - self.score / 100.0
        
        if enemy_type == 1:
            return {"pos": pos, "type": 1, "speed": base_speed * 1.5, "fire_cooldown": self.np_random.integers(60, 121), "fire_rate": max(30, 90 * base_fire_rate)}
        elif enemy_type == 2:
            return {"pos": pos, "type": 2, "speed": base_speed * 2.0, "fire_cooldown": self.np_random.integers(30, 91), "fire_rate": max(25, 75 * base_fire_rate)}
        else: # type 3
            return {"pos": pos, "type": 3, "speed": base_speed * 2.5, "fire_cooldown": self.np_random.integers(15, 61), "fire_rate": max(20, 60 * base_fire_rate)}

    def _create_particle(self, pos, p_type, **kwargs):
        if p_type == "thruster":
            vel = pygame.math.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
            self.particles.append({"pos": pos, "vel": vel, "type": "thruster", "timer": 10, "color": self.COLOR_PLAYER_THRUSTER})
        elif p_type == "ore":
            vel = (self.player["pos"] - pos).normalize() * 3
            self.particles.append({"pos": pos, "vel": vel, "type": "ore", "timer": 90, "color": self.COLOR_ORE})
        elif p_type == "explosion":
            num_shards = kwargs.get("num", 30)
            base_color = kwargs.get("color", (255,150,50))
            for _ in range(num_shards):
                angle = self.np_random.uniform(0, 2*math.pi)
                speed = self.np_random.uniform(1, 5)
                vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
                timer = self.np_random.integers(20, 41)
                color = (max(0, min(255, base_color[0] + self.np_random.integers(-20, 21))),
                         max(0, min(255, base_color[1] + self.np_random.integers(-20, 21))),
                         max(0, min(255, base_color[2] + self.np_random.integers(-20, 21))))
                self.particles.append({"pos": pos.copy(), "vel": vel, "type": "explosion", "timer": timer, "color": color})
    
    # --- Update Logic ---
    def _handle_player_input(self, movement):
        accel = pygame.math.Vector2(0, 0)
        if movement == 1: accel.y = -1 # Up
        if movement == 2: accel.y = 1  # Down
        if movement == 3: accel.x = -1 # Left
        if movement == 4: accel.x = 1  # Right
        
        if accel.length() > 0:
            accel.scale_to_length(self.PLAYER_SPEED * 0.15)
            self.player["vel"] += accel
            if self.player["vel"].length() > self.PLAYER_SPEED:
                self.player["vel"].scale_to_length(self.PLAYER_SPEED)
            
            # Thruster particles
            if self.steps % 2 == 0:
                thruster_pos = self.player["pos"] - self.player["vel"].normalize() * 10
                self._create_particle(thruster_pos, "thruster")

    def _handle_mining(self):
        closest_asteroid = None
        min_dist_sq = self.PLAYER_MINING_RANGE ** 2
        for asteroid in self.asteroids:
            if asteroid["depleted_timer"] > 0: continue
            dist_sq = self.player["pos"].distance_squared_to(asteroid["pos"])
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_asteroid = asteroid
        
        ore_collected = 0
        reward = 0
        if closest_asteroid:
            self.player["mining_target"] = closest_asteroid
            # Mine one ore per 5 frames of continuous mining
            if self.steps % 5 == 0 and closest_asteroid["ore"] > 0:
                closest_asteroid["ore"] -= 1
                self._create_particle(closest_asteroid["pos"], "ore")
                # sound: "ore_collect"
                ore_collected = 1
                reward = 0.1
        return ore_collected, reward

    def _update_enemies(self):
        for enemy in self.enemies:
            # Movement AI
            direction = (self.player["pos"] - enemy["pos"])
            if direction.length() > 0:
                enemy["pos"] += direction.normalize() * enemy["speed"]
            self._wrap_around_world(enemy["pos"])
            
            # Firing AI
            enemy["fire_cooldown"] -= 1
            if enemy["fire_cooldown"] <= 0 and direction.length() < self.SCREEN_WIDTH / 1.5:
                enemy["fire_cooldown"] = enemy["fire_rate"]
                vel = direction.normalize() * 5
                self.projectiles.append({"pos": enemy["pos"].copy(), "vel": vel, "owner": "enemy"})
                # sound: "enemy_shoot"

    def _update_projectiles(self):
        self.projectiles[:] = [p for p in self.projectiles if self.screen.get_rect().collidepoint(self._world_to_screen(p["pos"]))]
        for p in self.projectiles:
            p["pos"] += p["vel"]

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid["angle"] += asteroid["rot_speed"]
            if asteroid["ore"] <= 0 and asteroid["depleted_timer"] == 0:
                asteroid["depleted_timer"] = self.ASTEROID_RESPAWN_TIME_BASE - (self.score * 0.1 * self.FPS)
            
            if asteroid["depleted_timer"] > 0:
                asteroid["depleted_timer"] -= 1
                if asteroid["depleted_timer"] <= 0:
                    # Respawn asteroid somewhere else
                    new_pos = pygame.math.Vector2(
                        self.np_random.integers(0, self.WORLD_WIDTH),
                        self.np_random.integers(0, self.WORLD_HEIGHT)
                    )
                    new_asteroid = self._create_asteroid(new_pos)
                    asteroid.update(new_asteroid)

    def _update_particles(self):
        for p in self.particles:
            p["timer"] -= 1
            p["pos"] += p["vel"]
            if p["type"] == "ore": # Ore homes in on player
                direction = self.player["pos"] - p["pos"]
                if direction.length() > 5:
                    p["vel"] = p["vel"] * 0.95 + direction.normalize() * 0.5
            else: # Other particles have drag
                 p["vel"] *= 0.98
        self.particles[:] = [p for p in self.particles if p["timer"] > 0]

    def _update_difficulty(self):
        # Introduce new enemy types
        if 15 <= self.score < 30 and len([e for e in self.enemies if e["type"] == 2]) == 0:
            self.enemies.append(self._create_enemy(2))
        if self.score >= 30 and len([e for e in self.enemies if e["type"] == 3]) == 0:
            self.enemies.append(self._create_enemy(3))

    def _handle_collisions(self):
        reward = 0
        if self.player["invincible_timer"] > 0:
            return reward

        # Player vs Asteroids
        for asteroid in self.asteroids:
            if asteroid["depleted_timer"] > 0: continue
            if self.player["pos"].distance_to(asteroid["pos"]) < asteroid["size"] + 5:
                self._damage_player(asteroid["pos"])
                reward -= 10 # Penalty for collision
                return reward # Only one collision per frame

        # Player vs Enemies
        for enemy in self.enemies:
            if self.player["pos"].distance_to(enemy["pos"]) < 20:
                self._damage_player(enemy["pos"])
                reward -= 10
                return reward

        # Player vs Projectiles
        for proj in self.projectiles:
            if self.player["pos"].distance_to(proj["pos"]) < 12:
                self._damage_player(proj["pos"])
                self.projectiles.remove(proj)
                reward -= 5
                return reward
        
        return reward

    def _damage_player(self, collision_pos):
        if self.player["invincible_timer"] <= 0:
            self.player["lives"] -= 1
            self.player["invincible_timer"] = self.PLAYER_INVINCIBILITY_FRAMES
            self._create_particle(collision_pos, "explosion", num=50, color=(255, 80, 0))
            # sound: "player_hit" / "explosion"

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.game_over_message:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _world_to_screen(self, world_pos):
        screen_x = world_pos.x - self.player["pos"].x + self.SCREEN_WIDTH / 2
        screen_y = world_pos.y - self.player["pos"].y + self.SCREEN_HEIGHT / 2
        return pygame.math.Vector2(screen_x, screen_y)

    def _wrap_around_world(self, pos):
        pos.x %= self.WORLD_WIDTH
        pos.y %= self.WORLD_HEIGHT

    def _render_game(self):
        # Draw stars
        for x, y, size in self.stars:
            screen_pos = self._world_to_screen(pygame.math.Vector2(x, y))
            brightness = 50 * size
            pygame.draw.circle(self.screen, (brightness, brightness, brightness), (int(screen_pos.x), int(screen_pos.y)), size // 2)

        # Draw mining beam
        if self.player["is_mining"] and self.player["mining_target"]:
            start_pos = self._world_to_screen(self.player["pos"])
            end_pos = self._world_to_screen(self.player["mining_target"]["pos"])
            pygame.draw.aaline(self.screen, self.COLOR_BEAM[:3], start_pos, end_pos, int(1 + 2 * math.sin(self.steps * 0.5)))
        
        # Draw particles
        for p in self.particles:
            screen_pos = self._world_to_screen(p["pos"])
            size = max(1, int(p["timer"] / 5)) if p["type"] == "explosion" else (3 if p["type"] == "ore" else 2)
            alpha = int(255 * (p["timer"] / (40 if p["type"] == "explosion" else 10)))
            if alpha > 0 and size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, p["color"] + (alpha,), (size, size), size)
                self.screen.blit(s, (int(screen_pos.x - size), int(screen_pos.y - size)))

        # Draw asteroids
        for asteroid in self.asteroids:
            if asteroid["depleted_timer"] > 0: continue
            screen_pos = self._world_to_screen(asteroid["pos"])
            if self.screen.get_rect().colliderect(pygame.Rect(screen_pos.x - asteroid["size"], screen_pos.y - asteroid["size"], asteroid["size"]*2, asteroid["size"]*2)):
                rotated_shape = [p.rotate(asteroid["angle"]) + screen_pos for p in asteroid["shape"]]
                if len(rotated_shape) > 2:
                    color = self.COLOR_ASTEROID_GOLD if asteroid["ore"] > self.ASTEROID_MAX_ORE * 0.7 else self.COLOR_ASTEROID
                    pygame.gfxdraw.aapolygon(self.screen, rotated_shape, color)
                    pygame.gfxdraw.filled_polygon(self.screen, rotated_shape, color)

        # Draw enemies
        for enemy in self.enemies:
            screen_pos = self._world_to_screen(enemy["pos"])
            color = self.COLOR_ENEMY_1 if enemy["type"] == 1 else (self.COLOR_ENEMY_2 if enemy["type"] == 2 else self.COLOR_ENEMY_3)
            size = 8 if enemy["type"] == 1 else (10 if enemy["type"] == 2 else 12)
            points = [
                (screen_pos.x, screen_pos.y - size),
                (screen_pos.x - size*0.8, screen_pos.y + size*0.8),
                (screen_pos.x + size*0.8, screen_pos.y + size*0.8),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Draw projectiles
        for p in self.projectiles:
            screen_pos = self._world_to_screen(p["pos"])
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, (int(screen_pos.x), int(screen_pos.y)), 4)
            
        # Draw player
        if self.player["lives"] > 0:
            if self.player["invincible_timer"] == 0 or self.steps % 10 < 5:
                screen_pos = self._world_to_screen(self.player["pos"])
                size = 12
                p1 = screen_pos + pygame.math.Vector2(size, 0)
                p2 = screen_pos + pygame.math.Vector2(-size/2, -size/2)
                p3 = screen_pos + pygame.math.Vector2(-size/4, 0)
                p4 = screen_pos + pygame.math.Vector2(-size/2, size/2)
                
                angle_to_mouse = (self.player["vel"] * 50).angle_to(pygame.math.Vector2(1,0)) if self.player["vel"].length() > 0.1 else self.player["angle"]
                self.player["angle"] = angle_to_mouse

                # FIX: Replaced 'rotate_around' with the equivalent vector math operation.
                # To rotate point P around pivot C: (P - C).rotate(angle) + C
                points = [(p - screen_pos).rotate(-self.player["angle"]) + screen_pos for p in [p1,p2,p3,p4]]
                
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"ORE: {self.score}/{self.WIN_SCORE}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.player["lives"]):
            points = [
                (self.SCREEN_WIDTH - 20 - i*25, 15),
                (self.SCREEN_WIDTH - 20 - i*25 - 5, 25),
                (self.SCREEN_WIDTH - 20 - i*25 + 5, 25),
            ]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points, 2)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        text = self.font_large.render(self.game_over_message, True, self.COLOR_WHITE)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player["lives"],
            "player_pos": tuple(self.player["pos"]),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        # We need to reset first to populate the state for rendering
        _, _ = self.reset(seed=123)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=123)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # This part requires a display. If you run this in a headless environment,
    # it will fail unless you have a virtual display setup.
    try:
        pygame.display.set_caption("Space Miner")
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        clock = pygame.time.Clock()
        
        running = True
        while running:
            movement = 0 # no-op
            space_held = 0
            shift_held = 0

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()

            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Transpose observation back for Pygame display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
                # Wait for a moment before auto-resetting or quitting
                pygame.time.wait(3000)
                obs, info = env.reset()

            clock.tick(env.FPS)
    except pygame.error as e:
        print(f"Pygame display error: {e}")
        print("This can happen in a headless environment. The environment itself is likely working.")
        print("Running a short headless test...")
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.reset()
        print("Headless test completed without crashing.")


    env.close()