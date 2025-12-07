
# Generated: 2025-08-28T00:25:19.063123
# Source Brief: brief_03782.md
# Brief Index: 3782

        
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
        "Controls: Use arrow keys to move your ship. Hold space to activate your mining laser."
    )
    game_description = (
        "Mine asteroids for minerals in an isometric 2D space while dodging enemies to collect 500 minerals."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    WORLD_WIDTH, WORLD_HEIGHT = 1200, 1200
    
    # Colors
    COLOR_BG = (16, 16, 32)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_CORE = (255, 255, 255)
    COLOR_ENEMY = (255, 64, 64)
    COLOR_ENEMY_CORE = (255, 150, 150)
    COLOR_ASTEROID = (139, 69, 19)
    COLOR_ASTEROID_OUTLINE = (90, 45, 12)
    COLOR_MINERAL = (255, 215, 0)
    COLOR_LASER = (0, 255, 255)
    COLOR_TEXT = (255, 255, 255)

    # Player
    PLAYER_ACCELERATION = 0.25
    PLAYER_FRICTION = 0.96
    PLAYER_MAX_SPEED = 4.0
    PLAYER_LIVES = 3
    PLAYER_INVINCIBILITY_STEPS = 90 # 3 seconds at 30fps

    # Mining
    MINING_RANGE = 120
    MINING_RATE = 1
    WIN_SCORE = 500

    # Asteroids
    NUM_ASTEROIDS = 10
    ASTEROID_MIN_MINERALS = 50
    ASTEROID_MAX_MINERALS = 150
    ASTEROID_RESPAWN_TIME = 300 # 10 seconds

    # Enemies
    NUM_ENEMIES = 5
    ENEMY_BASE_SPEED = 0.02 # Radians per step
    ENEMY_PATROL_RADIUS = 150

    # Episode
    MAX_STEPS = 1500

    # Isometric projection
    ISO_TILE_WIDTH_HALF = 1.0
    ISO_TILE_HEIGHT_HALF = 0.5
    
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
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.player = {}
        self.asteroids = []
        self.enemies = []
        self.particles = []
        self.stars = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.player = {
            "pos": np.array([self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2], dtype=np.float32),
            "vel": np.array([0.0, 0.0], dtype=np.float32),
            "lives": self.PLAYER_LIVES,
            "invincibility_timer": 0,
            "size": 12
        }

        self.asteroids = [self._create_asteroid() for _ in range(self.NUM_ASTEROIDS)]
        self.enemies = [self._create_enemy() for _ in range(self.NUM_ENEMIES)]
        self.particles = []
        self.stars = [
            (self.np_random.integers(0, self.WORLD_WIDTH), self.np_random.integers(0, self.WORLD_HEIGHT), self.np_random.uniform(0.2, 0.8))
            for _ in range(200)
        ]
        
        self.is_mining = False
        self.screen_shake_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        self._update_player(movement)
        mining_reward = self._handle_mining(space_held)
        self._update_enemies()
        self._update_asteroids()
        self._update_particles()
        self._handle_collisions()

        if self.is_mining:
            reward += mining_reward
        else:
            reward -= 0.02

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            if self.game_won:
                reward += 100
            else: # Lost all lives or max steps
                reward -= 100
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_asteroid(self, position=None):
        if position is None:
            pos = np.array([
                self.np_random.uniform(0, self.WORLD_WIDTH),
                self.np_random.uniform(0, self.WORLD_HEIGHT)
            ], dtype=np.float32)
        else:
            pos = np.array(position, dtype=np.float32)

        num_vertices = self.np_random.integers(7, 12)
        radius = self.np_random.uniform(20, 40)
        base_vertices = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            r = radius + self.np_random.uniform(-radius*0.2, radius*0.2)
            base_vertices.append(np.array([math.cos(angle) * r, math.sin(angle) * r]))
            
        return {
            "pos": pos,
            "angle": self.np_random.uniform(0, 2 * math.pi),
            "rot_speed": self.np_random.uniform(-0.01, 0.01),
            "base_vertices": base_vertices,
            "vertices": base_vertices.copy(),
            "minerals": self.np_random.integers(self.ASTEROID_MIN_MINERALS, self.ASTEROID_MAX_MINERALS + 1),
            "max_minerals": self.ASTEROID_MAX_MINERALS,
            "respawn_timer": 0,
            "size": radius
        }

    def _create_enemy(self):
        return {
            "center": np.array([self.np_random.uniform(0, self.WORLD_WIDTH), self.np_random.uniform(0, self.WORLD_HEIGHT)], dtype=np.float32),
            "radius": self.np_random.uniform(self.ENEMY_PATROL_RADIUS * 0.5, self.ENEMY_PATROL_RADIUS * 1.5),
            "angle": self.np_random.uniform(0, 2 * math.pi),
            "angular_vel_dir": self.np_random.choice([-1, 1]),
            "pos": np.array([0.0, 0.0], dtype=np.float32),
            "size": 10
        }

    def _update_player(self, movement):
        if self.player["invincibility_timer"] > 0:
            self.player["invincibility_timer"] -= 1

        accel = np.array([0.0, 0.0], dtype=np.float32)
        if movement == 1: accel[1] = -self.PLAYER_ACCELERATION # Up
        elif movement == 2: accel[1] = self.PLAYER_ACCELERATION # Down
        elif movement == 3: accel[0] = -self.PLAYER_ACCELERATION # Left
        elif movement == 4: accel[0] = self.PLAYER_ACCELERATION # Right
        
        self.player["vel"] += accel
        speed = np.linalg.norm(self.player["vel"])
        if speed > self.PLAYER_MAX_SPEED:
            self.player["vel"] = self.player["vel"] * (self.PLAYER_MAX_SPEED / speed)
        
        self.player["vel"] *= self.PLAYER_FRICTION
        self.player["pos"] += self.player["vel"]

        # World wrapping
        self.player["pos"][0] %= self.WORLD_WIDTH
        self.player["pos"][1] %= self.WORLD_HEIGHT

        # Engine particles
        if np.linalg.norm(accel) > 0:
            for _ in range(2):
                self._create_particle(
                    pos=self.player["pos"].copy(),
                    vel=-accel * self.np_random.uniform(5, 10),
                    lifespan=15,
                    start_size=4,
                    end_size=0,
                    start_color=self.COLOR_PLAYER,
                    end_color=self.COLOR_BG
                )

    def _handle_mining(self, space_held):
        self.is_mining = False
        if not space_held:
            return 0

        closest_asteroid = None
        min_dist_sq = self.MINING_RANGE ** 2

        for asteroid in self.asteroids:
            if asteroid["minerals"] <= 0:
                continue
            
            dist_sq = self._get_wrapped_dist_sq(self.player["pos"], asteroid["pos"])
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_asteroid = asteroid
        
        reward = 0
        if closest_asteroid:
            self.is_mining = True
            mined_amount = min(self.MINING_RATE, closest_asteroid["minerals"])
            closest_asteroid["minerals"] -= mined_amount
            self.score += mined_amount
            reward += mined_amount * 0.1 # Reward for minerals
            # // Sound: mining_laser.wav (loop)

            # Mineral particles
            for _ in range(int(mined_amount)):
                self._create_particle(
                    pos=closest_asteroid["pos"].copy(),
                    vel=(self.player["pos"] - closest_asteroid["pos"]) * 0.05 + self.np_random.uniform(-0.5, 0.5, 2),
                    lifespan=20,
                    start_size=3,
                    end_size=1,
                    start_color=self.COLOR_MINERAL,
                    end_color=self.COLOR_MINERAL
                )

            if closest_asteroid["minerals"] <= 0:
                closest_asteroid["respawn_timer"] = self.ASTEROID_RESPAWN_TIME
                reward += 1.0 # Reward for depleting asteroid
                # // Sound: asteroid_depleted.wav

        return reward

    def _update_enemies(self):
        speed_multiplier = 1.0 + (self.score // 100) * 0.05
        for enemy in self.enemies:
            enemy["angle"] += self.ENEMY_BASE_SPEED * speed_multiplier * enemy["angular_vel_dir"]
            offset = np.array([math.cos(enemy["angle"]), math.sin(enemy["angle"])]) * enemy["radius"]
            enemy["pos"] = enemy["center"] + offset

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            if asteroid["respawn_timer"] > 0:
                asteroid["respawn_timer"] -= 1
                if asteroid["respawn_timer"] == 0:
                    new_asteroid = self._create_asteroid()
                    asteroid.update(new_asteroid)
            else:
                asteroid["angle"] += asteroid["rot_speed"]
                rot_matrix = np.array([
                    [math.cos(asteroid["angle"]), -math.sin(asteroid["angle"])],
                    [math.sin(asteroid["angle"]), math.cos(asteroid["angle"])]
                ])
                asteroid["vertices"] = [v @ rot_matrix for v in asteroid["base_vertices"]]

    def _handle_collisions(self):
        if self.player["invincibility_timer"] > 0:
            return

        player_pos = self.player["pos"]
        player_size = self.player["size"]

        for enemy in self.enemies:
            dist_sq = self._get_wrapped_dist_sq(player_pos, enemy["pos"])
            if dist_sq < (player_size + enemy["size"]) ** 2:
                self._player_hit()
                return

    def _player_hit(self):
        self.player["lives"] -= 1
        self.player["invincibility_timer"] = self.PLAYER_INVINCIBILITY_STEPS
        self.screen_shake_timer = 20
        # // Sound: player_explosion.wav
        
        # Explosion particles
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self._create_particle(
                pos=self.player["pos"].copy(),
                vel=vel,
                lifespan=self.np_random.integers(20, 40),
                start_size=self.np_random.uniform(2, 5),
                end_size=0,
                start_color=self.np_random.choice([self.COLOR_ENEMY, self.COLOR_MINERAL, self.COLOR_ENEMY_CORE]),
                end_color=self.COLOR_BG
            )

        if self.player["lives"] > 0:
            # Respawn in a random safe spot
            self.player["pos"] = np.array([self.np_random.uniform(0, self.WORLD_WIDTH), self.np_random.uniform(0, self.WORLD_HEIGHT)], dtype=np.float32)
            self.player["vel"] = np.array([0.0, 0.0], dtype=np.float32)

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.game_won = True
            return True
        if self.player["lives"] <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        render_offset = np.array([0, 0])
        if self.screen_shake_timer > 0:
            self.screen_shake_timer -= 1
            render_offset = self.np_random.uniform(-5, 5, 2)

        self._render_game(render_offset)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, offset):
        # Render stars
        for x, y, z in self.stars:
            sx, sy = self._world_to_screen((x, y), parallax_scale=z)
            pygame.draw.circle(self.screen, (int(z*100), int(z*100), int(z*120)), (sx + offset[0], sy + offset[1]), 1)

        # Collect and sort all objects by their isometric y-coordinate for correct layering
        renderables = []
        for asteroid in self.asteroids:
            if asteroid["respawn_timer"] == 0:
                sx, sy = self._world_to_screen(asteroid["pos"])
                renderables.append(("asteroid", asteroid, sy))
        
        for enemy in self.enemies:
            sx, sy = self._world_to_screen(enemy["pos"])
            renderables.append(("enemy", enemy, sy))

        player_sx, player_sy = self._world_to_screen(self.player["pos"])
        renderables.append(("player", self.player, player_sy))
        
        renderables.sort(key=lambda item: item[2])
        
        # Render objects in sorted order
        for r_type, obj, _ in renderables:
            if r_type == "asteroid": self._render_asteroid(obj, offset)
            elif r_type == "enemy": self._render_enemy(obj, offset)
            elif r_type == "player": self._render_player(offset)

        # Render mining laser on top
        if self.is_mining:
            self._render_laser(offset)
            
        # Render particles
        self._render_particles(offset)

    def _render_player(self, offset):
        sx, sy = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
        size = self.player["size"]
        
        # Create a flashing effect for invincibility
        if self.player["invincibility_timer"] > 0 and (self.steps // 3) % 2 == 0:
            return

        ship_points = [
            (sx, sy - size),
            (sx - size * 0.7, sy + size * 0.7),
            (sx, sy + size * 0.3),
            (sx + size * 0.7, sy + size * 0.7)
        ]
        ship_points = [(p[0] + offset[0], p[1] + offset[1]) for p in ship_points]

        pygame.gfxdraw.filled_polygon(self.screen, ship_points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, ship_points, self.COLOR_PLAYER_CORE)
        core_pos = (int(sx + offset[0]), int(sy + offset[1]))
        pygame.draw.circle(self.screen, self.COLOR_PLAYER_CORE, core_pos, max(0, int(size/3)))

    def _render_asteroid(self, asteroid, offset):
        sx, sy = self._world_to_screen(asteroid["pos"])
        
        screen_vertices = [(v[0] + sx + offset[0], v[1] + sy + offset[1]) for v in asteroid["vertices"]]
        
        if len(screen_vertices) > 2:
            pygame.gfxdraw.filled_polygon(self.screen, screen_vertices, self.COLOR_ASTEROID)
            pygame.gfxdraw.aapolygon(self.screen, screen_vertices, self.COLOR_ASTEROID_OUTLINE)

        # Mineral health bar
        if asteroid["minerals"] < asteroid["max_minerals"]:
            bar_width = 30
            bar_height = 4
            bar_x = sx - bar_width / 2 + offset[0]
            bar_y = sy - asteroid['size'] - 10 + offset[1]
            fill_ratio = asteroid["minerals"] / asteroid["max_minerals"]
            pygame.draw.rect(self.screen, (60, 60, 60), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_MINERAL, (bar_x, bar_y, bar_width * fill_ratio, bar_height))

    def _render_enemy(self, enemy, offset):
        sx, sy = self._world_to_screen(enemy["pos"])
        size = enemy["size"]
        
        points = [
            (sx + offset[0], sy - size + offset[1]),
            (sx - size + offset[0], sy + offset[1]),
            (sx + offset[0], sy + size + offset[1]),
            (sx + size + offset[0], sy + offset[1]),
        ]
        
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY_CORE)

    def _render_laser(self, offset):
        closest_asteroid = None
        min_dist_sq = self.MINING_RANGE ** 2
        for asteroid in self.asteroids:
            if asteroid["minerals"] <= 0: continue
            dist_sq = self._get_wrapped_dist_sq(self.player["pos"], asteroid["pos"])
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_asteroid = asteroid
        
        if closest_asteroid:
            start_pos = (self.SCREEN_WIDTH // 2 + offset[0], self.SCREEN_HEIGHT // 2 + offset[1])
            end_pos = self._world_to_screen(closest_asteroid["pos"])
            end_pos = (end_pos[0] + offset[0], end_pos[1] + offset[1])
            
            pulse = (math.sin(self.steps * 0.5) + 1) / 2
            width = int(2 + pulse * 3)
            alpha = int(150 + pulse * 105)
            
            # Use a temporary surface for transparency
            line_surf = self.screen.copy()
            line_surf.set_colorkey((0,0,0))
            line_surf.set_alpha(alpha)
            pygame.draw.line(line_surf, self.COLOR_LASER, start_pos, end_pos, width)
            pygame.draw.circle(line_surf, self.COLOR_LASER, start_pos, int(width*1.5))
            pygame.draw.circle(line_surf, self.COLOR_LASER, end_pos, int(width*1.5))
            self.screen.blit(line_surf, (0,0))


    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"MINERALS: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        life_icon_size = 8
        for i in range(self.player["lives"]):
            sx = self.SCREEN_WIDTH - 20 - i * (life_icon_size * 2.5)
            sy = 20
            ship_points = [
                (sx, sy - life_icon_size),
                (sx - life_icon_size * 0.7, sy + life_icon_size * 0.7),
                (sx, sy + life_icon_size * 0.3),
                (sx + life_icon_size * 0.7, sy + life_icon_size * 0.7)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, ship_points, self.COLOR_PLAYER)

        if self.game_over:
            msg = "MISSION COMPLETE" if self.game_won else "GAME OVER"
            color = self.COLOR_PLAYER if self.game_won else self.COLOR_ENEMY
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
            
    def _create_particle(self, pos, vel, lifespan, start_size, end_size, start_color, end_color):
        self.particles.append({
            "pos": pos, "vel": vel, "lifespan": lifespan, "max_lifespan": lifespan,
            "start_size": start_size, "end_size": end_size,
            "start_color": start_color, "end_color": end_color
        })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _render_particles(self, offset):
        for p in self.particles:
            progress = p["lifespan"] / p["max_lifespan"]
            size = int(p["end_size"] + (p["start_size"] - p["end_size"]) * progress)
            if size < 1: continue
            
            color = [
                int(p["end_color"][i] + (p["start_color"][i] - p["end_color"][i]) * progress)
                for i in range(3)
            ]
            
            sx, sy = self._world_to_screen(p["pos"])
            pygame.draw.circle(self.screen, color, (sx + offset[0], sy + offset[1]), size)

    def _world_to_screen(self, pos, parallax_scale=1.0):
        world_x, world_y = pos
        player_x, player_y = self.player["pos"]

        # Handle world wrapping for rendering
        dx = (world_x - player_x) * parallax_scale
        dy = (world_y - player_y) * parallax_scale
        
        # This logic ensures objects on the other side of the world wrap around correctly
        if dx > self.WORLD_WIDTH / 2: dx -= self.WORLD_WIDTH
        if dx < -self.WORLD_WIDTH / 2: dx += self.WORLD_WIDTH
        if dy > self.WORLD_HEIGHT / 2: dy -= self.WORLD_HEIGHT
        if dy < -self.WORLD_HEIGHT / 2: dy += self.WORLD_HEIGHT

        iso_x = (dx - dy) * self.ISO_TILE_WIDTH_HALF
        iso_y = (dx + dy) * self.ISO_TILE_HEIGHT_HALF

        screen_x = self.SCREEN_WIDTH / 2 + iso_x
        screen_y = self.SCREEN_HEIGHT / 2 + iso_y

        return int(screen_x), int(screen_y)
    
    def _get_wrapped_dist_sq(self, pos1, pos2):
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        
        if dx > self.WORLD_WIDTH / 2:
            dx = self.WORLD_WIDTH - dx
        if dy > self.WORLD_HEIGHT / 2:
            dy = self.WORLD_HEIGHT - dy
            
        return dx**2 + dy**2

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player["lives"],
            "player_pos": self.player["pos"].tolist()
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    # You might need to install pygame: pip install pygame
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroid Miner")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    # Set auto_advance=False for manual play to feel responsive
    env.auto_advance = False 

    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_SHIFT]: shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()