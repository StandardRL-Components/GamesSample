
# Generated: 2025-08-28T02:42:41.137759
# Source Brief: brief_04544.md
# Brief Index: 4544

        
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
        "Controls: ↑↓←→ to move. Hold Shift for a temporary shield. Press Space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of asteroids for 60 seconds in this top-down arcade shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_EPISODE_STEPS = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_ASTEROID = (128, 132, 142)
    COLOR_LASER = (255, 80, 80)
    COLOR_SHIELD = (80, 80, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_EXPLOSION = (255, 255, 255)

    # Player settings
    PLAYER_SPEED = 6
    PLAYER_RADIUS = 10
    
    # Laser settings
    LASER_SPEED = 12
    LASER_COOLDOWN = 6  # frames (5 per second)
    
    # Shield settings
    SHIELD_DURATION = 90  # 3 seconds
    SHIELD_COOLDOWN = 210 # 7 seconds
    
    # Asteroid settings
    ASTEROID_INITIAL_COUNT = 5
    ASTEROID_MAX_COUNT = 20
    ASTEROID_SPAWN_INTERVAL = 150 # 5 seconds
    ASTEROID_SPEED_INCREASE_INTERVAL = 300 # 10 seconds
    ASTEROID_SPEED_INCREASE_AMOUNT = 0.05

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        self.player = {}
        self.asteroids = []
        self.lasers = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.prev_space_held = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.prev_space_held = False

        self.player = {
            "x": self.WIDTH / 2,
            "y": self.HEIGHT / 2,
            "last_move_dir": (0, -1),  # Default up
            "shield_timer": 0,
            "shield_cooldown": 0,
            "laser_cooldown": 0,
        }

        self.asteroids = []
        self.lasers = []
        self.particles = []

        self.max_asteroids = self.ASTEROID_INITIAL_COUNT
        self.asteroid_base_speed = 1.0

        for _ in range(self.ASTEROID_INITIAL_COUNT):
            self._spawn_asteroid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Survival penalty
        
        self._handle_input(movement, space_held, shift_held)
        
        if self.player['shield_timer'] > 0 and shift_held:
            # Small cost for activating shield
            reward -= 0.05
            
        self._update_game_state()
        
        collision_rewards = self._handle_collisions()
        reward += collision_rewards

        self.steps += 1
        
        # Spawn new asteroids and increase difficulty
        if self.steps % self.ASTEROID_SPAWN_INTERVAL == 0 and self.max_asteroids < self.ASTEROID_MAX_COUNT:
            self.max_asteroids += 1
        if self.steps % self.ASTEROID_SPEED_INCREASE_INTERVAL == 0:
            self.asteroid_base_speed += self.ASTEROID_SPEED_INCREASE_AMOUNT
        if len(self.asteroids) < self.max_asteroids:
            self._spawn_asteroid()

        terminated = self.game_over or self.steps >= self.MAX_EPISODE_STEPS

        if not self.game_over and self.steps >= self.MAX_EPISODE_STEPS:
            self.game_won = True
            reward += 100 # Survival bonus

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, movement, space_held, shift_held):
        # --- Movement ---
        move_vector = [0, 0]
        if movement == 1: move_vector[1] = -1  # Up
        elif movement == 2: move_vector[1] = 1   # Down
        elif movement == 3: move_vector[0] = -1  # Left
        elif movement == 4: move_vector[0] = 1   # Right
        
        if movement != 0:
            self.player["last_move_dir"] = (move_vector[0], move_vector[1])

        self.player["x"] += move_vector[0] * self.PLAYER_SPEED
        self.player["y"] += move_vector[1] * self.PLAYER_SPEED
        self.player["x"] = np.clip(self.player["x"], 0, self.WIDTH)
        self.player["y"] = np.clip(self.player["y"], 0, self.HEIGHT)
        
        # --- Shield ---
        if shift_held and self.player["shield_cooldown"] == 0 and self.player["shield_timer"] == 0:
            self.player["shield_timer"] = self.SHIELD_DURATION
            self.player["shield_cooldown"] = self.SHIELD_COOLDOWN + self.SHIELD_DURATION
            self._create_particles(self.player['x'], self.player['y'], self.COLOR_SHIELD, 15, 3, 20)

        # --- Firing ---
        if space_held and not self.prev_space_held and self.player["laser_cooldown"] == 0:
            self._spawn_laser()
            self.player["laser_cooldown"] = self.LASER_COOLDOWN
        
        self.prev_space_held = space_held

    def _update_game_state(self):
        # Update timers
        if self.player["shield_timer"] > 0: self.player["shield_timer"] -= 1
        if self.player["shield_cooldown"] > 0: self.player["shield_cooldown"] -= 1
        if self.player["laser_cooldown"] > 0: self.player["laser_cooldown"] -= 1

        # Move asteroids
        for asteroid in self.asteroids:
            asteroid["x"] += asteroid["vx"]
            asteroid["y"] += asteroid["vy"]
            asteroid["angle"] += asteroid["rot_speed"]

        # Move lasers
        for laser in self.lasers:
            laser["x"] += laser["vx"]
            laser["y"] += laser["vy"]

        # Update particles
        for p in self.particles:
            p['life'] -= 1
            p['radius'] += p['growth']

        # Remove off-screen entities and dead particles
        self.asteroids = [a for a in self.asteroids if -50 < a['x'] < self.WIDTH + 50 and -50 < a['y'] < self.HEIGHT + 50]
        self.lasers = [l for l in self.lasers if 0 < l['x'] < self.WIDTH and 0 < l['y'] < self.HEIGHT]
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _handle_collisions(self):
        total_reward = 0
        
        # Player-Asteroid collision
        if self.player["shield_timer"] == 0 and not self.game_over:
            for asteroid in self.asteroids:
                dist = math.hypot(self.player["x"] - asteroid["x"], self.player["y"] - asteroid["y"])
                if dist < self.PLAYER_RADIUS + asteroid["radius"]:
                    self.game_over = True
                    self._create_particles(self.player['x'], self.player['y'], self.COLOR_PLAYER, 50, 2, 30)
                    # sound: player_death.wav
                    break
        
        # Laser-Asteroid collision
        lasers_to_remove = []
        asteroids_to_remove = []
        
        for i, laser in enumerate(self.lasers):
            for j, asteroid in enumerate(self.asteroids):
                if i in lasers_to_remove or j in asteroids_to_remove:
                    continue
                dist = math.hypot(laser["x"] - asteroid["x"], laser["y"] - asteroid["y"])
                if dist < asteroid["radius"]:
                    lasers_to_remove.append(i)
                    asteroids_to_remove.append(j)
                    
                    if asteroid['radius'] > 20: # Large asteroid
                        self.score += 20
                        total_reward += 20
                    else: # Small asteroid
                        self.score += 10
                        total_reward += 10
                    
                    self._create_particles(asteroid['x'], asteroid['y'], self.COLOR_EXPLOSION, 30, 2, 20)
                    # sound: asteroid_explosion.wav

        # Remove collided entities by creating new lists
        self.lasers = [l for i, l in enumerate(self.lasers) if i not in lasers_to_remove]
        self.asteroids = [a for j, a in enumerate(self.asteroids) if j not in asteroids_to_remove]
        
        return total_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['radius']), (*p['color'], alpha))

        # Render asteroids
        for asteroid in self.asteroids:
            points = self._get_rotated_polygon(asteroid["shape_points"], asteroid["angle"], (asteroid["x"], asteroid["y"]))
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

        # Render lasers
        for laser in self.lasers:
            pygame.draw.circle(self.screen, self.COLOR_LASER, (int(laser["x"]), int(laser["y"])), 3)

        # Render player
        if not self.game_over:
            # Shield effect
            if self.player["shield_timer"] > 0:
                pulse = (math.sin(self.steps * 0.5) + 1) / 2 # 0 to 1
                radius = self.PLAYER_RADIUS + 3 + pulse * 3
                alpha = 50 + pulse * 50
                pygame.gfxdraw.aacircle(self.screen, int(self.player["x"]), int(self.player["y"]), int(radius), (*self.COLOR_SHIELD, alpha))
            
            # Player ship
            angle = math.atan2(self.player["last_move_dir"][1], self.player["last_move_dir"][0]) + math.pi / 2
            player_points = [(-7, 8), (0, -12), (7, 8)]
            rotated_points = self._get_rotated_polygon(player_points, angle, (self.player["x"], self.player["y"]))
            pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_left = max(0, (self.MAX_EPISODE_STEPS - self.steps) / self.FPS)
        minutes = int(time_left // 60)
        seconds = int(time_left % 60)
        timer_text = self.font_small.render(f"TIME: {minutes:02}:{seconds:02}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        # Game Over / Win message
        if self.game_over:
            msg = "GAME OVER"
            color = self.COLOR_LASER
        elif self.game_won:
            msg = "YOU SURVIVED!"
            color = self.COLOR_SHIELD
        else:
            return

        end_text = self.font_large.render(msg, True, color)
        text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _spawn_asteroid(self):
        edge = self.np_random.integers(4)
        if edge == 0:  # Top
            x, y = self.np_random.uniform(0, self.WIDTH), -40
            angle = self.np_random.uniform(0.1 * math.pi, 0.9 * math.pi)
        elif edge == 1:  # Right
            x, y = self.WIDTH + 40, self.np_random.uniform(0, self.HEIGHT)
            angle = self.np_random.uniform(0.6 * math.pi, 1.4 * math.pi)
        elif edge == 2:  # Bottom
            x, y = self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 40
            angle = self.np_random.uniform(1.1 * math.pi, 1.9 * math.pi)
        else:  # Left
            x, y = -40, self.np_random.uniform(0, self.HEIGHT)
            angle = self.np_random.uniform(-0.4 * math.pi, 0.4 * math.pi)

        speed = self.asteroid_base_speed + self.np_random.uniform(-0.5, 0.5)
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        
        radius = self.np_random.uniform(15, 30)
        num_vertices = self.np_random.integers(7, 12)
        shape_points = []
        for i in range(num_vertices):
            angle_vert = 2 * math.pi * i / num_vertices
            dist = self.np_random.uniform(radius * 0.8, radius * 1.2)
            shape_points.append((math.cos(angle_vert) * dist, math.sin(angle_vert) * dist))

        self.asteroids.append({
            "x": x, "y": y, "vx": vx, "vy": vy,
            "radius": radius, "angle": 0,
            "rot_speed": self.np_random.uniform(-0.03, 0.03),
            "shape_points": shape_points
        })

    def _spawn_laser(self):
        dir_x, dir_y = self.player["last_move_dir"]
        # Position laser at the tip of the ship
        start_x = self.player["x"] + dir_x * 15
        start_y = self.player["y"] + dir_y * 15
        
        self.lasers.append({
            "x": start_x, "y": start_y,
            "vx": dir_x * self.LASER_SPEED,
            "vy": dir_y * self.LASER_SPEED,
        })
        # sound: laser_fire.wav
        
    def _create_particles(self, x, y, color, count, speed_max, life):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_max)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            self.particles.append({
                'x': x, 'y': y, 'vx': vx, 'vy': vy,
                'radius': self.np_random.uniform(1, 3),
                'life': life, 'max_life': life,
                'color': color, 'growth': self.np_random.uniform(0.1, 0.4)
            })

    def _get_rotated_polygon(self, points, angle, center):
        rotated_points = []
        for x, y in points:
            x_rot = x * math.cos(angle) - y * math.sin(angle) + center[0]
            y_rot = x * math.sin(angle) + y * math.cos(angle) + center[1]
            rotated_points.append((int(x_rot), int(y_rot)))
        return rotated_points

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Asteroid Survival")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Pygame uses a different coordinate system for surfaces
        # so we need to transpose the observation for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    env.close()