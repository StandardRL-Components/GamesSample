
# Generated: 2025-08-28T03:46:46.870671
# Source Brief: brief_02119.md
# Brief Index: 2119

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to move your ship. Hold space near an asteroid to mine it. Avoid the red debris!"
    )

    # User-facing description of the game
    game_description = (
        "Pilot a spaceship in a top-down asteroid field, mining ore for survival and riches while dodging hazardous space debris."
    )

    # Frames auto-advance at 30fps
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 50, bold=True)

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_STAR = (200, 200, 220)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.COLOR_ASTEROID = (120, 110, 100)
        self.COLOR_DEBRIS = (255, 50, 50)
        self.COLOR_ORE = (255, 220, 0)
        self.COLOR_TEXT = (240, 240, 255)
        self.COLOR_EXPLOSION = (255, 150, 50)
        self.COLOR_LASER = (255, 255, 255)
        self.COLOR_INVINCIBLE = (200, 220, 255)

        # Game constants
        self.MAX_STEPS = 5000
        self.WIN_ORE_COUNT = 100
        self.INITIAL_LIVES = 3
        self.PLAYER_SPEED = 4.5
        self.PLAYER_RADIUS = 12
        self.MINING_RANGE = 80
        self.INVINCIBILITY_FRAMES = 90  # 3 seconds at 30fps

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.player_target_angle = None
        self.player_lives = None
        self.invincibility_timer = None
        self.ore_collected = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.asteroids = []
        self.debris = []
        self.particles = []
        self.stars = []
        self.debris_speed = None
        self.asteroid_respawn_time = None
        
        # Initialize and validate
        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset player state
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0, 0], dtype=np.float32)
        self.player_angle = -90.0
        self.player_target_angle = -90.0
        self.player_lives = self.INITIAL_LIVES
        self.invincibility_timer = 0

        # Reset game state
        self.steps = 0
        self.score = 0
        self.ore_collected = 0
        self.game_over = False
        
        # Reset progression
        self.debris_speed = 1.5
        self.asteroid_respawn_time = 30 * 3 # 3 seconds

        # Generate background stars
        self.stars = [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.uniform(0.5, 1.5))
            for _ in range(100)
        ]
        
        # Clear entity lists
        self.asteroids.clear()
        self.debris.clear()
        self.particles.clear()
        
        # Spawn initial entities
        for _ in range(5):
            self._spawn_asteroid()
        for _ in range(8):
            self._spawn_debris()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
        
        reward = -0.02 # Small penalty for surviving
        self.steps += 1

        if not self.game_over:
            # Unpack action
            movement = action[0]
            space_held = action[1] == 1

            # Update game logic
            self._handle_input(movement)
            self._update_player()
            self._update_asteroids(space_held)
            self._update_debris()
            self._update_particles()
            
            # Handle collisions and rewards
            reward += self._check_collisions()
            reward += self._collect_ore_particles()

            # Update difficulty
            if self.steps > 0 and self.steps % 500 == 0:
                self.debris_speed += 0.05
            if self.steps > 0 and self.steps % 1000 == 0:
                self.asteroid_respawn_time = max(30, self.asteroid_respawn_time - 3) # min 1 sec
        
        # Check for termination
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.ore_collected >= self.WIN_ORE_COUNT:
                reward += 100
            elif self.player_lives <= 0:
                reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement):
        move_vec = np.array([0, 0], dtype=np.float32)
        if movement == 1:  # Up
            move_vec[1] = -1
            self.player_target_angle = -90
        elif movement == 2:  # Down
            move_vec[1] = 1
            self.player_target_angle = 90
        elif movement == 3:  # Left
            move_vec[0] = -1
            self.player_target_angle = 180
        elif movement == 4:  # Right
            move_vec[0] = 1
            self.player_target_angle = 0
        
        if np.linalg.norm(move_vec) > 0:
            move_vec = move_vec / np.linalg.norm(move_vec)

        self.player_vel = move_vec * self.PLAYER_SPEED

    def _update_player(self):
        # Update position
        self.player_pos += self.player_vel
        
        # Clamp position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

        # Smooth rotation
        angle_diff = (self.player_target_angle - self.player_angle + 180) % 360 - 180
        self.player_angle += angle_diff * 0.2
        self.player_angle %= 360
        
        # Invincibility
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1
        
        # Engine trail
        if np.linalg.norm(self.player_vel) > 0.1:
            if self.steps % 3 == 0:
                angle_rad = math.radians(self.player_angle + 180)
                offset = np.array([math.cos(angle_rad), math.sin(angle_rad)]) * self.PLAYER_RADIUS
                p_pos = self.player_pos + offset
                p_vel = offset * -0.5 + self.np_random.uniform(-0.5, 0.5, 2)
                p_life = 15
                p_size = self.np_random.uniform(3, 6)
                self.particles.append({"pos": p_pos, "vel": p_vel, "life": p_life, "size": p_size, "color": self.COLOR_INVINCIBLE})

    def _update_asteroids(self, is_mining):
        mining_target = None
        min_dist = self.MINING_RANGE

        # Find closest minable asteroid
        for asteroid in self.asteroids:
            if asteroid["timer"] <= 0:
                dist = np.linalg.norm(self.player_pos - asteroid["pos"])
                if dist < min_dist:
                    min_dist = dist
                    mining_target = asteroid

        # Handle mining action
        if is_mining and mining_target:
            # sfx: mine_loop
            mining_target["ore"] -= 1
            if self.steps % 5 == 0: # Create ore particle
                p_pos = mining_target["pos"] + self.np_random.uniform(-5, 5, 2)
                self.particles.append({"pos": p_pos, "vel": (0,0), "life": 120, "size": 5, "color": self.COLOR_ORE, "type": "ore"})
            
            if mining_target["ore"] <= 0:
                # sfx: asteroid_depleted
                if mining_target["base_size"] == 'large': reward_val = 1.0
                elif mining_target["base_size"] == 'medium': reward_val = 0.5
                else: reward_val = 0.2
                self.score += reward_val
                mining_target["timer"] = self.asteroid_respawn_time

        # Update respawn timers
        for asteroid in self.asteroids:
            if asteroid["timer"] > 0:
                asteroid["timer"] -= 1
                if asteroid["timer"] <= 0:
                    self._respawn_asteroid(asteroid)

    def _update_debris(self):
        for d in self.debris:
            d["pos"] += d["vel"] * self.debris_speed
            # Wrap around screen
            if d["pos"][0] < -d["radius"]: d["pos"][0] = self.WIDTH + d["radius"]
            if d["pos"][0] > self.WIDTH + d["radius"]: d["pos"][0] = -d["radius"]
            if d["pos"][1] < -d["radius"]: d["pos"][1] = self.HEIGHT + d["radius"]
            if d["pos"][1] > self.HEIGHT + d["radius"]: d["pos"][1] = -d["radius"]

    def _update_particles(self):
        for p in self.particles[:]:
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
                continue
            
            if p.get("type") == "ore":
                # Ore particles home in on the player
                direction = self.player_pos - p["pos"]
                dist = np.linalg.norm(direction)
                if dist > 1:
                    direction /= dist
                p["vel"] = direction * 3
                p["pos"] += p["vel"]
            else:
                p["pos"] += p["vel"]
                p["size"] *= 0.95

    def _check_collisions(self):
        # Player vs Debris
        if self.invincibility_timer <= 0:
            for d in self.debris:
                dist = np.linalg.norm(self.player_pos - d["pos"])
                if dist < self.PLAYER_RADIUS + d["radius"]:
                    # sfx: explosion
                    self.player_lives -= 1
                    self.invincibility_timer = self.INVINCIBILITY_FRAMES
                    self._create_explosion(self.player_pos)
                    # Reset player to center to give a chance to recover
                    self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
                    return -10 # Penalty for getting hit
        return 0

    def _collect_ore_particles(self):
        reward = 0
        for p in self.particles[:]:
            if p.get("type") == "ore":
                dist = np.linalg.norm(self.player_pos - p["pos"])
                if dist < self.PLAYER_RADIUS:
                    # sfx: collect_ore
                    self.particles.remove(p)
                    self.ore_collected += 1
                    reward += 0.1
                    self.score += 0.1
        return reward

    def _check_termination(self):
        return (
            self.player_lives <= 0
            or self.ore_collected >= self.WIN_ORE_COUNT
            or self.steps >= self.MAX_STEPS
        )

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (x, y), size)
        
        # Debris
        for d in self.debris:
            pygame.gfxdraw.filled_circle(self.screen, int(d["pos"][0]), int(d["pos"][1]), int(d["radius"]), self.COLOR_DEBRIS)
            pygame.gfxdraw.aacircle(self.screen, int(d["pos"][0]), int(d["pos"][1]), int(d["radius"]), self.COLOR_DEBRIS)

        # Asteroids
        for a in self.asteroids:
            if a["timer"] <= 0:
                pygame.gfxdraw.filled_polygon(self.screen, a["shape"], self.COLOR_ASTEROID)
                pygame.gfxdraw.aapolygon(self.screen, a["shape"], self.COLOR_ASTEROID)

        # Particles
        for p in self.particles:
            color = p["color"]
            if p.get("type") != "ore": # Fade out non-ore particles
                alpha = int(255 * (p["life"] / 15))
                color = (*color, alpha)
            if p["size"] > 1:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["size"]), color)

        # Mining Laser
        is_mining = self.action_space.sample()[1] == 1 # A bit of a hack for rendering, but needed
        mining_target = None
        if is_mining:
            min_dist = self.MINING_RANGE
            for asteroid in self.asteroids:
                if asteroid["timer"] <= 0:
                    dist = np.linalg.norm(self.player_pos - asteroid["pos"])
                    if dist < min_dist:
                        min_dist = dist
                        mining_target = asteroid
        if mining_target:
            alpha = self.np_random.integers(100, 200)
            pygame.draw.line(self.screen, (*self.COLOR_LASER, alpha), self.player_pos, mining_target["pos"], 2)

        # Player
        color = self.COLOR_PLAYER if self.invincibility_timer % 10 < 5 else self.COLOR_INVINCIBLE
        if self.invincibility_timer <= 0: color = self.COLOR_PLAYER

        # Glow
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), int(self.PLAYER_RADIUS * 1.8), self.COLOR_PLAYER_GLOW)
        
        # Ship body
        player_points = self._get_rotated_ship_points()
        pygame.gfxdraw.filled_polygon(self.screen, player_points, color)
        pygame.gfxdraw.aapolygon(self.screen, player_points, color)


    def _get_rotated_ship_points(self):
        angle = math.radians(self.player_angle)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        
        points = [
            (self.PLAYER_RADIUS, 0),
            (-self.PLAYER_RADIUS * 0.7, self.PLAYER_RADIUS * 0.8),
            (-self.PLAYER_RADIUS * 0.4, 0),
            (-self.PLAYER_RADIUS * 0.7, -self.PLAYER_RADIUS * 0.8),
        ]
        
        rotated_points = []
        for x, y in points:
            new_x = self.player_pos[0] + x * cos_a - y * sin_a
            new_y = self.player_pos[1] + x * sin_a + y * cos_a
            rotated_points.append((int(new_x), int(new_y)))
            
        return rotated_points

    def _render_ui(self):
        # Ore counter
        ore_text = self.font_main.render(f"ORE: {self.ore_collected}/{self.WIN_ORE_COUNT}", True, self.COLOR_TEXT)
        self.screen.blit(ore_text, (10, 10))

        # Lives display
        for i in range(self.player_lives):
            ship_icon_points = [
                (self.WIDTH - 25 - i * 25, 15),
                (self.WIDTH - 35 - i * 25, 25),
                (self.WIDTH - 35 - i * 25, 5)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, ship_icon_points)

        # Game Over/Win message
        if self.game_over:
            if self.ore_collected >= self.WIN_ORE_COUNT:
                msg = "MISSION COMPLETE"
                color = self.COLOR_ORE
            else:
                msg = "GAME OVER"
                color = self.COLOR_DEBRIS
            
            text_surf = self.font_big.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ore_collected": self.ore_collected,
            "lives": self.player_lives,
        }
        
    def _spawn_asteroid(self):
        pos = self.np_random.uniform([50, 50], [self.WIDTH - 50, self.HEIGHT - 50])
        size_type = self.np_random.choice(['small', 'medium', 'large'], p=[0.5, 0.3, 0.2])
        if size_type == 'small':
            radius, ore = 15, 20
        elif size_type == 'medium':
            radius, ore = 25, 50
        else: # large
            radius, ore = 35, 100
        
        shape_points = self._create_asteroid_shape(pos, radius, self.np_random.uniform(0.3, 0.7), 7)
        self.asteroids.append({"pos": pos, "radius": radius, "ore": ore, "timer": 0, "shape": shape_points, "base_size": size_type})

    def _respawn_asteroid(self, asteroid):
        asteroid["pos"] = self.np_random.uniform([50, 50], [self.WIDTH - 50, self.HEIGHT - 50])
        if asteroid["base_size"] == 'small': asteroid["ore"] = 20
        elif asteroid["base_size"] == 'medium': asteroid["ore"] = 50
        else: asteroid["ore"] = 100
        asteroid["shape"] = self._create_asteroid_shape(asteroid["pos"], asteroid["radius"], self.np_random.uniform(0.3, 0.7), 7)

    def _create_asteroid_shape(self, center, avg_radius, irregularity, num_vertices):
        points = []
        for i in range(num_vertices):
            angle = i * (2 * math.pi / num_vertices)
            radius = self.np_random.uniform(avg_radius * (1 - irregularity), avg_radius * (1 + irregularity))
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((int(x), int(y)))
        return points
        
    def _spawn_debris(self):
        edge = self.np_random.integers(4)
        if edge == 0: # top
            pos = np.array([self.np_random.uniform(0, self.WIDTH), -20], dtype=np.float32)
            vel = np.array([self.np_random.uniform(-1, 1), 1], dtype=np.float32)
        elif edge == 1: # bottom
            pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20], dtype=np.float32)
            vel = np.array([self.np_random.uniform(-1, 1), -1], dtype=np.float32)
        elif edge == 2: # left
            pos = np.array([-20, self.np_random.uniform(0, self.HEIGHT)], dtype=np.float32)
            vel = np.array([1, self.np_random.uniform(-1, 1)], dtype=np.float32)
        else: # right
            pos = np.array([self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT)], dtype=np.float32)
            vel = np.array([-1, self.np_random.uniform(-1, 1)], dtype=np.float32)

        vel /= np.linalg.norm(vel)
        radius = self.np_random.uniform(5, 12)
        self.debris.append({"pos": pos, "vel": vel, "radius": radius})
        
    def _create_explosion(self, position):
        for _ in range(30):
            vel = self.np_random.uniform(-3, 3, 2)
            life = self.np_random.integers(10, 20)
            size = self.np_random.uniform(2, 8)
            self.particles.append({"pos": position.copy(), "vel": vel, "life": life, "size": size, "color": self.COLOR_EXPLOSION})

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headlessly
    
    env = GameEnv()
    obs, info = env.reset()
    
    print("Starting random agent test...")
    terminated = False
    total_reward = 0
    step_count = 0
    
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        if terminated or truncated:
            break
            
    print(f"Random agent finished in {step_count} steps.")
    print(f"Final score: {info['score']:.2f}, Total reward: {total_reward:.2f}")
    print(f"Ore collected: {info['ore_collected']}, Lives left: {info['lives']}")
    env.close()