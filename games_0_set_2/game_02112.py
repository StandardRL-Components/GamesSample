
# Generated: 2025-08-27T19:18:04.795511
# Source Brief: brief_02112.md
# Brief Index: 2112

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold space to fire. Survive the asteroid field!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a relentless asteroid field for 60 seconds in this top-down space shooter. Destroy asteroids for points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = self.FPS * 60  # 60 seconds

        # Player settings
        self.PLAYER_SPEED = 5
        self.PLAYER_SIZE = 12
        self.FIRE_COOLDOWN = 10 # frames

        # Asteroid settings
        self.ASTEROID_SIZES = [10, 20, 30]
        self.INITIAL_ASTEROID_SPAWN_RATE = 120 # frames (2 seconds)
        self.INITIAL_ASTEROID_SPEED = 1.0

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_GLOW = (50, 255, 50, 50)
        self.COLOR_PROJECTILE = (255, 80, 80)
        self.COLOR_ASTEROID = (180, 180, 180)
        self.COLOR_EXPLOSION = (255, 180, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_STARS = [(255, 255, 255), (200, 200, 200), (150, 150, 150)]

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_small = pygame.font.SysFont('Consolas', 18)
        
        # Initialize state variables to be populated in reset()
        self.player_pos = None
        self.last_move_dir = None
        self.projectiles = []
        self.asteroids = []
        self.explosions = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fire_cooldown_timer = 0
        self.asteroid_spawn_timer = 0
        self.asteroid_spawn_rate = 0
        self.asteroid_speed = 0
        self.np_random = None

        # This call to reset() is essential for initializing the state
        self.reset()
        
        # self.validate_implementation() # Optional self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = [self.WIDTH / 2, self.HEIGHT - 50]
        self.last_move_dir = [0, -1] # Default aim upwards
        self.projectiles = []
        self.asteroids = []
        self.explosions = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fire_cooldown_timer = 0
        
        self.asteroid_spawn_rate = self.INITIAL_ASTEROID_SPAWN_RATE
        self.asteroid_speed = self.INITIAL_ASTEROID_SPEED
        self.asteroid_spawn_timer = self.asteroid_spawn_rate

        self._generate_starfield()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.1  # Survival reward per frame
        self.steps += 1
        
        if self.game_over:
            # If game is over, no updates, just return terminal state
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        self._handle_input(movement, space_held)
        self._update_player(movement)
        self._update_projectiles()
        self._update_asteroids()
        self._update_explosions()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward
        
        self._spawn_asteroids()
        self._update_difficulty()
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS

        if terminated:
            if not self.game_over: # Survived the full time
                reward += 100
            else: # Hit by asteroid
                reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _handle_input(self, movement, space_held):
        # Handle firing
        if self.fire_cooldown_timer > 0:
            self.fire_cooldown_timer -= 1
        
        if space_held and self.fire_cooldown_timer <= 0:
            self._fire_projectile()

        # Update aiming direction
        move_vec = [0, 0]
        if movement == 1: move_vec = [0, -1] # Up
        elif movement == 2: move_vec = [0, 1]  # Down
        elif movement == 3: move_vec = [-1, 0] # Left
        elif movement == 4: move_vec = [1, 0]  # Right
        
        if movement != 0:
            self.last_move_dir = move_vec

    def _update_player(self, movement):
        move_vec = [0, 0]
        if movement == 1: move_vec = [0, -1] # Up
        elif movement == 2: move_vec = [0, 1]  # Down
        elif movement == 3: move_vec = [-1, 0] # Left
        elif movement == 4: move_vec = [1, 0]  # Right

        self.player_pos[0] += move_vec[0] * self.PLAYER_SPEED
        self.player_pos[1] += move_vec[1] * self.PLAYER_SPEED

        # Screen wrapping
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT
    
    def _fire_projectile(self):
        # Sound: Laser fire
        self.fire_cooldown_timer = self.FIRE_COOLDOWN
        proj_start_pos = list(self.player_pos)
        
        # Offset to ship's nose
        angle = math.atan2(self.last_move_dir[1], self.last_move_dir[0])
        proj_start_pos[0] += math.cos(angle) * self.PLAYER_SIZE
        proj_start_pos[1] += math.sin(angle) * self.PLAYER_SIZE

        self.projectiles.append({
            "pos": proj_start_pos,
            "vel": [v * 10 for v in self.last_move_dir]
        })

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            if not (0 <= p["pos"][0] < self.WIDTH and 0 <= p["pos"][1] < self.HEIGHT):
                self.projectiles.remove(p)

    def _spawn_asteroids(self):
        self.asteroid_spawn_timer -= 1
        if self.asteroid_spawn_timer <= 0:
            self.asteroid_spawn_timer = self.asteroid_spawn_rate
            
            pos = [self.np_random.uniform(0, self.WIDTH), -30]
            angle = self.np_random.uniform(math.pi * 0.25, math.pi * 0.75)
            speed = self.asteroid_speed + self.np_random.uniform(-0.5, 0.5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = self.np_random.choice(self.ASTEROID_SIZES)
            
            self.asteroids.append({
                "pos": pos,
                "vel": vel,
                "size": size,
                "angle": 0,
                "rot_speed": self.np_random.uniform(-0.02, 0.02),
                "shape": self._generate_asteroid_shape(size)
            })

    def _update_asteroids(self):
        for a in self.asteroids:
            a["pos"][0] += a["vel"][0]
            a["pos"][1] += a["vel"][1]
            a["angle"] += a["rot_speed"]
            
            # Wrap around screen
            if a["pos"][0] < -a["size"]: a["pos"][0] = self.WIDTH + a["size"]
            if a["pos"][0] > self.WIDTH + a["size"]: a["pos"][0] = -a["size"]
            if a["pos"][1] < -a["size"]: a["pos"][1] = self.HEIGHT + a["size"]
            if a["pos"][1] > self.HEIGHT + a["size"]: a["pos"][1] = -a["size"]

    def _update_explosions(self):
        for e in self.explosions[:]:
            e["radius"] += e["expand_rate"]
            e["alpha"] = max(0, e["alpha"] - e["fade_rate"])
            if e["alpha"] == 0:
                self.explosions.remove(e)

    def _update_difficulty(self):
        # Increase speed every 10 seconds
        if self.steps > 0 and self.steps % (self.FPS * 10) == 0:
            self.asteroid_speed += 0.2

        # Increase spawn rate every 30 seconds
        if self.steps > 0 and self.steps % (self.FPS * 30) == 0:
            self.asteroid_spawn_rate = max(20, self.asteroid_spawn_rate * 0.75)

    def _handle_collisions(self):
        reward = 0
        
        # Projectile-Asteroid collisions
        for p in self.projectiles[:]:
            for a in self.asteroids[:]:
                dist = math.hypot(p["pos"][0] - a["pos"][0], p["pos"][1] - a["pos"][1])
                if dist < a["size"]:
                    # Sound: Explosion
                    self._create_explosion(a["pos"], a["size"])
                    self.score += int(40 - a["size"]) # Smaller asteroids are worth more
                    reward += 1
                    if p in self.projectiles: self.projectiles.remove(p)
                    self.asteroids.remove(a)
                    break

        # Player-Asteroid collisions
        for a in self.asteroids:
            dist = math.hypot(self.player_pos[0] - a["pos"][0], self.player_pos[1] - a["pos"][1])
            if dist < a["size"] + self.PLAYER_SIZE * 0.5:
                # Sound: Player explosion
                self.game_over = True
                self._create_explosion(self.player_pos, 40, is_player=True)
                break
        
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_asteroids()
        self._render_projectiles()
        if not self.game_over:
            self._render_player()
        self._render_explosions()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_stars(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, star['color'], (int(star['x']), int(star['y'])), star['size'])

    def _render_player(self):
        # Calculate points for the triangle
        angle = math.atan2(self.last_move_dir[1], self.last_move_dir[0]) + math.pi / 2
        p1 = (
            self.player_pos[0] + self.PLAYER_SIZE * math.cos(angle),
            self.player_pos[1] + self.PLAYER_SIZE * math.sin(angle)
        )
        p2 = (
            self.player_pos[0] + self.PLAYER_SIZE * math.cos(angle + 2 * math.pi / 3),
            self.player_pos[1] + self.PLAYER_SIZE * math.sin(angle + 2 * math.pi / 3)
        )
        p3 = (
            self.player_pos[0] + self.PLAYER_SIZE * math.cos(angle + 4 * math.pi / 3),
            self.player_pos[1] + self.PLAYER_SIZE * math.sin(angle + 4 * math.pi / 3)
        )
        
        # Draw glow
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), 
                                     int(self.PLAYER_SIZE * 1.5), self.COLOR_PLAYER_GLOW)
        # Draw ship
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
        
    def _render_projectiles(self):
        for p in self.projectiles:
            start_pos = (int(p["pos"][0]), int(p["pos"][1]))
            end_pos = (int(p["pos"][0] - p["vel"][0]), int(p["pos"][1] - p["vel"][1]))
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, start_pos, end_pos, 3)

    def _render_asteroids(self):
        for a in self.asteroids:
            rotated_shape = []
            for point in a["shape"]:
                x = point[0] * math.cos(a["angle"]) - point[1] * math.sin(a["angle"])
                y = point[0] * math.sin(a["angle"]) + point[1] * math.cos(a["angle"])
                rotated_shape.append((x + a["pos"][0], y + a["pos"][1]))
            
            if len(rotated_shape) > 2:
                pygame.gfxdraw.aapolygon(self.screen, rotated_shape, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, rotated_shape, self.COLOR_ASTEROID)

    def _render_explosions(self):
        for e in self.explosions:
            color = (*self.COLOR_EXPLOSION, e["alpha"])
            pygame.gfxdraw.filled_circle(self.screen, int(e["pos"][0]), int(e["pos"][1]), int(e["radius"]), color)

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        if self.game_over:
            result_text = "YOU SURVIVED!" if self.steps >= self.MAX_STEPS else "GAME OVER"
            result_surf = self.font.render(result_text, True, self.COLOR_TEXT)
            self.screen.blit(result_surf, (self.WIDTH/2 - result_surf.get_width()/2, self.HEIGHT/2 - result_surf.get_height()/2))

    def _generate_starfield(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'x': self.np_random.uniform(0, self.WIDTH),
                'y': self.np_random.uniform(0, self.HEIGHT),
                'size': self.np_random.choice([1, 2]),
                'color': self.np_random.choice(self.COLOR_STARS)
            })

    def _generate_asteroid_shape(self, radius):
        points = []
        num_vertices = self.np_random.integers(7, 12)
        for i in range(num_vertices):
            angle = i * (2 * math.pi / num_vertices)
            dist = self.np_random.uniform(radius * 0.7, radius * 1.3)
            points.append((dist * math.cos(angle), dist * math.sin(angle)))
        return points

    def _create_explosion(self, pos, size, is_player=False):
        self.explosions.append({
            "pos": list(pos),
            "radius": size * 0.5,
            "alpha": 255,
            "expand_rate": 2 if is_player else 1,
            "fade_rate": 8 if is_player else 12
        })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
    # This block allows you to run the file directly to play the game
    import sys
    
    env = GameEnv(render_mode="rgb_array")
    env.reset()

    # Use a dummy screen for human play, as the env itself is headless
    pygame.display.set_caption("Asteroid Survival")
    human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    total_reward = 0
    
    # Game loop for human play
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0 # Unused in this game

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        if keys[pygame.K_SPACE]: space_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before resetting or quitting
            pygame.time.wait(2000)
            env.reset()
            total_reward = 0

        env.clock.tick(env.FPS)

    env.close()
    sys.exit()