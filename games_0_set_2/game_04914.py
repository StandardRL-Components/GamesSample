
# Generated: 2025-08-28T03:23:26.858856
# Source Brief: brief_04914.md
# Brief Index: 4914

        
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
        "Controls: Use arrow keys to move. Hold space to mine nearby asteroids."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a space miner through an asteroid field, collecting valuable minerals while dodging collisions."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.WIN_SCORE = 500
        self.MAX_STEPS = 5000
        self.MAX_LIVES = 3
        self.PLAYER_SPEED = 0.5
        self.PLAYER_FRICTION = 0.95
        self.PLAYER_MAX_SPEED = 6.0
        self.PLAYER_RADIUS = 12
        self.PLAYER_INVINCIBILITY_FRAMES = 90 # 3 seconds at 30fps
        self.ASTEROID_SPAWN_RATE = 40
        self.MAX_ASTEROIDS = 15
        self.MINING_RANGE = 120
        self.DIFFICULTY_INTERVAL = 200

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_THRUSTER = (255, 180, 0)
        self.COLOR_ASTEROID = [(80, 80, 90), (100, 90, 80), (70, 60, 60)]
        self.COLOR_MINERALS = {1: (255, 215, 0), 2: (100, 255, 100), 5: (200, 100, 255)}
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_RED = (255, 50, 50)
        self.COLOR_LASER = (50, 255, 50)
        
        # Initialize state variables to avoid attribute errors
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.win = False
        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_invincibility = 0
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.mining_target = None
        self.asteroid_speed_multiplier = 1.0

        # Pre-generate static background stars
        for _ in range(150):
            self.stars.append(
                (
                    random.randint(0, self.SCREEN_WIDTH),
                    random.randint(0, self.SCREEN_HEIGHT),
                    random.choice([1, 2, 2, 3]),
                )
            )

        # Initialize state
        self.reset()

        # Run validation check
        # self.validate_implementation() # Commented out for submission, but useful for testing
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.win = False
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT / 2.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_invincibility = self.PLAYER_INVINCIBILITY_FRAMES

        self.asteroids = []
        self.particles = []
        self.mining_target = None
        self.asteroid_speed_multiplier = 1.0

        for _ in range(5):
            self._spawn_asteroid(on_edge=True)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        # --- UPDATE GAME LOGIC ---

        # 1. Handle Player Input and Movement
        accel = np.array([0.0, 0.0])
        if movement == 1: accel[1] -= self.PLAYER_SPEED
        if movement == 2: accel[1] += self.PLAYER_SPEED
        if movement == 3: accel[0] -= self.PLAYER_SPEED
        if movement == 4: accel[0] += self.PLAYER_SPEED
        
        self.player_vel += accel
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel / speed * self.PLAYER_MAX_SPEED
            
        self.player_pos += self.player_vel
        self.player_vel *= self.PLAYER_FRICTION
        
        # Keep player on screen
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

        # Update invincibility
        if self.player_invincibility > 0:
            self.player_invincibility -= 1
        
        # 2. Update Asteroids and check collisions
        for asteroid in self.asteroids[:]:
            asteroid["pos"] += asteroid["vel"] * self.asteroid_speed_multiplier
            
            # Screen wrapping for asteroids
            if asteroid["pos"][0] < -asteroid["radius"]: asteroid["pos"][0] = self.SCREEN_WIDTH + asteroid["radius"]
            if asteroid["pos"][0] > self.SCREEN_WIDTH + asteroid["radius"]: asteroid["pos"][0] = -asteroid["radius"]
            if asteroid["pos"][1] < -asteroid["radius"]: asteroid["pos"][1] = self.SCREEN_HEIGHT + asteroid["radius"]
            if asteroid["pos"][1] > self.SCREEN_HEIGHT + asteroid["radius"]: asteroid["pos"][1] = -asteroid["radius"]

            # Player-asteroid collision
            dist = np.linalg.norm(self.player_pos - asteroid["pos"])
            if dist < self.PLAYER_RADIUS + asteroid["radius"] and self.player_invincibility == 0:
                self._handle_player_hit()
                reward -= 10
                self.asteroids.remove(asteroid)
                break # Only one collision per frame

        # 3. Handle Mining
        self.mining_target = None
        is_mining = False
        if space_held:
            closest_asteroid = None
            min_dist = self.MINING_RANGE
            
            for asteroid in self.asteroids:
                dist = np.linalg.norm(self.player_pos - asteroid["pos"])
                if dist < min_dist:
                    min_dist = dist
                    closest_asteroid = asteroid
            
            if closest_asteroid:
                is_mining = True
                self.mining_target = closest_asteroid
                
                # Deplete minerals and grant reward
                mined_amount = min(closest_asteroid["minerals"], 1)
                closest_asteroid["minerals"] -= mined_amount
                
                mineral_reward = closest_asteroid["value_tier"] * mined_amount
                self.score += mineral_reward
                reward += mineral_reward
                
                # Sound: mining_laser.wav
                # Sound: mineral_collect.wav

                # Create collection particles
                if self.steps % 2 == 0:
                    self._create_particles(
                        1, 
                        closest_asteroid["pos"], 
                        target=self.player_pos, 
                        color=self.COLOR_MINERALS[closest_asteroid["value_tier"]], 
                        speed_min=2, 
                        speed_max=4, 
                        lifespan=min_dist / 3,
                        size=3
                    )
                
                if closest_asteroid["minerals"] <= 0:
                    self._create_particles(20, closest_asteroid["pos"], color=(150, 150, 160), speed_min=1, speed_max=3, size=2)
                    self.asteroids.remove(closest_asteroid)
                    self.mining_target = None
                    # Sound: asteroid_destroyed.wav

        if not is_mining:
            reward -= 0.1

        # 4. Update Particles
        for p in self.particles[:]:
            if p["target"] is not None:
                direction = p["target"] - p["pos"]
                dist = np.linalg.norm(direction)
                if dist > 1:
                    p["vel"] = direction / dist * p["speed"]
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

        # 5. Spawn new asteroids
        if self.steps % self.ASTEROID_SPAWN_RATE == 0 and len(self.asteroids) < self.MAX_ASTEROIDS:
            self._spawn_asteroid(on_edge=True)
            
        # 6. Difficulty Scaling
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.asteroid_speed_multiplier += 0.05

        # 7. Update Step Counter
        self.steps += 1
        
        # 8. Check Termination Conditions
        if self.lives <= 0:
            self.game_over = True
            terminated = True
        if self.score >= self.WIN_SCORE:
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _handle_player_hit(self):
        self.lives -= 1
        self._create_particles(50, self.player_pos, color=self.COLOR_PLAYER, speed_min=2, speed_max=5, size=3)
        self._create_particles(20, self.player_pos, color=self.COLOR_RED, speed_min=1, speed_max=4, size=4)
        # Sound: player_explosion.wav
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT / 2.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_invincibility = self.PLAYER_INVINCIBILITY_FRAMES

    def _spawn_asteroid(self, on_edge=False):
        if on_edge:
            edge = random.randint(0, 3)
            if edge == 0: # Top
                pos = np.array([random.uniform(0, self.SCREEN_WIDTH), -30.0])
            elif edge == 1: # Bottom
                pos = np.array([random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 30.0])
            elif edge == 2: # Left
                pos = np.array([-30.0, random.uniform(0, self.SCREEN_HEIGHT)])
            else: # Right
                pos = np.array([self.SCREEN_WIDTH + 30.0, random.uniform(0, self.SCREEN_HEIGHT)])
        else:
            pos = np.array([random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT)])
        
        radius = random.uniform(15, 40)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.2, 1.0)
        vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
        
        value_prob = random.random()
        if value_prob < 0.6:
            value_tier = 1
        elif value_prob < 0.9:
            value_tier = 2
        else:
            value_tier = 5
        
        minerals = int(radius * 2)

        # Create irregular polygon shape
        num_vertices = random.randint(7, 12)
        shape_points = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            dist = random.uniform(radius * 0.8, radius)
            shape_points.append((math.cos(angle) * dist, math.sin(angle) * dist))

        self.asteroids.append({
            "pos": pos,
            "vel": vel,
            "radius": radius,
            "minerals": minerals,
            "max_minerals": minerals,
            "value_tier": value_tier,
            "shape_points": shape_points,
            "color": random.choice(self.COLOR_ASTEROID),
            "rot": random.uniform(0, 2*math.pi),
            "rot_speed": random.uniform(-0.02, 0.02)
        })

    def _create_particles(self, num, pos, color, speed_min, speed_max, lifespan=20, size=2, target=None):
        for _ in range(num):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(speed_min, speed_max)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": random.uniform(lifespan * 0.8, lifespan * 1.2),
                "max_lifespan": lifespan,
                "color": color,
                "size": size,
                "target": target,
                "speed": speed
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render stars
        for x, y, size in self.stars:
            c = 50 + (size - 1) * 40
            pygame.draw.rect(self.screen, (c,c,c), (x, y, size, size))
            
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            color = p["color"]
            s = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (color[0], color[1], color[2], alpha), (p["size"], p["size"]), p["size"])
            self.screen.blit(s, (int(p["pos"][0] - p["size"]), int(p["pos"][1] - p["size"])))

        # Render mining laser
        if self.mining_target:
            start_pos = (int(self.player_pos[0]), int(self.player_pos[1]))
            end_pos = (int(self.mining_target["pos"][0]), int(self.mining_target["pos"][1]))
            width = int(2 + math.sin(self.steps * 0.5) * 1.5)
            pygame.draw.line(self.screen, self.COLOR_LASER, start_pos, end_pos, width)
            pygame.gfxdraw.aacircle(self.screen, end_pos[0], end_pos[1], 10, self.COLOR_LASER)

        # Render asteroids
        for asteroid in self.asteroids:
            asteroid["rot"] += asteroid["rot_speed"]
            points = []
            for x, y in asteroid["shape_points"]:
                rot_x = x * math.cos(asteroid["rot"]) - y * math.sin(asteroid["rot"])
                rot_y = x * math.sin(asteroid["rot"]) + y * math.cos(asteroid["rot"])
                points.append((int(rot_x + asteroid["pos"][0]), int(rot_y + asteroid["pos"][1])))
            
            # Draw a darker 'back' face for pseudo-3D effect
            back_points = [(p[0]+2, p[1]+2) for p in points]
            pygame.gfxdraw.filled_polygon(self.screen, back_points, (30,30,40))

            pygame.gfxdraw.aapolygon(self.screen, points, asteroid["color"])
            pygame.gfxdraw.filled_polygon(self.screen, points, asteroid["color"])

        # Render player
        is_visible = self.player_invincibility == 0 or self.steps % 10 < 5
        if is_visible:
            # Thruster flame
            if np.linalg.norm(self.player_vel) > 0.5:
                angle = math.atan2(self.player_vel[1], self.player_vel[0]) + math.pi
                flame_len = 10 + np.linalg.norm(self.player_vel) * 2 + random.uniform(-2, 2)
                p1 = (int(self.player_pos[0]), int(self.player_pos[1]))
                p2 = (int(p1[0] + math.cos(angle - 0.2) * flame_len), int(p1[1] + math.sin(angle - 0.2) * flame_len))
                p3 = (int(p1[0] + math.cos(angle + 0.2) * flame_len), int(p1[1] + math.sin(angle + 0.2) * flame_len))
                pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_THRUSTER)
                pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_THRUSTER)

            # Ship body
            angle = math.atan2(self.player_vel[1], self.player_vel[0])
            r = self.PLAYER_RADIUS
            p1 = (int(self.player_pos[0] + math.cos(angle) * r), int(self.player_pos[1] + math.sin(angle) * r))
            p2 = (int(self.player_pos[0] + math.cos(angle + 2.5) * r * 0.8), int(self.player_pos[1] + math.sin(angle + 2.5) * r * 0.8))
            p3 = (int(self.player_pos[0] + math.cos(angle - 2.5) * r * 0.8), int(self.player_pos[1] + math.sin(angle - 2.5) * r * 0.8))
            
            # Draw a darker 'back' face for pseudo-3D effect
            back_points = [(p[0]+2, p[1]+2) for p in [p1, p2, p3]]
            pygame.gfxdraw.filled_polygon(self.screen, back_points, (0, 100, 120))

            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

        # Render UI
        score_text = self.font_ui.render(f"SCORE: {self.score}/{self.WIN_SCORE}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))

        # Render lives
        for i in range(self.lives):
            p1 = (self.SCREEN_WIDTH - 30 - i * 25, 15)
            p2 = (p1[0] - 8, p1[1] + 15)
            p3 = (p1[0] + 8, p1[1] + 15)
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1, p2, p3])

        # Render Game Over/Win message
        if self.game_over:
            msg_text = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_MINERALS[1] if self.win else self.COLOR_RED
            msg_render = self.font_msg.render(msg_text, True, color)
            msg_rect = msg_render.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_render, msg_rect)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "win": self.win,
        }

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
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # Override screen for direct rendering
    env.screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Space Miner")

    terminated = False
    total_reward = 0
    
    # --- Keyboard Control Mapping ---
    # This maps pygame keys to the MultiDiscrete action space
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Main game loop
    running = True
    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        # Movement
        movement_action = 0 # no-op
        for key, move_val in key_map.items():
            if keys[key]:
                movement_action = move_val
                break # Prioritize first key found in map if multiple are pressed
        action[0] = movement_action
        
        # Space and Shift
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        env.screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30) # Run at 30 FPS

    env.close()