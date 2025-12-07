
# Generated: 2025-08-27T13:38:34.264295
# Source Brief: brief_00434.md
# Brief Index: 434

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your ship. Hold Space to activate the mining beam."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship in a dangerous asteroid field. Harvest ore from asteroids while dodging enemy lasers. Collect 100 ore to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Game parameters
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 2000
    WIN_SCORE = 100
    STARTING_LIVES = 3
    
    # Colors
    COLOR_BG = (10, 5, 20)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_INVULN = (255, 255, 255)
    COLOR_ASTEROID = (120, 110, 100)
    COLOR_LASER = (255, 20, 20)
    COLOR_ORE = (255, 220, 0)
    COLOR_EXPLOSION = (255, 100, 0)
    COLOR_TEXT = (220, 220, 255)
    
    # Player physics
    PLAYER_THRUST = 0.25
    PLAYER_DRAG = 0.98
    PLAYER_MAX_SPEED = 6
    PLAYER_SIZE = 12
    INVULNERABILITY_FRAMES = 90 # 3 seconds at 30fps

    # Mining
    MINING_RANGE = 150
    MINING_ANGLE = 0.3 # Radians, approx 17 degrees
    MINING_RATE = 0.5
    
    # Asteroids
    MIN_ASTEROIDS = 10
    MAX_ASTEROIDS = 20
    ASTEROID_SPAWN_RADIUS = 500
    ASTEROID_DESPAWN_RADIUS = 600
    ASTEROID_MIN_SIZE = 15
    ASTEROID_MAX_SIZE = 40
    LARGE_ASTEROID_THRESHOLD = 30
    
    # Lasers
    LASER_SPEED = 7
    BASE_LASER_CHANCE = 0.01
    LASER_CHANCE_PER_ORE = 0.002
    MAX_LASER_CHANCE = 0.2
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.invulnerable_timer = None
        
        self.asteroids = []
        self.lasers = []
        self.particles = []
        self.stars = []
        
        self.reset()

        # Run self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_angle = -math.pi / 2
        
        self.lives = self.STARTING_LIVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.invulnerable_timer = 0
        
        self.asteroids = []
        self.lasers = []
        self.particles = []
        
        # Create starfield
        self.stars = []
        for i in range(3): # 3 layers for parallax
            for _ in range(100):
                self.stars.append({
                    "pos": np.array([self.np_random.uniform(-self.WIDTH, self.WIDTH), 
                                     self.np_random.uniform(-self.HEIGHT, self.HEIGHT)]),
                    "layer": i + 1
                })
        
        # Initial asteroid spawn
        for _ in range(self.MAX_ASTEROIDS):
            self._spawn_asteroid(initial_spawn=True)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # Update game logic
        self._update_player(movement)
        is_mining = self._update_mining(space_held)
        self._update_asteroids()
        self._update_lasers()
        self._update_particles()
        
        # Calculate reward
        reward += self._calculate_reward(is_mining)
        
        # Check termination conditions
        terminated = self._check_termination()
        
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        # Apply thrust based on input
        thrust = np.array([0.0, 0.0])
        if movement == 1: # Up
            thrust[1] -= self.PLAYER_THRUST
        elif movement == 2: # Down
            thrust[1] += self.PLAYER_THRUST
        elif movement == 3: # Left
            thrust[0] -= self.PLAYER_THRUST
        elif movement == 4: # Right
            thrust[0] += self.PLAYER_THRUST
            
        self.player_vel += thrust
        
        # Apply drag
        self.player_vel *= self.PLAYER_DRAG
        
        # Clamp speed
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel / speed * self.PLAYER_MAX_SPEED
        
        # Update position
        self.player_pos += self.player_vel
        
        # Update angle to face velocity direction
        if speed > 0.1:
            self.player_angle = math.atan2(self.player_vel[1], self.player_vel[0])

        # Update invulnerability
        if self.invulnerable_timer > 0:
            self.invulnerable_timer -= 1

    def _update_mining(self, space_held):
        is_mining = False
        if not space_held:
            return is_mining

        target_asteroid = None
        min_dist = self.MINING_RANGE

        for asteroid in self.asteroids:
            vec_to_asteroid = asteroid["pos"] - self.player_pos
            dist = np.linalg.norm(vec_to_asteroid)

            if 0 < dist < min_dist:
                angle_to_asteroid = math.atan2(vec_to_asteroid[1], vec_to_asteroid[0])
                angle_diff = abs((self.player_angle - angle_to_asteroid + math.pi) % (2 * math.pi) - math.pi)

                if angle_diff < self.MINING_ANGLE:
                    min_dist = dist
                    target_asteroid = asteroid
        
        if target_asteroid:
            is_mining = True
            # // Mining sound effect
            mined_amount = min(target_asteroid["ore"], self.MINING_RATE)
            target_asteroid["ore"] -= mined_amount
            self.score += mined_amount
            
            # Create ore particles
            for _ in range(2):
                self._create_particle(
                    pos=target_asteroid["pos"].copy(),
                    vel=(self.player_pos - target_asteroid["pos"]) / 30,
                    ttl=30,
                    radius=3,
                    color=self.COLOR_ORE
                )
        
        return is_mining

    def _update_asteroids(self):
        destroyed_asteroids = []
        for asteroid in self.asteroids:
            # Move asteroids
            asteroid["pos"] += asteroid["vel"]

            # Despawn if too far
            if np.linalg.norm(asteroid["pos"] - self.player_pos) > self.ASTEROID_DESPAWN_RADIUS:
                destroyed_asteroids.append(asteroid)
                continue

            if asteroid["ore"] <= 0:
                destroyed_asteroids.append(asteroid)
                # // Asteroid destruction sound
                self._create_explosion(asteroid["pos"], int(asteroid["radius"] / 2), self.COLOR_ASTEROID)
                if asteroid["is_large"]:
                    self.step_reward += 1 # Bonus for destroying large asteroid

        # Remove depleted/despawned asteroids
        self.asteroids = [a for a in self.asteroids if a not in destroyed_asteroids]

        # Spawn new asteroids if needed
        while len(self.asteroids) < self.MIN_ASTEROIDS:
            self._spawn_asteroid()

    def _update_lasers(self):
        # Spawn new lasers
        laser_chance = min(self.MAX_LASER_CHANCE, self.BASE_LASER_CHANCE + self.score * self.LASER_CHANCE_PER_ORE)
        if self.np_random.random() < laser_chance:
            self._spawn_laser()

        active_lasers = []
        for laser in self.lasers:
            laser["pos"] += laser["vel"]
            laser["ttl"] -= 1
            
            # Check collision with player
            if self.invulnerable_timer == 0 and np.linalg.norm(laser["pos"] - self.player_pos) < self.PLAYER_SIZE:
                self.lives -= 1
                self.invulnerable_timer = self.INVULNERABILITY_FRAMES
                self._create_explosion(self.player_pos, 20, self.COLOR_EXPLOSION)
                # // Player hit/explosion sound
                laser["ttl"] = 0 # Mark for removal
            
            if laser["ttl"] > 0:
                active_lasers.append(laser)
        
        self.lasers = active_lasers

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"] += p["vel"]
            p["ttl"] -= 1
            if p["ttl"] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _calculate_reward(self, is_mining):
        # This is a bit of a workaround to integrate event-based rewards
        # from other methods into the step reward.
        reward = self.step_reward
        self.step_reward = 0 # Reset for next step

        if is_mining:
            reward += 0.1 * self.MINING_RATE # From brief: +0.1 per ore
        else:
            reward -= 0.02 # Penalty for not mining
        
        return reward

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.score = self.WIN_SCORE
            return True
        if self.lives <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_offset = np.array([self.WIDTH / 2, self.HEIGHT / 2]) - self.player_pos

        # Render stars (parallax)
        for star in self.stars:
            star_pos = (star["pos"] + cam_offset / star["layer"])
            # Wrap around screen
            star_pos[0] = star_pos[0] % self.WIDTH
            star_pos[1] = star_pos[1] % self.HEIGHT
            pygame.draw.circle(self.screen, (star["layer"]*40,)*3, star_pos, star["layer"] * 0.5)

        # Render asteroids
        for asteroid in self.asteroids:
            self._draw_asteroid(asteroid, cam_offset)

        # Render mining beam
        if any(p['color'] == self.COLOR_ORE for p in self.particles):
            self._draw_mining_beam(cam_offset)
            
        # Render player
        self._draw_player()

        # Render lasers
        for laser in self.lasers:
            start_pos = (laser["pos"] + cam_offset).astype(int)
            end_pos = (laser["pos"] - laser["vel"] * 2 + cam_offset).astype(int)
            pygame.draw.aaline(self.screen, self.COLOR_LASER, start_pos, end_pos, 2)
        
        # Render particles
        for p in self.particles:
            pos = (p["pos"] + cam_offset).astype(int)
            radius = int(p["radius"] * (p["ttl"] / p["initial_ttl"]))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p["color"])
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, p["color"])

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"ORE: {int(self.score)} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        life_icon = self._get_player_points(size=8, angle=-math.pi/2)
        for i in range(self.lives):
            offset = np.array([self.WIDTH - 20 - i * 25, 20])
            points = [p + offset for p in life_icon]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)

        # Game Over message
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "MISSION COMPLETE"
                color = self.COLOR_ORE
            else:
                msg = "GAME OVER"
                color = self.COLOR_LASER
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives
        }

    # --- Helper methods for spawning and drawing ---

    def _spawn_asteroid(self, initial_spawn=False):
        # Spawn away from the center
        angle = self.np_random.uniform(0, 2 * math.pi)
        if initial_spawn:
            dist = self.np_random.uniform(100, self.ASTEROID_SPAWN_RADIUS)
        else:
            dist = self.ASTEROID_SPAWN_RADIUS
        
        pos = self.player_pos + np.array([math.cos(angle) * dist, math.sin(angle) * dist])

        # Check for overlap
        for a in self.asteroids:
            if np.linalg.norm(pos - a["pos"]) < a["radius"] + self.ASTEROID_MAX_SIZE:
                return # Don't spawn if too close to another

        radius = self.np_random.uniform(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE)
        ore = radius * 2
        vel = (self.player_pos - pos) / self.np_random.uniform(300, 500) # Move slowly towards player area
        
        # Create jagged shape
        num_points = self.np_random.integers(7, 12)
        points = []
        for i in range(num_points):
            a = 2 * math.pi * i / num_points
            r = radius + self.np_random.uniform(-radius*0.2, radius*0.2)
            points.append((math.cos(a) * r, math.sin(a) * r))

        self.asteroids.append({
            "pos": pos,
            "radius": radius,
            "ore": ore,
            "vel": vel,
            "shape_points": points,
            "is_large": radius > self.LARGE_ASTEROID_THRESHOLD
        })

    def _spawn_laser(self):
        # Spawn from off-screen edge
        edge = self.np_random.integers(0, 4)
        if edge == 0: # top
            pos = np.array([self.np_random.uniform(0, self.WIDTH), -20])
        elif edge == 1: # bottom
            pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20])
        elif edge == 2: # left
            pos = np.array([-20, self.np_random.uniform(0, self.HEIGHT)])
        elif edge == 3: # right
            pos = np.array([self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT)])
        
        pos = pos - np.array([self.WIDTH/2, self.HEIGHT/2]) + self.player_pos
        
        # Target player or a large asteroid
        target_pos = self.player_pos
        large_asteroids = [a for a in self.asteroids if a["is_large"]]
        if large_asteroids and self.np_random.random() < 0.25:
            target_pos = self.np_random.choice(large_asteroids)["pos"]

        direction = (target_pos - pos)
        dist = np.linalg.norm(direction)
        if dist == 0: return
        
        vel = (direction / dist) * self.LASER_SPEED
        # // Laser fire sound
        self.lasers.append({"pos": pos, "vel": vel, "ttl": 150})

    def _create_particle(self, pos, vel, ttl, radius, color):
        self.particles.append({
            "pos": pos,
            "vel": vel + self.np_random.uniform(-0.5, 0.5, 2),
            "ttl": ttl,
            "initial_ttl": ttl,
            "radius": radius,
            "color": color
        })

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self._create_particle(
                pos=pos.copy(),
                vel=vel,
                ttl=self.np_random.integers(20, 40),
                radius=self.np_random.uniform(2, 4),
                color=color
            )

    def _get_player_points(self, size, angle):
        p1 = (size, 0)
        p2 = (-size * 0.5, -size * 0.7)
        p3 = (-size * 0.5, size * 0.7)
        
        # Rotate points
        c, s = math.cos(angle), math.sin(angle)
        rot_matrix = np.array([[c, -s], [s, c]])
        
        p1_rot = np.dot(rot_matrix, p1)
        p2_rot = np.dot(rot_matrix, p2)
        p3_rot = np.dot(rot_matrix, p3)
        
        return [p1_rot, p2_rot, p3_rot]

    def _draw_player(self):
        screen_center = np.array([self.WIDTH / 2, self.HEIGHT / 2])
        points = self._get_player_points(self.PLAYER_SIZE, self.player_angle)
        points = [p + screen_center for p in points]
        
        # Flashing when invulnerable
        is_visible = self.invulnerable_timer == 0 or (self.invulnerable_timer // 3) % 2 == 0
        if is_visible:
            color = self.COLOR_PLAYER if self.invulnerable_timer == 0 else self.COLOR_PLAYER_INVULN
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_asteroid(self, asteroid, cam_offset):
        points = [p + asteroid["pos"] + cam_offset for p in asteroid["shape_points"]]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

    def _draw_mining_beam(self, cam_offset):
        # This is a visual effect, actual mining logic is elsewhere
        start_pos = np.array([self.WIDTH/2, self.HEIGHT/2])
        end_point_dir = np.array([math.cos(self.player_angle), math.sin(self.player_angle)])
        
        # Find intersection with a mined asteroid for a nice visual effect
        end_pos = start_pos + end_point_dir * self.MINING_RANGE
        
        for p in self.particles:
            if p['color'] == self.COLOR_ORE:
                end_pos = p['pos'] + cam_offset
                break
        
        # Draw translucent cone
        angle_l = self.player_angle - self.MINING_ANGLE
        angle_r = self.player_angle + self.MINING_ANGLE
        p_l = start_pos + np.array([math.cos(angle_l), math.sin(angle_l)]) * np.linalg.norm(end_pos - start_pos)
        p_r = start_pos + np.array([math.cos(angle_r), math.sin(angle_r)]) * np.linalg.norm(end_pos - start_pos)

        beam_surface = self.screen.convert_alpha()
        beam_surface.fill((0,0,0,0))
        pygame.gfxdraw.filled_trigon(beam_surface, int(start_pos[0]), int(start_pos[1]), int(p_l[0]), int(p_l[1]), int(p_r[0]), int(p_r[1]), (*self.COLOR_ORE, 70))
        self.screen.blit(beam_surface, (0,0))

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
        self.step_reward = 0 # Initialize for test
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
    # Set the video driver to a dummy one for headless execution if not rendering to screen
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # --- For manual play ---
    # Set to True to play manually with keyboard
    manual_play = True
    if manual_play:
        # Re-initialize pygame with a display
        pygame.display.init()
        pygame.display.set_caption("Asteroid Miner")
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))

        obs, info = env.reset()
        terminated = False
        
        # Game loop for manual play
        while not terminated:
            # Action defaults
            movement = 0 # none
            space = 0 # released
            shift = 0 # released

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

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
            
            # Render the observation from the environment to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30) # Control the frame rate

    # --- For agent training ---
    else:
        obs, info = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"Episode finished. Final Info: {info}")
                obs, info = env.reset()
    
    env.close()