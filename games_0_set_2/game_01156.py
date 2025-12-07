
# Generated: 2025-08-27T16:13:40.071981
# Source Brief: brief_01156.md
# Brief Index: 1156

        
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
        "Controls: ↑↓←→ to move. Hold space near an asteroid to mine minerals."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship through an asteroid field, dodging enemies and collecting minerals to reach a target yield."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Screen dimensions
    WIDTH, HEIGHT = 640, 400

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (200, 50, 50)
    COLOR_ASTEROID_DARK = (80, 80, 90)
    COLOR_ASTEROID_LIGHT = (180, 180, 190)
    COLOR_MINERAL_SPARK = (255, 220, 50)
    COLOR_EXPLOSION = (255, 150, 0)
    COLOR_UI_TEXT = (220, 220, 240)

    # Game parameters
    PLAYER_ACCELERATION = 0.4
    PLAYER_FRICTION = 0.96
    PLAYER_MAX_SPEED = 6
    PLAYER_RADIUS = 12
    PLAYER_INVINCIBILITY_FRAMES = 90  # 3 seconds at 30fps

    INITIAL_ASTEROIDS = 15
    MIN_ASTEROIDS = 10
    ASTEROID_MIN_RADIUS = 15
    ASTEROID_MAX_RADIUS = 40
    
    INITIAL_ENEMIES = 3
    ENEMY_RADIUS = 10
    ENEMY_BASE_SPEED = 1.0
    
    MINING_RANGE = 80
    MINING_RATE = 0.5  # minerals per frame

    WIN_MINERAL_COUNT = 100
    MAX_STEPS = 5000
    INITIAL_LIVES = 3
    
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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Etc...        
        
        # Initialize state variables
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0, 0], dtype=np.float32)
        self.player_lives = self.INITIAL_LIVES
        self.player_invincible_timer = 0

        # Game state
        self.steps = 0
        self.score = 0
        self.mineral_count = 0
        self.game_over = False
        self.game_outcome = "" # "VICTORY" or "DEFEAT"
        
        # Entities
        self.enemy_speed = self.ENEMY_BASE_SPEED
        self.asteroids = [self._create_asteroid() for _ in range(self.INITIAL_ASTEROIDS)]
        self.enemies = [self._create_enemy() for _ in range(self.INITIAL_ENEMIES)]
        
        # Effects
        self.particles = [] # For mining, explosions, etc.
        
        # Background
        self.stars = self._create_stars(200)
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean - unused

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        # 1. Handle player input and update player state
        self._update_player(movement)
        
        # 2. Update game world
        self._update_enemies()
        self._update_particles()
        
        # 3. Handle interactions and rewards
        minerals_collected, depleted_asteroid_reward = self._handle_mining(space_held)
        if minerals_collected > 0:
            self.mineral_count += minerals_collected
            reward += 0.1 * minerals_collected # Continuous reward for mining
        else:
            reward -= 0.01 # Small penalty for inactivity

        reward += depleted_asteroid_reward # Event-based reward for finishing an asteroid

        self._handle_collisions() # Modifies player lives

        # 4. Respawn entities
        self._respawn_asteroids()

        # 5. Update game state
        self.steps += 1
        self._update_difficulty()
        
        # 6. Check for termination
        terminated = self._check_termination()
        if terminated:
            if self.game_outcome == "VICTORY":
                reward += 100
            elif self.game_outcome == "DEFEAT":
                reward -= 100
        
        self.score += reward

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    # --- Entity Creation and Management ---

    def _create_asteroid(self, on_screen=True):
        if on_screen:
            pos = self.np_random.uniform([0, 0], [self.WIDTH, self.HEIGHT])
        else: # Spawn off-screen
            side = self.np_random.integers(4)
            if side == 0: # Top
                pos = [self.np_random.uniform(0, self.WIDTH), -self.ASTEROID_MAX_RADIUS]
            elif side == 1: # Bottom
                pos = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ASTEROID_MAX_RADIUS]
            elif side == 2: # Left
                pos = [-self.ASTEROID_MAX_RADIUS, self.np_random.uniform(0, self.HEIGHT)]
            else: # Right
                pos = [self.WIDTH + self.ASTEROID_MAX_RADIUS, self.np_random.uniform(0, self.HEIGHT)]

        radius = self.np_random.uniform(self.ASTEROID_MIN_RADIUS, self.ASTEROID_MAX_RADIUS)
        minerals = int((radius / self.ASTEROID_MAX_RADIUS) * 100)
        return {
            "pos": np.array(pos, dtype=np.float32),
            "radius": int(radius),
            "minerals": minerals,
            "initial_minerals": max(1, minerals)
        }

    def _create_enemy(self):
        start_x = self.np_random.choice([-self.ENEMY_RADIUS, self.WIDTH + self.ENEMY_RADIUS])
        start_y = self.np_random.uniform(self.HEIGHT * 0.2, self.HEIGHT * 0.8)
        return {
            "pos": np.array([start_x, start_y], dtype=np.float32),
            "vel_dir": 1 if start_x < 0 else -1,
            "y_base": start_y,
            "phase": self.np_random.uniform(0, 2 * math.pi),
            "amplitude": self.np_random.uniform(self.HEIGHT * 0.1, self.HEIGHT * 0.3),
            "frequency": self.np_random.uniform(0.005, 0.01)
        }
        
    def _create_stars(self, count):
        return [
            (
                self.np_random.uniform(0, self.WIDTH),
                self.np_random.uniform(0, self.HEIGHT),
                self.np_random.uniform(0.5, 1.5),
                self.np_random.uniform(50, 150) # Brightness
            )
            for _ in range(count)
        ]

    def _respawn_asteroids(self):
        while len(self.asteroids) < self.MIN_ASTEROIDS:
            self.asteroids.append(self._create_asteroid(on_screen=False))

    # --- Game Logic Updates ---

    def _update_player(self, movement):
        if movement == 1: # Up
            self.player_vel[1] -= self.PLAYER_ACCELERATION
        elif movement == 2: # Down
            self.player_vel[1] += self.PLAYER_ACCELERATION
        if movement == 3: # Left
            self.player_vel[0] -= self.PLAYER_ACCELERATION
        elif movement == 4: # Right
            self.player_vel[0] += self.PLAYER_ACCELERATION
            
        self.player_vel *= self.PLAYER_FRICTION
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = (self.player_vel / speed) * self.PLAYER_MAX_SPEED
            
        self.player_pos += self.player_vel
        
        # Screen wrap-around
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT
        
        if self.player_invincible_timer > 0:
            self.player_invincible_timer -= 1

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy['pos'][0] += self.enemy_speed * enemy['vel_dir']
            enemy['pos'][1] = enemy['y_base'] + enemy['amplitude'] * math.sin(enemy['pos'][0] * enemy['frequency'] + enemy['phase'])

            if enemy['vel_dir'] == 1 and enemy['pos'][0] > self.WIDTH + self.ENEMY_RADIUS * 2:
                enemy['pos'][0] = -self.ENEMY_RADIUS * 2
            elif enemy['vel_dir'] == -1 and enemy['pos'][0] < -self.ENEMY_RADIUS * 2:
                enemy['pos'][0] = self.WIDTH + self.ENEMY_RADIUS * 2

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p.get('radius_decay', 0) > 0:
                p['radius'] -= p['radius_decay']

    def _handle_mining(self, space_held):
        minerals_collected = 0
        depleted_reward = 0
        
        if not space_held:
            return 0, 0

        # Find closest asteroid in range
        target_asteroid = None
        min_dist = float('inf')
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < min_dist and dist < self.MINING_RANGE + asteroid['radius']:
                min_dist = dist
                target_asteroid = asteroid
        
        if target_asteroid and target_asteroid['minerals'] > 0:
            # Mine it
            mined_amount = self.MINING_RATE
            target_asteroid['minerals'] -= mined_amount
            minerals_collected = mined_amount
            
            # Create mining particles
            # Sound effect: sfx_mining_laser
            for _ in range(2):
                angle = self.np_random.uniform(0, 2 * math.pi)
                offset = self.np_random.uniform(0, target_asteroid['radius'])
                start_pos = target_asteroid['pos'] + np.array([math.cos(angle) * offset, math.sin(angle) * offset])
                vel = (self.player_pos - start_pos) / 30.0 + self.np_random.uniform(-0.5, 0.5, 2)
                self.particles.append({
                    "pos": start_pos, "vel": vel, "life": 30, 
                    "color": self.COLOR_MINERAL_SPARK, "radius": self.np_random.uniform(1, 3)
                })
            
            if target_asteroid['minerals'] <= 0:
                # Asteroid depleted
                # Sound effect: sfx_asteroid_destroyed
                depleted_reward = 1.0 if target_asteroid['radius'] > (self.ASTEROID_MIN_RADIUS + self.ASTEROID_MAX_RADIUS) / 2 else 0.5
                self.asteroids.remove(target_asteroid)

        return minerals_collected, depleted_reward

    def _handle_collisions(self):
        if self.player_invincible_timer > 0:
            return

        for enemy in self.enemies:
            dist = np.linalg.norm(self.player_pos - enemy['pos'])
            if dist < self.PLAYER_RADIUS + self.ENEMY_RADIUS:
                # Collision occurred
                # Sound effect: sfx_player_hit
                self.player_lives -= 1
                self.player_invincible_timer = self.PLAYER_INVINCIBILITY_FRAMES
                self._create_explosion(self.player_pos, 20)
                # Reset player position to center for fairness
                self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
                self.player_vel = np.array([0, 0], dtype=np.float32)
                return # Only one collision per frame

    def _create_explosion(self, pos, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(), "vel": vel, "life": self.np_random.integers(20, 40),
                "color": self.COLOR_EXPLOSION, "radius": self.np_random.uniform(3, 6),
                "radius_decay": 0.1
            })

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 500 == 0:
            self.enemy_speed += 0.05
    
    def _check_termination(self):
        if self.game_over:
            return True
        if self.mineral_count >= self.WIN_MINERAL_COUNT:
            self.game_over = True
            self.game_outcome = "VICTORY"
        elif self.player_lives <= 0:
            self.game_over = True
            self.game_outcome = "DEFEAT"
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.game_outcome = "DEFEAT" # Time out is a loss
        return self.game_over
        
    # --- Rendering ---

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_stars()
        self._render_particles()
        self._render_asteroids()
        self._render_enemies()
        self._render_player()

    def _render_stars(self):
        for x, y, size, brightness in self.stars:
            color_val = int(brightness)
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), (x, y), size)
            
    def _render_asteroids(self):
        for asteroid in self.asteroids:
            p = asteroid['minerals'] / asteroid['initial_minerals']
            color = tuple(int(self.COLOR_ASTEROID_DARK[i] + (self.COLOR_ASTEROID_LIGHT[i] - self.COLOR_ASTEROID_DARK[i]) * p) for i in range(3))
            pos_int = (int(asteroid['pos'][0]), int(asteroid['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], asteroid['radius'], color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], asteroid['radius'], color)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos_int = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            r = self.ENEMY_RADIUS
            points = [
                (pos_int[0], pos_int[1] - r), (pos_int[0] + r, pos_int[1]),
                (pos_int[0], pos_int[1] + r), (pos_int[0] - r, pos_int[1]),
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY_GLOW)
            
    def _render_player(self):
        pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
        r = self.PLAYER_RADIUS
        
        # Blinking effect when invincible
        if self.player_invincible_timer > 0 and self.steps % 10 < 5:
            return

        # Glow effect
        glow_r = r + 4 + int(abs(math.sin(self.steps * 0.1)) * 4)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], glow_r, self.COLOR_PLAYER_GLOW)
        
        # Main body (triangle)
        angle = math.atan2(self.player_vel[1], self.player_vel[0]) if np.linalg.norm(self.player_vel) > 0.1 else -math.pi / 2
        p1 = (pos_int[0] + r * math.cos(angle), pos_int[1] + r * math.sin(angle))
        p2 = (pos_int[0] + r * math.cos(angle + 2.2), pos_int[1] + r * math.sin(angle + 2.2))
        p3 = (pos_int[0] + r * math.cos(angle - 2.2), pos_int[1] + r * math.sin(angle - 2.2))
        
        pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_PLAYER)
        pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            if p['radius'] > 0:
                pos_int = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.draw.circle(self.screen, p['color'], pos_int, max(0, int(p['radius'])))

    def _render_ui(self):
        # Mineral Count
        mineral_text = self.font_ui.render(f"MINERALS: {int(self.mineral_count)} / {self.WIN_MINERAL_COUNT}", True, self.COLOR_UI_TEXT)
        self.screen.blit(mineral_text, (10, 10))
        
        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.player_lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            text = self.font_game_over.render(self.game_outcome, True, self.COLOR_UI_TEXT)
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            overlay.blit(text, text_rect)
            self.screen.blit(overlay, (0, 0))

    # --- Gymnasium Interface Helpers ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "mineral_count": int(self.mineral_count),
            "lives": self.player_lives,
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
        
        # print("✓ Implementation validated successfully")