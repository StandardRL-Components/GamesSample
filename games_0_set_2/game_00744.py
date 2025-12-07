
# Generated: 2025-08-27T14:37:42.435326
# Source Brief: brief_00744.md
# Brief Index: 744

        
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
        "Controls: Arrow keys to move. Hold Space for a hyper-boost."
    )

    game_description = (
        "Pilot a spaceship, dodging deadly mines and collecting valuable asteroids for points. "
        "Collect 25 asteroids to win, but be careful - you only have 3 lives!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # --- Colors and Fonts ---
        self.COLOR_BG = (10, 15, 25)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.COLOR_MINE = (255, 50, 50)
        self.COLOR_MINE_GLOW = (255, 50, 50, 70)
        self.COLOR_ASTEROID = (150, 150, 160)
        self.COLOR_TEXT = (220, 220, 230)
        self.FONT_UI = pygame.font.Font(None, 28)
        self.FONT_MSG = pygame.font.Font(None, 50)

        # --- Game Constants ---
        self.MAX_STEPS = 2500
        self.NUM_ASTEROIDS = 25
        self.NUM_MINES = 10
        self.PLAYER_SPEED = 3.5
        self.PLAYER_BOOST_MULTIPLIER = 2.0
        self.PLAYER_RADIUS = 10
        self.ASTEROID_RADIUS = 12
        self.MINE_RADIUS = 8
        self.INITIAL_LIVES = 3
        self.INVINCIBILITY_FRAMES = 60 # 2 seconds at 30fps

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_vel = None
        self.player_lives = 0
        self.invincibility_timer = 0
        self.asteroids = []
        self.mines = []
        self.particles = []
        self.asteroids_collected = 0
        self.victory = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        self.player_pos = np.array([self.width / 2, self.height / 2], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.player_lives = self.INITIAL_LIVES
        self.invincibility_timer = 0
        
        self.asteroids_collected = 0
        self._spawn_asteroids()
        self._spawn_mines()
        
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Calculate continuous rewards (before movement) ---
        dist_asteroid_before = self._get_closest_distance(self.player_pos, self.asteroids)
        dist_mine_before = self._get_closest_distance(self.player_pos, self.mines)

        # --- Unpack Action and Update Player ---
        movement, space_held, _ = action
        speed = self.PLAYER_SPEED * (self.PLAYER_BOOST_MULTIPLIER if space_held else 1.0)
        
        if movement == 1: # Up
            self.player_vel[1] = -speed
        elif movement == 2: # Down
            self.player_vel[1] = speed
        elif movement == 3: # Left
            self.player_vel[0] = -speed
        elif movement == 4: # Right
            self.player_vel[0] = speed
        
        # Apply movement and friction
        self.player_pos += self.player_vel
        self.player_vel *= 0.85 # Friction

        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.width)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.height)

        # --- Update Game World ---
        self._update_mines()
        self._update_particles()
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1

        # --- Handle Collisions and Event Rewards ---
        event_reward = 0
        # Player vs Asteroids
        for asteroid in self.asteroids:
            if not asteroid['collected']:
                dist = np.linalg.norm(self.player_pos - asteroid['pos'])
                if dist < self.PLAYER_RADIUS + self.ASTEROID_RADIUS:
                    asteroid['collected'] = True
                    self.asteroids_collected += 1
                    event_reward += 10
                    # Sound: sfx_collect_asteroid.wav
                    self._create_particles(asteroid['pos'], self.COLOR_ASTEROID, 20, 2.0)

        # Player vs Mines
        if self.invincibility_timer == 0:
            for mine in self.mines:
                dist = np.linalg.norm(self.player_pos - mine['pos'])
                if dist < self.PLAYER_RADIUS + self.MINE_RADIUS:
                    self.player_lives -= 1
                    event_reward -= 50
                    self.invincibility_timer = self.INVINCIBILITY_FRAMES
                    # Sound: sfx_explosion.wav
                    self._create_particles(self.player_pos, self.COLOR_MINE, 40, 4.0)
                    break
        
        self.score += event_reward

        # --- Calculate continuous rewards (after movement) ---
        dist_asteroid_after = self._get_closest_distance(self.player_pos, self.asteroids)
        dist_mine_after = self._get_closest_distance(self.player_pos, self.mines)
        
        # Reward for getting closer to asteroids, penalty for getting closer to mines
        continuous_reward = 0
        if dist_asteroid_before > 0: # Avoid division by zero or issues on first frame
            continuous_reward += (dist_asteroid_before - dist_asteroid_after) * 0.01
        if dist_mine_before > 0:
            continuous_reward -= (dist_mine_before - dist_mine_after) * 0.05

        reward = event_reward + continuous_reward

        # --- Check Termination Conditions ---
        self.steps += 1
        terminated = False
        if self.player_lives <= 0:
            terminated = True
            self.game_over = True
        elif self.asteroids_collected >= self.NUM_ASTEROIDS:
            terminated = True
            self.game_over = True
            self.victory = True
            reward += 100 # Victory bonus
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.clock.tick(30)
        self.screen.fill(self.COLOR_BG)
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "asteroids_collected": self.asteroids_collected,
        }

    # --- Helper Methods ---

    def _spawn_asteroids(self):
        self.asteroids = []
        for _ in range(self.NUM_ASTEROIDS):
            while True:
                pos = np.array([
                    self.np_random.uniform(self.ASTEROID_RADIUS, self.width - self.ASTEROID_RADIUS),
                    self.np_random.uniform(self.ASTEROID_RADIUS, self.height - self.ASTEROID_RADIUS)
                ])
                if np.linalg.norm(pos - self.player_pos) > 100: # Don't spawn on player
                    break
            
            # Create a random-looking polygon for the asteroid
            num_vertices = self.np_random.integers(5, 9)
            angles = np.sort(self.np_random.uniform(0, 2 * np.pi, num_vertices))
            radii = self.np_random.uniform(self.ASTEROID_RADIUS * 0.7, self.ASTEROID_RADIUS * 1.3, num_vertices)
            points = [ (r * np.cos(a), r * np.sin(a)) for r, a in zip(radii, angles) ]

            self.asteroids.append({'pos': pos, 'collected': False, 'points': points})

    def _spawn_mines(self):
        self.mines = []
        mine_speed = 0.02
        for _ in range(self.NUM_MINES):
            while True:
                center = np.array([
                    self.np_random.uniform(50, self.width - 50),
                    self.np_random.uniform(50, self.height - 50)
                ])
                if np.linalg.norm(center - self.player_pos) > 150:
                    break
            
            radius = self.np_random.uniform(30, 80)
            angle = self.np_random.uniform(0, 2 * np.pi)
            speed = self.np_random.uniform(mine_speed * 0.8, mine_speed * 1.2) * self.np_random.choice([-1, 1])
            pos = center + np.array([np.cos(angle) * radius, np.sin(angle) * radius])
            self.mines.append({'pos': pos, 'center': center, 'radius': radius, 'angle': angle, 'speed': speed})

    def _update_mines(self):
        # Difficulty scaling
        base_speed_increase = 0.00005 * self.steps # 0.05 per 500 steps is 0.0001 per step. Let's make it more gradual.
        for mine in self.mines:
            current_speed = mine['speed']
            direction = np.sign(current_speed)
            new_speed = direction * (abs(current_speed) + base_speed_increase)
            mine['angle'] += new_speed
            mine['pos'] = mine['center'] + np.array([np.cos(mine['angle']) * mine['radius'], np.sin(mine['angle']) * mine['radius']])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] *= 0.98

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * np.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            vel = np.array([np.cos(angle), np.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'radius': self.np_random.uniform(2, 5),
                'color': color
            })
    
    def _get_closest_distance(self, pos, entity_list):
        min_dist = float('inf')
        active_entities = [e for e in entity_list if not e.get('collected', False)]
        if not active_entities:
            return 0
        for entity in active_entities:
            dist = np.linalg.norm(pos - entity['pos'])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _render_game(self):
        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'].astype(int), int(p['radius']))
        
        # Mines
        for mine in self.mines:
            pos_int = mine['pos'].astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.MINE_RADIUS, self.COLOR_MINE)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.MINE_RADIUS, self.COLOR_MINE)
            # Glow effect
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.MINE_RADIUS + 3, self.COLOR_MINE_GLOW)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.MINE_RADIUS + 6, self.COLOR_MINE_GLOW)
        
        # Asteroids
        for asteroid in self.asteroids:
            if not asteroid['collected']:
                points_abs = [ (p[0] + asteroid['pos'][0], p[1] + asteroid['pos'][1]) for p in asteroid['points'] ]
                pygame.draw.polygon(self.screen, self.COLOR_ASTEROID, points_abs)

        # Player Ship
        if self.player_lives > 0:
            pos_int = self.player_pos.astype(int)
            # Invincibility flash
            if self.invincibility_timer > 0 and self.invincibility_timer % 10 < 5:
                # Do not draw player to create a flashing effect
                pass
            else:
                # Body
                angle = math.atan2(self.player_vel[1], self.player_vel[0]) if np.linalg.norm(self.player_vel) > 0.1 else -np.pi/2
                p1 = (pos_int[0] + self.PLAYER_RADIUS * math.cos(angle), pos_int[1] + self.PLAYER_RADIUS * math.sin(angle))
                p2 = (pos_int[0] + self.PLAYER_RADIUS * 0.7 * math.cos(angle + 2.5), pos_int[1] + self.PLAYER_RADIUS * 0.7 * math.sin(angle + 2.5))
                p3 = (pos_int[0] + self.PLAYER_RADIUS * 0.7 * math.cos(angle - 2.5), pos_int[1] + self.PLAYER_RADIUS * 0.7 * math.sin(angle - 2.5))
                pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1, p2, p3])
                
                # Glow
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER_GLOW)
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS + 4, self.COLOR_PLAYER_GLOW)


    def _render_ui(self):
        # Score
        score_text = self.FONT_UI.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.width - score_text.get_width() - 10, 10))
        
        # Lives
        for i in range(self.player_lives):
            p1 = (20 + i * 25, 25)
            p2 = (12 + i * 25, 10)
            p3 = (28 + i * 25, 10)
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1, p2, p3])

        # Asteroid Counter
        asteroid_text = self.FONT_UI.render(f"ASTEROIDS: {self.asteroids_collected}/{self.NUM_ASTEROIDS}", True, self.COLOR_TEXT)
        self.screen.blit(asteroid_text, (self.width // 2 - asteroid_text.get_width() // 2, 10))

        # Game Over / Victory Message
        if self.game_over:
            if self.victory:
                msg = "VICTORY!"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_MINE
            
            msg_surf = self.FONT_MSG.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Setup a display window
    pygame.display.set_caption("Asteroid Dodger")
    display_screen = pygame.display.set_mode((env.width, env.height))

    total_reward = 0
    
    # Game loop
    while not done:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

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
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Pygame Event Handling & Rendering ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # Blit the observation from the env to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()