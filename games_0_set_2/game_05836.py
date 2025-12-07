import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Hold space to activate your mining laser."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Mine asteroids for ore while dodging enemy patrols in a top-down space arcade."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
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
        
        # Game constants
        self.PLAYER_SPEED = 4
        self.PLAYER_RADIUS = 10
        self.ENEMY_RADIUS = 8
        self.INITIAL_ENEMY_SPEED = 1.5
        self.MINING_RANGE = 100
        self.MINING_RATE = 1
        self.MAX_STEPS = 5000
        self.WIN_ORE = 500
        self.INITIAL_LIVES = 3
        self.INITIAL_ASTEROIDS = 15
        self.INITIAL_ENEMIES = 4

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (255, 50, 50, 50)
        self.COLOR_ASTEROID_LOW = (100, 100, 110)
        self.COLOR_ASTEROID_MID = (140, 140, 150)
        self.COLOR_ASTEROID_HIGH = (200, 190, 150)
        self.COLOR_ORE_PARTICLE = (255, 220, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_LASER = (255, 255, 255, 150)

        # Fonts
        try:
            self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
            self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 24)
            self.font_game_over = pygame.font.Font(None, 50)

        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_lives = self.INITIAL_LIVES
        self.ore = 0
        self.mining_target = None
        
        self.enemy_speed_multiplier = 1.0

        self.stars = self._create_stars(200)
        self.asteroids = [self._create_asteroid() for _ in range(self.INITIAL_ASTEROIDS)]
        self.enemies = [self._create_enemy() for _ in range(self.INITIAL_ENEMIES)]
        self.particles = []

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.02  # Small penalty for each step
        
        # Unpack factorized action
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Update game logic
        self._handle_player_movement(movement)
        
        ore_mined, asteroid_destroyed = self._handle_mining(space_held)
        if ore_mined > 0:
            reward += ore_mined
        if asteroid_destroyed:
            reward += 10

        self._update_enemies()
        self._update_particles()
        
        if self._check_player_enemy_collisions():
            self.player_lives -= 1
            reward -= 100 if self.player_lives <= 0 else 25
            self._create_explosion(self.player_pos)
            if self.player_lives > 0:
                self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        
        self._spawn_new_asteroids()
        self._update_difficulty()

        self.steps += 1
        terminated = self._check_termination()

        if terminated and self.game_won:
            reward += 100

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        direction = np.array([0, 0], dtype=np.float32)
        if movement == 1: direction[1] = -1 # Up
        elif movement == 2: direction[1] = 1 # Down
        elif movement == 3: direction[0] = -1 # Left
        elif movement == 4: direction[0] = 1 # Right
        
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        
        self.player_pos += direction * self.PLAYER_SPEED
        
        # World wrapping
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT

    def _handle_mining(self, space_held):
        ore_mined = 0
        asteroid_destroyed = False
        if not space_held:
            self.mining_target = None
            return ore_mined, asteroid_destroyed

        # Find closest asteroid in range
        target_asteroid = None
        min_dist = self.MINING_RANGE
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos']) - asteroid['radius']
            if dist < min_dist:
                min_dist = dist
                target_asteroid = asteroid
        
        self.mining_target = target_asteroid

        if target_asteroid:
            mined_amount = min(target_asteroid['ore'], self.MINING_RATE)
            target_asteroid['ore'] -= mined_amount
            self.ore += mined_amount
            ore_mined += mined_amount
            
            # Create ore particles
            if self.steps % 3 == 0:
                self._create_ore_particle(target_asteroid['pos'])

            if target_asteroid['ore'] <= 0:
                self.asteroids.remove(target_asteroid)
                asteroid_destroyed = True
                self.mining_target = None

        return ore_mined, asteroid_destroyed

    def _update_enemies(self):
        for enemy in self.enemies:
            if enemy['patrol_type'] == 'circular':
                enemy['angle'] += enemy['angular_speed'] * self.enemy_speed_multiplier
                offset_x = math.cos(enemy['angle']) * enemy['patrol_radius']
                offset_y = math.sin(enemy['angle']) * enemy['patrol_radius']
                enemy['pos'][0] = enemy['center'][0] + offset_x
                enemy['pos'][1] = enemy['center'][1] + offset_y
            else: # linear
                enemy['pos'] += enemy['direction'] * self.INITIAL_ENEMY_SPEED * self.enemy_speed_multiplier
                if np.linalg.norm(enemy['pos'] - enemy['start']) >= enemy['patrol_radius']:
                    enemy['direction'] *= -1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _check_player_enemy_collisions(self):
        for enemy in self.enemies:
            dist = np.linalg.norm(self.player_pos - enemy['pos'])
            if dist < self.PLAYER_RADIUS + self.ENEMY_RADIUS:
                return True
        return False

    def _spawn_new_asteroids(self):
        if self.np_random.random() < 0.01 and len(self.asteroids) < self.INITIAL_ASTEROIDS + 5:
            self.asteroids.append(self._create_asteroid())
    
    def _update_difficulty(self):
        self.enemy_speed_multiplier = 1.0 + (self.steps // 100) * 0.05

    def _check_termination(self):
        if self.player_lives <= 0:
            self.game_over = True
            return True
        if self.ore >= self.WIN_ORE:
            self.game_over = True
            self.game_won = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._render_stars()
        self._render_asteroids()
        self._render_enemies()
        self._render_particles()
        self._render_player()

    def _render_stars(self):
        for star in self.stars:
            pos = (star['pos'] - self.player_pos * star['parallax'])
            pos[0] %= self.WIDTH
            pos[1] %= self.HEIGHT
            pygame.draw.circle(self.screen, star['color'], (int(pos[0]), int(pos[1])), star['radius'])

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            points = [(p[0] + asteroid['pos'][0], p[1] + asteroid['pos'][1]) for p in asteroid['shape']]
            pygame.gfxdraw.aapolygon(self.screen, points, asteroid['color'])
            pygame.gfxdraw.filled_polygon(self.screen, points, asteroid['color'])

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ENEMY_RADIUS, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ENEMY_RADIUS, self.COLOR_ENEMY)
            # Glow
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ENEMY_RADIUS + 3, self.COLOR_ENEMY_GLOW)


    def _render_player(self):
        if self.player_lives <= 0: return

        pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
        
        # Glow
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS + 5, self.COLOR_PLAYER_GLOW)
        
        # Ship body
        p1 = (pos_int[0], pos_int[1] - self.PLAYER_RADIUS)
        p2 = (pos_int[0] - self.PLAYER_RADIUS * 0.8, pos_int[1] + self.PLAYER_RADIUS * 0.7)
        p3 = (pos_int[0] + self.PLAYER_RADIUS * 0.8, pos_int[1] + self.PLAYER_RADIUS * 0.7)
        points = [p1, p2, p3]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Mining laser
        if self.mining_target:
            target_pos = (int(self.mining_target['pos'][0]), int(self.mining_target['pos'][1]))
            pygame.draw.aaline(self.screen, self.COLOR_LASER, pos_int, target_pos, 1)


    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = p['color'] + (alpha,)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], p['radius'], color)

    def _render_ui(self):
        # Ore display
        ore_text = self.font_ui.render(f"ORE: {self.ore}/{self.WIN_ORE}", True, self.COLOR_TEXT)
        self.screen.blit(ore_text, (10, 10))

        # Lives display
        for i in range(self.player_lives):
            p1 = (self.WIDTH - 20 - i * 20, 10)
            p2 = (self.WIDTH - 28 - i * 20, 25)
            p3 = (self.WIDTH - 12 - i * 20, 25)
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1, p2, p3])
            
        # Game Over / Win message
        if self.game_over:
            msg = "MISSION COMPLETE" if self.game_won else "GAME OVER"
            color = self.COLOR_PLAYER if self.game_won else self.COLOR_ENEMY
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ore": self.ore,
            "lives": self.player_lives,
        }

    # --- Entity Creation Helpers ---
    
    def _create_stars(self, count):
        return [{
            'pos': np.array([self.np_random.random() * self.WIDTH, self.np_random.random() * self.HEIGHT]),
            'radius': int(self.np_random.choice([1, 1, 1, 2])),
            'color': random.choice([(100,100,100), (150,150,150), (200,200,200)]),
            'parallax': self.np_random.uniform(0.05, 0.2)
        } for _ in range(count)]

    def _create_asteroid(self):
        ore_val = self.np_random.integers(20, 101)
        if ore_val < 40:
            color = self.COLOR_ASTEROID_LOW
            radius = self.np_random.integers(12, 18)
        elif ore_val < 80:
            color = self.COLOR_ASTEROID_MID
            radius = self.np_random.integers(18, 25)
        else:
            color = self.COLOR_ASTEROID_HIGH
            radius = self.np_random.integers(25, 35)

        # Generate a random convex polygon for the shape
        num_vertices = self.np_random.integers(5, 9)
        angles = sorted([self.np_random.uniform(0, 2 * math.pi) for _ in range(num_vertices)])
        shape = []
        for angle in angles:
            r = radius * self.np_random.uniform(0.8, 1.2)
            shape.append((r * math.cos(angle), r * math.sin(angle)))

        return {
            'pos': np.array([self.np_random.random() * self.WIDTH, self.np_random.random() * self.HEIGHT]),
            'radius': radius,
            'ore': ore_val,
            'color': color,
            'shape': shape
        }

    def _create_enemy(self):
        patrol_type = self.np_random.choice(['circular', 'linear'])
        pos = np.array([self.np_random.random() * self.WIDTH, self.np_random.random() * self.HEIGHT])
        
        if patrol_type == 'circular':
            return {
                'pos': pos,
                'patrol_type': 'circular',
                'center': pos.copy(),
                'patrol_radius': self.np_random.uniform(50, 150),
                'angle': self.np_random.uniform(0, 2 * math.pi),
                'angular_speed': self.np_random.uniform(0.01, 0.03) * self.np_random.choice([-1, 1])
            }
        else: # linear
            direction = np.array([self.np_random.random() - 0.5, self.np_random.random() - 0.5])
            if np.linalg.norm(direction) > 0:
                direction /= np.linalg.norm(direction)
            return {
                'pos': pos,
                'patrol_type': 'linear',
                'start': pos.copy(),
                'patrol_radius': self.np_random.uniform(80, 200),
                'direction': direction
            }

    def _create_explosion(self, position):
        for _ in range(40):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': position.copy(),
                'vel': vel,
                'life': self.np_random.integers(20, 40),
                'max_life': 40,
                'radius': self.np_random.integers(1, 4),
                'color': self.np_random.choice([self.COLOR_ENEMY, (255, 150, 0), (255, 255, 255)])
            })

    def _create_ore_particle(self, asteroid_pos):
        # Particle moves from asteroid towards player
        direction = self.player_pos - asteroid_pos
        dist = np.linalg.norm(direction)
        if dist > 0:
            direction /= dist
        
        start_pos = asteroid_pos + direction * self.np_random.uniform(10, 20)
        
        self.particles.append({
            'pos': start_pos,
            'vel': direction * 3,
            'life': int(dist / 3),
            'max_life': int(dist / 3) + 1,
            'radius': 2,
            'color': self.COLOR_ORE_PARTICLE
        })

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


if __name__ == '__main__':
    # This block allows you to play the game directly
    # To do so, you must comment out the line: os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # and install pygame.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Space Miner")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
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
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # In a real game, you might wait for a key press to reset
            # For this example, we'll just let it show the final screen
            # running = False # uncomment to exit after one episode
            
        clock.tick(30) # Corresponds to 30 FPS
        
    env.close()