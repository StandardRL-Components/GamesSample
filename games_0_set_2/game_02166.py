
# Generated: 2025-08-27T19:27:37.775616
# Source Brief: brief_02166.md
# Brief Index: 2166

        
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
        "Controls: Arrow keys to move. Collect gems and avoid the enemies."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect glittering gems while dodging cunning enemies in a fast-paced, top-down arcade environment."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.WIN_GEM_COUNT = 50
        self.STARTING_LIVES = 3
        self.NUM_GEMS = 5

        # Colors
        self.COLOR_BG = (16, 16, 24) # Dark blue-gray
        self.COLOR_PLAYER = (0, 160, 255)
        self.COLOR_PLAYER_GLOW = (0, 160, 255, 50)
        self.ENEMY_COLORS = {
            "slow": (255, 50, 50),
            "medium": (255, 150, 50),
            "fast": (255, 255, 50)
        }
        self.COLOR_TEXT = (240, 240, 240)
        
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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_main = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Game entity parameters
        self.PLAYER_SIZE = 12
        self.PLAYER_SPEED = 15
        self.GEM_SIZE = 8
        self.ENEMY_SIZE = 14

        # Initialize state variables
        self.player_pos = None
        self.gems = None
        self.enemies = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.lives = None
        self.gem_count = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.lives = self.STARTING_LIVES
        self.gem_count = 0
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        
        self.enemies = []
        self._spawn_enemies()
        
        self.gems = []
        for _ in range(self.NUM_GEMS):
            self._spawn_gem()

        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Store state for reward calculation
        old_player_pos = self.player_pos.copy()
        closest_gem_dist_before = self._get_closest_entity_dist(self.gems)
        closest_enemy_dist_before = self._get_closest_entity_dist(self.enemies)

        # 1. Update Player
        self._move_player(movement)
        
        # 2. Update Enemies
        self._move_enemies()
        
        # 3. Handle Interactions and calculate event rewards
        event_reward = 0
        
        # Player-Gem collision
        collected_gem = None
        for gem in self.gems:
            if self.player_pos.distance_to(gem['pos']) < self.PLAYER_SIZE / 2 + gem['size']:
                collected_gem = gem
                break
        
        if collected_gem:
            # SFX: gem_collect.wav
            self.gems.remove(collected_gem)
            self._spawn_gem()
            self.gem_count += 1
            event_reward += 10
            self.score += 10
            self._create_particles(collected_gem['pos'], collected_gem['color'], 20)
        
        # Player-Enemy collision
        for enemy in self.enemies:
            if self.player_pos.distance_to(enemy['pos']) < self.PLAYER_SIZE / 2 + self.ENEMY_SIZE / 2:
                # SFX: player_hit.wav
                self.lives -= 1
                event_reward -= 25 # Immediate penalty for getting hit
                self._create_particles(self.player_pos, (255, 0, 0), 30)
                self.player_pos.update(self.WIDTH / 2, self.HEIGHT / 2) # Reset position
                break
        
        # 4. Calculate continuous rewards
        closest_gem_dist_after = self._get_closest_entity_dist(self.gems)
        closest_enemy_dist_after = self._get_closest_entity_dist(self.enemies)
        
        continuous_reward = 0
        if closest_gem_dist_after < closest_gem_dist_before:
            continuous_reward += 1.0  # Moved closer to a gem
        elif closest_gem_dist_after > closest_gem_dist_before:
            continuous_reward -= 2.0 # Moved away from a gem (less severe than brief)

        if closest_enemy_dist_after < closest_enemy_dist_before:
            continuous_reward -= 0.1 # Moved closer to an enemy

        # 5. Update animations and step counter
        self._update_particles()
        self._update_gems_animation()
        self.steps += 1
        
        # 6. Check for termination
        terminated = self._check_termination()
        terminal_reward = 0
        if terminated:
            if self.win:
                terminal_reward = 100
            else: # Loss or timeout
                terminal_reward = -100
        
        reward = event_reward + continuous_reward + terminal_reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "gem_count": self.gem_count,
        }

    def _render_game(self):
        # Render gems
        for gem in self.gems:
            pos = (int(gem['pos'].x), int(gem['pos'].y))
            size = int(gem['size'])
            color = gem['color']
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)

        # Render enemies
        for enemy in self.enemies:
            pos = enemy['pos']
            color = self.ENEMY_COLORS[enemy['type']]
            points = [
                (pos.x, pos.y - self.ENEMY_SIZE / 2),
                (pos.x - self.ENEMY_SIZE / 2, pos.y + self.ENEMY_SIZE / 2),
                (pos.x + self.ENEMY_SIZE / 2, pos.y + self.ENEMY_SIZE / 2)
            ]
            int_points = [(int(p[0]), int(p[1])) for p in points]
            pygame.gfxdraw.aapolygon(self.screen, int_points, color)
            pygame.gfxdraw.filled_polygon(self.screen, int_points, color)

        # Render player
        player_rect = pygame.Rect(
            self.player_pos.x - self.PLAYER_SIZE / 2,
            self.player_pos.y - self.PLAYER_SIZE / 2,
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )
        # Glow effect
        glow_radius = int(self.PLAYER_SIZE * 1.5)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (int(self.player_pos.x - glow_radius), int(self.player_pos.y - glow_radius)))
        # Player square
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)
        
        # Render particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), int(p['radius']))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Gem Count
        gem_text = self.font_ui.render(f"GEMS: {self.gem_count}/{self.WIN_GEM_COUNT}", True, self.COLOR_TEXT)
        self.screen.blit(gem_text, (self.WIDTH - gem_text.get_width() - 10, 10))
        
        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH / 2 - lives_text.get_width() / 2, self.HEIGHT - 30))

        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_main.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _spawn_gem(self):
        while True:
            pos = pygame.Vector2(
                random.uniform(self.GEM_SIZE, self.WIDTH - self.GEM_SIZE),
                random.uniform(self.GEM_SIZE, self.HEIGHT - self.GEM_SIZE)
            )
            # Ensure it doesn't spawn too close to the player
            if pos.distance_to(self.player_pos) > 50:
                self.gems.append({
                    'pos': pos,
                    'size': self.GEM_SIZE,
                    'size_dir': 1,
                    'color': random.choice([(255,0,255), (0,255,255), (255,255,0), (128,0,255)])
                })
                break
    
    def _spawn_enemies(self):
        # Slow, horizontal patrol
        self.enemies.append({
            'pos': pygame.Vector2(50, 50),
            'type': 'slow', 'speed': 1.5, 'pattern': 'horizontal',
            'state': {'dir': 1, 'bounds': (50, self.WIDTH - 50)}
        })
        # Medium, circular patrol
        self.enemies.append({
            'pos': pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2),
            'type': 'medium', 'speed': 0.04, 'pattern': 'circular',
            'state': {'center': pygame.Vector2(self.WIDTH - 100, self.HEIGHT - 100), 'radius': 70, 'angle': 0}
        })
        # Fast, random walk
        self.enemies.append({
            'pos': pygame.Vector2(self.WIDTH - 50, 50),
            'type': 'fast', 'speed': 3.0, 'pattern': 'random_walk',
            'state': {'vel': pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize(), 'timer': 0}
        })

    def _move_player(self, movement):
        if movement == 1: # Up
            self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_pos.y += self.PLAYER_SPEED
        elif movement == 3: # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos.x += self.PLAYER_SPEED
        
        # Clamp player position
        self.player_pos.x = max(self.PLAYER_SIZE / 2, min(self.player_pos.x, self.WIDTH - self.PLAYER_SIZE / 2))
        self.player_pos.y = max(self.PLAYER_SIZE / 2, min(self.player_pos.y, self.HEIGHT - self.PLAYER_SIZE / 2))

    def _move_enemies(self):
        for enemy in self.enemies:
            if enemy['pattern'] == 'horizontal':
                enemy['pos'].x += enemy['speed'] * enemy['state']['dir']
                if enemy['pos'].x <= enemy['state']['bounds'][0] or enemy['pos'].x >= enemy['state']['bounds'][1]:
                    enemy['state']['dir'] *= -1
            elif enemy['pattern'] == 'circular':
                enemy['state']['angle'] += enemy['speed']
                enemy['pos'].x = enemy['state']['center'].x + enemy['state']['radius'] * math.cos(enemy['state']['angle'])
                enemy['pos'].y = enemy['state']['center'].y + enemy['state']['radius'] * math.sin(enemy['state']['angle'])
            elif enemy['pattern'] == 'random_walk':
                enemy['pos'] += enemy['state']['vel'] * enemy['speed']
                enemy['state']['timer'] += 1
                if enemy['state']['timer'] > 60: # Change direction every 2 seconds (at 30fps)
                    enemy['state']['vel'] = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()
                    enemy['state']['timer'] = 0
                # Bounce off walls
                if not (0 < enemy['pos'].x < self.WIDTH): enemy['state']['vel'].x *= -1
                if not (0 < enemy['pos'].y < self.HEIGHT): enemy['state']['vel'].y *= -1
                enemy['pos'].x = max(0, min(self.WIDTH, enemy['pos'].x))
                enemy['pos'].y = max(0, min(self.HEIGHT, enemy['pos'].y))

    def _update_gems_animation(self):
        for gem in self.gems:
            gem['size'] += gem['size_dir'] * 0.1
            if gem['size'] > self.GEM_SIZE or gem['size'] < self.GEM_SIZE * 0.7:
                gem['size_dir'] *= -1

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(random.uniform(-2, 2), random.uniform(-2, 2)),
                'radius': random.uniform(2, 5),
                'lifespan': random.randint(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['radius'] -= 0.1
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['radius'] > 0]

    def _get_closest_entity_dist(self, entities):
        if not entities:
            return float('inf')
        return min(self.player_pos.distance_to(e['pos']) for e in entities)

    def _check_termination(self):
        if self.gem_count >= self.WIN_GEM_COUNT:
            self.game_over = True
            self.win = True
        elif self.lives <= 0:
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up a window to display the game
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4

        # In a real-time game, you'd handle held keys for space/shift
        # action[1] = 1 if keys[pygame.K_SPACE] else 0
        # action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        if done:
            # If game is over, wait for a key press to reset
            if any(keys):
                obs, info = env.reset()
                done = False
        else:
            # Only step if a movement key is pressed
            if action[0] != 0:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Done: {done}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Since auto_advance is False, we control the "frame rate" here
        # This loop will run as fast as possible, waiting for key presses
        pygame.time.wait(16) # ~60 FPS cap to not burn CPU

    env.close()