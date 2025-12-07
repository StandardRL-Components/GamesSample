
# Generated: 2025-08-28T02:56:12.413186
# Source Brief: brief_01861.md
# Brief Index: 1861

        
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
        "Controls: Use arrow keys (↑↓←→) to move your character. Avoid the red monsters and collect the gold coins."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Dodge procedurally generated monsters and collect coins in a top-down arcade game to amass a fortune before getting caught."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 0, 30)
    COLOR_COIN = (255, 215, 0)
    COLOR_COIN_GLOW = (255, 215, 0, 40)
    COLOR_MONSTER = (255, 50, 50)
    COLOR_MONSTER_GLOW = (255, 50, 50, 40)
    COLOR_PARTICLE = (220, 220, 220)
    COLOR_TEXT = (255, 255, 255)
    COLOR_HEART = (255, 80, 80)

    PLAYER_SIZE = 16
    PLAYER_SPEED = 5
    COIN_RADIUS = 6
    MAX_LIVES = 3
    WIN_SCORE = 50
    MAX_STEPS = 1000
    INITIAL_MONSTERS = 5
    INITIAL_COINS = 10
    
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
        self.font = pygame.font.Font(None, 36)
        
        # Game state variables are initialized in reset()
        self.player_pos = None
        self.player_lives = 0
        self.monsters = []
        self.coins = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_monster_speed = 0
        
        # Initialize state variables
        self.reset()

        # This check is not part of the brief but is good practice.
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_lives = self.MAX_LIVES
        
        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        
        self.base_monster_speed = 0.5
        self.monsters = []
        for _ in range(self.INITIAL_MONSTERS):
            self._spawn_monster()
            
        self.coins = []
        for _ in range(self.INITIAL_COINS):
            self._spawn_coin()
            
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Reward Initialization ---
        reward = 0.1  # Survival reward

        # --- Handle Actions ---
        self._handle_player_movement(action)

        # --- Update Game Logic ---
        self._update_monsters()
        self._update_particles()
        
        # --- Handle Collisions & Events ---
        reward += self._handle_collisions()
        
        self.steps += 1
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100  # Win bonus
            elif self.player_lives <= 0:
                reward -= 100  # Loss penalty
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_player_movement(self, action):
        movement = action[0]
        if movement == 1:  # Up
            self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos.y += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos.x += self.PLAYER_SPEED

        # Clamp player position to screen boundaries
        self.player_pos.x = max(self.PLAYER_SIZE / 2, min(self.player_pos.x, self.WIDTH - self.PLAYER_SIZE / 2))
        self.player_pos.y = max(self.PLAYER_SIZE / 2, min(self.player_pos.y, self.HEIGHT - self.PLAYER_SIZE / 2))

    def _update_monsters(self):
        current_monster_speed = self.base_monster_speed + (self.score // 10) * 0.05
        for monster in self.monsters:
            monster['time'] += 1
            t = monster['time'] * 0.02 * current_monster_speed
            
            if monster['pattern'] == 'circle':
                monster['pos'].x = monster['center'].x + monster['radius'] * math.cos(t)
                monster['pos'].y = monster['center'].y + monster['radius'] * math.sin(t)
            elif monster['pattern'] == 'figure_eight':
                monster['pos'].x = monster['center'].x + monster['radius'] * math.sin(t)
                monster['pos'].y = monster['center'].y + monster['radius'] * math.sin(2 * t)
            elif monster['pattern'] == 'line':
                monster['pos'].x = monster['center'].x + monster['radius'] * math.sin(t)

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(
            self.player_pos.x - self.PLAYER_SIZE / 2,
            self.player_pos.y - self.PLAYER_SIZE / 2,
            self.PLAYER_SIZE,
            self.PLAYER_SIZE
        )

        # Player-Coin Collisions
        for coin_pos in self.coins[:]:
            if coin_pos.distance_to(self.player_pos) < self.COIN_RADIUS + self.PLAYER_SIZE / 2:
                self.coins.remove(coin_pos)
                self._spawn_coin()
                self.score += 1
                reward += 1.0
                # Sound: coin_collect.wav

                # Check for risk bonus
                for monster in self.monsters:
                    if coin_pos.distance_to(monster['pos']) < 15 + monster['size']:
                        reward += 2.0
                        break # Only one bonus per coin
        
        # Player-Monster Collisions
        for monster in self.monsters:
            if player_rect.clipline(
                monster['pos'].x - monster['size'], monster['pos'].y,
                monster['pos'].x + monster['size'], monster['pos'].y
            ) or player_rect.clipline(
                monster['pos'].x, monster['pos'].y - monster['size'],
                monster['pos'].x, monster['pos'].y + monster['size']
            ) or player_rect.collidepoint(monster['pos']):
                closest_point_on_rect = pygame.math.Vector2(
                    max(player_rect.left, min(monster['pos'].x, player_rect.right)),
                    max(player_rect.top, min(monster['pos'].y, player_rect.bottom))
                )
                if closest_point_on_rect.distance_to(monster['pos']) < monster['size']:
                    self.player_lives -= 1
                    reward -= 5.0
                    self._create_hit_particles(self.player_pos)
                    # Sound: player_hit.wav
                    self.monsters.remove(monster)
                    self._spawn_monster()
                    if self.player_lives <= 0:
                        self.game_over = True
                        # Sound: game_over.wav
                    break

        return reward

    def _check_termination(self):
        return (
            self.player_lives <= 0 or 
            self.score >= self.WIN_SCORE or
            self.steps >= self.MAX_STEPS
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Coins
        for coin_pos in self.coins:
            pygame.gfxdraw.filled_circle(self.screen, int(coin_pos.x), int(coin_pos.y), self.COIN_RADIUS + 3, self.COLOR_COIN_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, int(coin_pos.x), int(coin_pos.y), self.COIN_RADIUS, self.COLOR_COIN)
            
        # Render Monsters
        for monster in self.monsters:
            pos = (int(monster['pos'].x), int(monster['pos'].y))
            size = int(monster['size'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size + 4, self.COLOR_MONSTER_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_MONSTER)
            
        # Render Player
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = (int(self.player_pos.x), int(self.player_pos.y))
        glow_rect = player_rect.inflate(8, 8)
        
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_PLAYER_GLOW, s.get_rect(), border_radius=3)
        self.screen.blit(s, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

        # Render Particles
        for p in self.particles:
            pygame.draw.rect(self.screen, p['color'], (p['pos'].x, p['pos'].y, 2, 2))

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives (Hearts)
        for i in range(self.player_lives):
            self._draw_heart(self.WIDTH - 30 - (i * 35), 25, 30)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
        }
        
    def _spawn_coin(self):
        pos = pygame.math.Vector2(
            self.np_random.integers(self.COIN_RADIUS, self.WIDTH - self.COIN_RADIUS),
            self.np_random.integers(self.COIN_RADIUS, self.HEIGHT - self.COIN_RADIUS)
        )
        self.coins.append(pos)
        
    def _spawn_monster(self):
        pattern = self.np_random.choice(['circle', 'figure_eight', 'line'])
        center = pygame.math.Vector2(
            self.np_random.integers(100, self.WIDTH - 100),
            self.np_random.integers(100, self.HEIGHT - 100)
        )
        monster = {
            'pos': pygame.math.Vector2(center),
            'center': center,
            'size': self.np_random.integers(8, 15),
            'radius': self.np_random.integers(50, 120),
            'pattern': pattern,
            'time': self.np_random.integers(0, 1000)
        }
        self.monsters.append(monster)

    def _create_hit_particles(self, position):
        for _ in range(30):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.math.Vector2(position),
                'vel': vel,
                'lifespan': self.np_random.integers(15, 30)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
                
    def _draw_heart(self, x, y, size):
        # Simple heart shape drawing function
        s2 = size / 2
        s3 = size / 3
        points = [
            (x, y - s3),
            (x + s2, y - s2),
            (x + s2, y),
            (x, y + s2),
            (x - s2, y),
            (x - s2, y - s2),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_HEART)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_HEART)
        pygame.gfxdraw.filled_circle(self.screen, int(x - s3), int(y - s3), int(s3), self.COLOR_HEART)
        pygame.gfxdraw.filled_circle(self.screen, int(x + s3), int(y - s3), int(s3), self.COLOR_HEART)

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

# Example of how to run the environment
if __name__ == '__main__':
    # Set up Pygame for human interaction
    pygame.display.set_caption("Arcade Dodger")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Use a separate clock for the display loop to avoid interfering with the env's clock
    display_clock = pygame.time.Clock()

    while running:
        # --- Human Input ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Render to Display ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            print("Press 'R' to restart.")

    env.close()