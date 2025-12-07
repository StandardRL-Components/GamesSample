
# Generated: 2025-08-27T16:05:31.368576
# Source Brief: brief_01112.md
# Brief Index: 1112

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class Particle:
    """A simple class for explosion particles."""
    def __init__(self, x, y, rng):
        self.x = x
        self.y = y
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = rng.integers(15, 30)
        self.color = random.choice([(255, 255, 0), (255, 165, 0), (255, 69, 0)])
        self.size = rng.integers(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.size = max(0, self.size - 0.1)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move. Hold space to fire."
    )

    game_description = (
        "A top-down shooter where you must destroy waves of descending aliens "
        "while dodging their projectiles to achieve the highest score."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        self.game_over_font = pygame.font.SysFont("monospace", 50, bold=True)

        # Colors
        self.COLOR_BG = (10, 0, 20)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 200)
        self.COLOR_ALIEN = (180, 50, 220)
        self.COLOR_PLAYER_PROJ = (100, 255, 100)
        self.COLOR_ALIEN_PROJ = (255, 50, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.STAR_COLORS = [(100, 100, 100), (150, 150, 150), (200, 200, 200)]

        # Game parameters
        self.MAX_STEPS = 2500
        self.PLAYER_SPEED = 7
        self.PLAYER_FIRE_COOLDOWN_MAX = 6 # 5 shots/sec
        self.PLAYER_PROJECTILE_SPEED = 10
        self.ALIEN_PROJECTILE_SPEED = 4
        self.ALIEN_FIRE_RATE = 1.0 # shots per second
        
        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_lives = 0
        self.player_pos = [0, 0]
        self.player_fire_cooldown = 0
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        self.stars = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_lives = 3
        self.player_pos = [self.WIDTH / 2, self.HEIGHT - 40]
        self.player_fire_cooldown = 0
        
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        self.stars = []
        
        self._spawn_stars()
        self._spawn_aliens()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = -0.01

        if not self.game_over:
            self._handle_input(action)
            self._update_player()
            self._update_aliens()
            self._update_projectiles()
            
            collision_reward = self._handle_collisions()
            reward += collision_reward

        self._update_particles()
        self._update_stars()
        
        self.steps += 1
        
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.WIDTH - 20)

        if space_held and self.player_fire_cooldown <= 0:
            # sfx: player_shoot.wav
            self.player_projectiles.append(list(self.player_pos))
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN_MAX

    def _update_player(self):
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1

    def _update_aliens(self):
        alien_descent_speed = 0.2
        for alien in self.aliens[:]:
            alien['pos'][1] += alien_descent_speed
            alien['pos'][0] = alien['base_x'] + math.sin(self.steps * 0.05 + alien['offset']) * 40
            
            if alien['pos'][1] > self.HEIGHT - 20:
                self.aliens.remove(alien)
                if self.player_lives > 0:
                    self.player_lives -=1
                    # sfx: player_hit.wav
                    self._create_explosion(self.player_pos[0], self.player_pos[1])
                
        # Alien Firing
        if self.aliens and self.np_random.random() < self.ALIEN_FIRE_RATE / 30:
            shooter = self.np_random.choice(self.aliens)
            # sfx: alien_shoot.wav
            self.alien_projectiles.append(list(shooter['pos']))

    def _update_projectiles(self):
        self.player_projectiles = [p for p in self.player_projectiles if p[1] > 0]
        for p in self.player_projectiles:
            p[1] -= self.PLAYER_PROJECTILE_SPEED

        self.alien_projectiles = [p for p in self.alien_projectiles if p[1] < self.HEIGHT]
        for p in self.alien_projectiles:
            p[1] += self.ALIEN_PROJECTILE_SPEED
            
    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Aliens
        for p in self.player_projectiles[:]:
            for a in self.aliens[:]:
                if math.hypot(p[0] - a['pos'][0], p[1] - a['pos'][1]) < 15:
                    # sfx: alien_explosion.wav
                    self._create_explosion(a['pos'][0], a['pos'][1])
                    if p in self.player_projectiles: self.player_projectiles.remove(p)
                    if a in self.aliens: self.aliens.remove(a)
                    self.score += 1
                    reward += 1
                    break
        
        # Alien projectiles vs Player
        if self.player_lives > 0:
            player_hitbox = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - 10, 20, 20)
            for p in self.alien_projectiles[:]:
                if player_hitbox.collidepoint(p[0], p[1]):
                    # sfx: player_hit.wav
                    self._create_explosion(self.player_pos[0], self.player_pos[1])
                    if p in self.alien_projectiles: self.alien_projectiles.remove(p)
                    self.player_lives -= 1
                    reward -= 1
                    break
        return reward

    def _check_termination(self):
        if self.player_lives <= 0:
            return True, -100
        if not self.aliens:
            return True, 100
        if self.steps >= self.MAX_STEPS:
            return True, 0
        return False, 0

    def _spawn_stars(self):
        for _ in range(100):
            self.stars.append({
                'pos': [self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)],
                'speed': self.np_random.uniform(0.2, 1.0),
                'size': self.np_random.integers(1, 3)
            })

    def _spawn_aliens(self):
        rows, cols = 5, 10
        for i in range(rows):
            for j in range(cols):
                self.aliens.append({
                    'pos': [j * 50 + 80, i * 40 + 50],
                    'base_x': j * 50 + 80,
                    'offset': i * 0.5 + j * 0.2
                })

    def _create_explosion(self, x, y):
        for _ in range(20):
            self.particles.append(Particle(x, y, self.np_random))

    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.lifespan <= 0:
                self.particles.remove(p)
                
    def _update_stars(self):
        for star in self.stars:
            star['pos'][1] += star['speed']
            if star['pos'][1] > self.HEIGHT:
                star['pos'][0] = self.np_random.integers(0, self.WIDTH)
                star['pos'][1] = 0

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
            "lives": self.player_lives,
        }

    def _render_game(self):
        self._render_stars()
        self._render_projectiles()
        self._render_aliens()
        if self.player_lives > 0:
            self._render_player()
        self._render_particles()

    def _render_stars(self):
        for star in self.stars:
            color_index = min(len(self.STAR_COLORS) - 1, int(star['speed']))
            pygame.draw.circle(self.screen, self.STAR_COLORS[color_index], (int(star['pos'][0]), int(star['pos'][1])), star['size'])

    def _render_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        points = [(x, y - 15), (x - 12, y + 10), (x + 12, y + 10)]
        pygame.gfxdraw.filled_circle(self.screen, x, y, 20, (*self.COLOR_PLAYER_GLOW, 100))
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        
    def _render_aliens(self):
        for alien in self.aliens:
            x, y = int(alien['pos'][0]), int(alien['pos'][1])
            pygame.gfxdraw.aacircle(self.screen, x, y, 10, self.COLOR_ALIEN)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 10, self.COLOR_ALIEN)
            pygame.gfxdraw.aacircle(self.screen, x, y-2, 5, (255,255,255))
            pygame.gfxdraw.filled_circle(self.screen, x, y-2, 5, (255,255,255))

    def _render_projectiles(self):
        for p in self.player_projectiles:
            start = (int(p[0]), int(p[1]))
            end = (int(p[0]), int(p[1]) - 10)
            pygame.draw.line(self.screen, self.COLOR_PLAYER_PROJ, start, end, 3)
        for p in self.alien_projectiles:
            pygame.gfxdraw.aacircle(self.screen, int(p[0]), int(p[1]), 4, self.COLOR_ALIEN_PROJ)
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), 4, self.COLOR_ALIEN_PROJ)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p.lifespan / 30))
            color = (*p.color, alpha)
            temp_surf = pygame.Surface((int(p.size*2), int(p.size*2)), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (int(p.size), int(p.size)), int(p.size))
            self.screen.blit(temp_surf, (p.x - p.size, p.y - p.size))

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        for i in range(self.player_lives):
            x, y = self.WIDTH - 30 - (i * 30), 25
            points = [(x, y - 10), (x - 8, y + 6), (x + 8, y + 6)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        
        if self.game_over:
            msg = "YOU WIN!" if not self.aliens and self.player_lives > 0 else "GAME OVER"
            color = (100, 255, 100) if msg == "YOU WIN!" else (255, 50, 50)
            end_text = self.game_over_font.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

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

if __name__ == '__main__':
    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy' # Run headless
    
    env = GameEnv()
    obs, info = env.reset(seed=42)
    print("Initial Observation Shape:", obs.shape)
    print("Initial Info:", info)

    terminated = False
    total_reward = 0
    for _ in range(300):
        if terminated:
            break
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print(f"Finished after 300 steps.")
    print(f"Final Info: {info}")
    print(f"Total Reward: {total_reward}")