
# Generated: 2025-08-27T21:42:51.423554
# Source Brief: brief_02884.md
# Brief Index: 2884

        
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
        "Controls: ←→ to move. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist side-view shooter where the player must destroy all invading aliens while dodging their projectiles."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PLAYER_SPEED = 8
        self.PLAYER_FIRE_COOLDOWN_MAX = 6  # Frames
        self.PLAYER_PROJECTILE_SPEED = 12
        self.ALIEN_PROJECTILE_SPEED = 5
        self.ALIEN_MOVE_INTERVAL = 30  # Frames between moves
        self.NUM_ALIENS_X = 10
        self.NUM_ALIENS_Y = 3
        self.MAX_STEPS = 1500 # Extended from 1000 to allow for completion

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_STAR = (50, 50, 70)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_OUTLINE = (200, 255, 220)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_PLAYER_PROJECTILE = (255, 255, 255)
        self.COLOR_ALIEN_PROJECTILE = (200, 100, 255)
        self.COLOR_UI = (220, 220, 220)
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_lives = None
        self.player_fire_cooldown = None
        self.player_projectiles = None
        self.aliens = None
        self.alien_projectiles = None
        self.alien_move_timer = None
        self.alien_direction = None
        self.alien_fire_prob = None
        self.particles = None
        self.stars = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize state variables
        self.reset()
        
        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player state
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 40]
        self.player_lives = 3
        self.player_fire_cooldown = 0
        
        # Projectiles
        self.player_projectiles = []
        self.alien_projectiles = []
        
        # Aliens
        self.aliens = self._create_aliens()
        self.alien_move_timer = 0
        self.alien_direction = 1 # 1 for right, -1 for left
        self.alien_fire_prob = 0.005 # Adjusted from brief for better gameplay
        
        # Effects
        self.particles = []
        self.stars = [(self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)) for _ in range(100)]
        
        return self._get_observation(), self._get_info()

    def _create_aliens(self):
        aliens = []
        alien_size = 20
        start_x = (self.WIDTH - (self.NUM_ALIENS_X * (alien_size + 30)) + 30) / 2
        for y in range(self.NUM_ALIENS_Y):
            for x in range(self.NUM_ALIENS_X):
                pos = [start_x + x * 50, 50 + y * 40]
                aliens.append({'pos': pos, 'alive': True, 'size': alien_size})
        return aliens
    
    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage speed
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Input ---
        movement = action[0]
        space_pressed = action[1] == 1
        
        if movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.WIDTH - 20)
        
        if space_pressed and self.player_fire_cooldown == 0:
            # Pew!
            self.player_projectiles.append(list(self.player_pos))
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN_MAX

        # --- Update Game Logic ---
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1
            
        self._update_projectiles()
        self._update_aliens()
        
        # --- Handle Collisions ---
        collision_reward = self._handle_collisions()
        reward += collision_reward
        
        # --- Update Effects ---
        self._update_particles()

        # --- Update Difficulty & Steps ---
        self.steps += 1
        if self.steps > 0 and self.steps % 100 == 0:
            self.alien_fire_prob = min(0.05, self.alien_fire_prob + 0.005)

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if all(not alien['alive'] for alien in self.aliens):
                reward += 100  # Win bonus
            else:
                reward -= 100  # Loss penalty
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_projectiles(self):
        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj[1] -= self.PLAYER_PROJECTILE_SPEED
            if proj[1] < 0:
                self.player_projectiles.remove(proj)
        
        # Alien projectiles
        for proj in self.alien_projectiles[:]:
            proj[1] += self.ALIEN_PROJECTILE_SPEED
            if proj[1] > self.HEIGHT:
                self.alien_projectiles.remove(proj)

    def _update_aliens(self):
        self.alien_move_timer += 1
        if self.alien_move_timer >= self.ALIEN_MOVE_INTERVAL:
            self.alien_move_timer = 0
            
            move_down = False
            living_aliens = [a for a in self.aliens if a['alive']]
            if not living_aliens: return

            for alien in living_aliens:
                if (alien['pos'][0] >= self.WIDTH - alien['size'] and self.alien_direction == 1) or \
                   (alien['pos'][0] <= alien['size'] and self.alien_direction == -1):
                    move_down = True
                    break
            
            if move_down:
                self.alien_direction *= -1
                for alien in self.aliens:
                    alien['pos'][1] += 15
            else:
                for alien in self.aliens:
                    alien['pos'][0] += self.alien_direction * 5

        # Alien firing
        for alien in self.aliens:
            if alien['alive'] and self.np_random.random() < self.alien_fire_prob:
                # Zap!
                self.alien_projectiles.append(list(alien['pos']))

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0] - 12, self.player_pos[1] - 8, 24, 20)

        # Alien projectiles vs Player
        for proj in self.alien_projectiles[:]:
            proj_rect = pygame.Rect(proj[0] - 3, proj[1] - 3, 6, 6)
            if player_rect.colliderect(proj_rect):
                self.alien_projectiles.remove(proj)
                self.player_lives -= 1
                reward -= 5
                # Player Hit!
                self._create_particle_effect(self.player_pos, self.COLOR_PLAYER, 20)
                break

        # Player projectiles vs Aliens
        for p_proj in self.player_projectiles[:]:
            proj_rect = pygame.Rect(p_proj[0] - 1, p_proj[1] - 10, 2, 20)
            for alien in self.aliens:
                if alien['alive']:
                    alien_rect = pygame.Rect(alien['pos'][0] - alien['size']/2, alien['pos'][1] - alien['size']/2, alien['size'], alien['size'])
                    if proj_rect.colliderect(alien_rect):
                        alien['alive'] = False
                        if p_proj in self.player_projectiles:
                            self.player_projectiles.remove(p_proj)
                        reward += 10
                        self.score += 100
                        # Alien Explodes!
                        self._create_particle_effect(alien['pos'], self.COLOR_ALIEN, 30)
                        break
        return reward

    def _create_particle_effect(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.player_lives <= 0:
            return True
        if all(not alien['alive'] for alien in self.aliens):
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for star_pos in self.stars:
            self.screen.set_at(star_pos, self.COLOR_STAR)

        # Aliens
        for alien in self.aliens:
            if alien['alive']:
                pos_int = (int(alien['pos'][0]), int(alien['pos'][1]))
                size = alien['size']
                rect = pygame.Rect(pos_int[0] - size//2, pos_int[1] - size//2, size, size)
                pygame.draw.rect(self.screen, self.COLOR_ALIEN, rect)

        # Player
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        player_points = [(px, py - 12), (px - 12, py + 8), (px + 12, py + 8)]
        pygame.gfxdraw.aapolygon(self.screen, player_points, self.COLOR_PLAYER_OUTLINE)
        pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER)

        # Projectiles
        for proj in self.player_projectiles:
            start = (int(proj[0]), int(proj[1]))
            end = (int(proj[0]), int(proj[1] - 10))
            pygame.draw.line(self.screen, self.COLOR_PLAYER_PROJECTILE, start, end, 2)
        
        for proj in self.alien_projectiles:
            pos_int = (int(proj[0]), int(proj[1]))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 4, self.COLOR_ALIEN_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 4, self.COLOR_ALIEN_PROJECTILE)
            
        # Particles
        for p in self.particles:
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = int(255 * (p['life'] / 30.0))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (pos_int[0] - 2, pos_int[1] - 2))

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Lives
        for i in range(self.player_lives):
            self._draw_heart(30 + i * 40, 25)

    def _draw_heart(self, x, y):
        points = [
            (x, y - 5), (x + 10, y - 15), (x + 15, y - 5), (x, y + 10),
            (x - 15, y - 5), (x - 10, y - 15)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "aliens_left": sum(1 for a in self.aliens if a['alive'])
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Minimalist Shooter")
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Blit the observation from the environment to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            env.reset()

        clock.tick(30) # Match the intended FPS
        
    pygame.quit()