
# Generated: 2025-08-27T23:18:24.093654
# Source Brief: brief_03424.md
# Brief Index: 3424

        
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
        "Controls: Arrow keys to move. Hold Space to fire. Destroy all aliens to win."
    )

    game_description = (
        "A retro top-down shooter. Destroy the alien invaders while dodging their fire. Your lives are limited!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 10000
        self.FPS = 30

        # Game parameters
        self.INITIAL_LIVES = 3
        self.NUM_ALIENS = 50
        self.PLAYER_SPEED = 7
        self.PLAYER_FIRE_COOLDOWN = 8  # frames
        self.PROJECTILE_SPEED = 12
        self.ALIEN_PROJECTILE_SPEED = 5
        self.ALIEN_INITIAL_FIRE_RATE = 0.2 # shots per second
        self.ALIEN_FIRE_RATE_INCREASE = 0.002
        self.ALIEN_FIRE_RATE_CAP = 1.0

        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_PLAYER_PROJ = (200, 255, 255)
        self.COLOR_ALIEN_PROJ = (255, 255, 100)
        self.COLOR_EXPLOSION = (255, 180, 0)
        self.COLOR_TEXT = (255, 255, 255)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.Font(None, 28)
        self.end_font = pygame.font.Font(None, 72)

        # Initialize state variables
        self.player = {}
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.aliens_destroyed = 0
        self.alien_fire_rate = 0.0
        self.alien_fire_timer = 0
        self.alien_direction = 1
        self.alien_move_down_timer = 0
        
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.aliens_destroyed = 0

        # Reset entities
        self._spawn_player()
        self._spawn_aliens()
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        
        # Reset alien behavior
        self.alien_fire_rate = self.ALIEN_INITIAL_FIRE_RATE
        self.alien_fire_timer = self._get_alien_fire_cooldown()
        self.alien_direction = 1
        self.alien_move_down_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)
        
        reward = -0.01  # Small penalty for each step to encourage efficiency

        if not self.game_over:
            # Unpack action
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

            # Update game logic
            reward += self._handle_input(movement, space_held)
            self._update_aliens()
            self._update_projectiles()
            self._update_particles()
            reward += self._handle_collisions()

        # Check for termination conditions
        terminated = self._check_termination()
        if terminated and not self.game_over:
             if len([a for a in self.aliens if a['alive']]) == 0:
                reward += 100 # Victory bonus
             self.game_over = True

        self.steps += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info(),
        )

    def _spawn_player(self):
        self.player = {
            'pos': np.array([self.WIDTH / 2, self.HEIGHT - 40], dtype=np.float64),
            'lives': self.INITIAL_LIVES,
            'size': 15.0,
            'cooldown': 0,
            'hit_timer': 0,
        }

    def _spawn_aliens(self):
        self.aliens = []
        rows, cols = 5, 10
        x_spacing, y_spacing = 50, 40
        start_x = (self.WIDTH - (cols - 1) * x_spacing) / 2
        start_y = 50
        for row in range(rows):
            for col in range(cols):
                self.aliens.append({
                    'pos': np.array([start_x + col * x_spacing, start_y + row * y_spacing], dtype=np.float64),
                    'alive': True,
                    'size': 12.0
                })
    
    def _find_nearest_alive_alien(self):
        alive_aliens = [a for a in self.aliens if a['alive']]
        if not alive_aliens:
            return None, float('inf')
        
        player_pos = self.player['pos']
        distances = [np.linalg.norm(player_pos - a['pos']) for a in alive_aliens]
        min_idx = np.argmin(distances)
        return alive_aliens[min_idx], distances[min_idx]

    def _handle_input(self, movement, space_held):
        reward = 0
        
        # Movement and reward for getting closer to an alien
        _, dist_before = self._find_nearest_alive_alien()
        
        if movement == 1: self.player['pos'][1] -= self.PLAYER_SPEED
        elif movement == 2: self.player['pos'][1] += self.PLAYER_SPEED
        elif movement == 3: self.player['pos'][0] -= self.PLAYER_SPEED
        elif movement == 4: self.player['pos'][0] += self.PLAYER_SPEED

        self.player['pos'][0] = np.clip(self.player['pos'][0], self.player['size'], self.WIDTH - self.player['size'])
        self.player['pos'][1] = np.clip(self.player['pos'][1], self.player['size'], self.HEIGHT - self.player['size'])
        
        _, dist_after = self._find_nearest_alive_alien()
        if dist_after < dist_before:
            reward += 0.1

        # Firing
        if self.player['cooldown'] > 0:
            self.player['cooldown'] -= 1
        
        if space_held and self.player['cooldown'] == 0:
            self.player_projectiles.append({
                'pos': self.player['pos'].copy() - np.array([0, self.player['size']]),
                'size': 4
            })
            self.player['cooldown'] = self.PLAYER_FIRE_COOLDOWN
            # sfx: player_shoot.wav

        return reward

    def _update_aliens(self):
        # Movement
        move_down = False
        if self.alien_move_down_timer > 0:
            self.alien_move_down_timer -= 1
            if self.alien_move_down_timer == 0:
                self.alien_direction *= -1
        else:
            leftmost = min([a['pos'][0] for a in self.aliens if a['alive']], default=self.WIDTH)
            rightmost = max([a['pos'][0] for a in self.aliens if a['alive']], default=0)

            if (rightmost > self.WIDTH - 20 and self.alien_direction > 0) or \
               (leftmost < 20 and self.alien_direction < 0):
                move_down = True
                self.alien_move_down_timer = 10 # frames to move down

        for alien in self.aliens:
            if alien['alive']:
                if move_down:
                    alien['pos'][1] += 15
                else:
                    alien['pos'][0] += self.alien_direction * 1.0

        # Firing
        self.alien_fire_timer -= 1
        if self.alien_fire_timer <= 0:
            alive_aliens = [a for a in self.aliens if a['alive']]
            if alive_aliens:
                shooter = self.np_random.choice(alive_aliens)
                self.alien_projectiles.append({
                    'pos': shooter['pos'].copy() + np.array([0, shooter['size']]),
                    'size': 4
                })
                self.alien_fire_timer = self._get_alien_fire_cooldown()
                # sfx: alien_shoot.wav

    def _get_alien_fire_cooldown(self):
        rate = min(self.alien_fire_rate, self.ALIEN_FIRE_RATE_CAP)
        if rate > 0:
            return self.FPS / rate
        return float('inf')

    def _update_projectiles(self):
        self.player_projectiles = [p for p in self.player_projectiles if p['pos'][1] > 0]
        for p in self.player_projectiles:
            p['pos'][1] -= self.PROJECTILE_SPEED

        self.alien_projectiles = [p for p in self.alien_projectiles if p['pos'][1] < self.HEIGHT]
        for p in self.alien_projectiles:
            p['pos'][1] += self.ALIEN_PROJECTILE_SPEED

    def _handle_collisions(self):
        reward = 0

        # Player projectiles vs Aliens
        for p in self.player_projectiles[:]:
            for a in self.aliens:
                if a['alive'] and np.linalg.norm(p['pos'] - a['pos']) < a['size'] + p['size']:
                    a['alive'] = False
                    if p in self.player_projectiles: self.player_projectiles.remove(p)
                    self._create_explosion(a['pos'], self.COLOR_EXPLOSION, 30)
                    self.score += 100
                    self.aliens_destroyed += 1
                    self.alien_fire_rate += self.ALIEN_FIRE_RATE_INCREASE
                    reward += 10
                    # sfx: alien_explosion.wav
                    break
        
        # Alien projectiles vs Player
        if self.player['hit_timer'] > 0:
            self.player['hit_timer'] -= 1
        else:
            for p in self.alien_projectiles[:]:
                if np.linalg.norm(p['pos'] - self.player['pos']) < self.player['size'] + p['size']:
                    if p in self.alien_projectiles: self.alien_projectiles.remove(p)
                    self.player['lives'] -= 1
                    self.player['hit_timer'] = self.FPS * 2 # 2 seconds of invincibility
                    self._create_explosion(self.player['pos'], self.COLOR_PLAYER, 40)
                    reward -= 10
                    # sfx: player_hit.wav
                    break
        return reward
    
    def _create_explosion(self, pos, color, max_radius):
        self.particles.append({'pos': pos.copy(), 'radius': 0, 'max_radius': max_radius, 'life': 15, 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            p['radius'] += p['max_radius'] / 15.0

    def _check_termination(self):
        if self.player['lives'] <= 0:
            return True
        if not any(a['alive'] for a in self.aliens):
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        
        # Check if aliens reached bottom
        lowest_alien_y = max([a['pos'][1] for a in self.aliens if a['alive']], default=0)
        if lowest_alien_y > self.HEIGHT - 20:
             self.player['lives'] = 0 # Instant loss
             return True

        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles (explosions)
        for p in self.particles:
            alpha = int(255 * (p['life'] / 15.0))
            if alpha > 0:
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), (*p['color'], alpha))

        # Render projectiles
        for p in self.player_projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, *pos, p['size'], self.COLOR_PLAYER_PROJ)
            pygame.gfxdraw.aacircle(self.screen, *pos, p['size'], self.COLOR_PLAYER_PROJ)
        for p in self.alien_projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, *pos, p['size'], self.COLOR_ALIEN_PROJ)
            pygame.gfxdraw.aacircle(self.screen, *pos, p['size'], self.COLOR_ALIEN_PROJ)
        
        # Render aliens
        for a in self.aliens:
            if a['alive']:
                pos = (int(a['pos'][0]), int(a['pos'][1]))
                size = int(a['size'])
                rect = pygame.Rect(pos[0] - size, pos[1] - size, size * 2, size * 2)
                pygame.draw.rect(self.screen, self.COLOR_ALIEN, rect)

        # Render player
        if self.player['lives'] > 0:
            # Invincibility flash
            if self.player['hit_timer'] == 0 or (self.player['hit_timer'] % 10 < 5):
                p = self.player
                s = p['size']
                points = [
                    (p['pos'][0], p['pos'][1] - s),
                    (p['pos'][0] - s * 0.8, p['pos'][1] + s * 0.8),
                    (p['pos'][0] + s * 0.8, p['pos'][1] + s * 0.8)
                ]
                int_points = [(int(x), int(y)) for x, y in points]
                pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_PLAYER)
                pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Render score
        score_text = self.ui_font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Render lives
        lives_text = self.ui_font.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 180, 10))
        for i in range(self.player['lives']):
            s = 8
            x_pos = self.WIDTH - 100 + i * 25
            y_pos = 12 + s
            points = [
                (x_pos, y_pos - s),
                (x_pos - s * 0.8, y_pos + s * 0.8),
                (x_pos + s * 0.8, y_pos + s * 0.8)
            ]
            int_points = [(int(x), int(y)) for x, y in points]
            pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_PLAYER)

        # Render game over message
        if self.game_over:
            win = not any(a['alive'] for a in self.aliens) and self.player['lives'] > 0
            msg = "YOU WIN!" if win else "GAME OVER"
            color = (0, 255, 0) if win else (255, 0, 0)
            end_text = self.end_font.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player['lives'],
            "aliens_remaining": len([a for a in self.aliens if a['alive']])
        }
    
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Alien Invasion")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("      HUMAN PLAY MODE")
    print("="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # Get human input
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle window closing and game reset
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()
                total_reward = 0

        if terminated:
            print(f"Episode finished. Total reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            print("Press 'R' to play again or close the window.")
            
        clock.tick(env.FPS)

    pygame.quit()