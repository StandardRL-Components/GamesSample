
# Generated: 2025-08-27T16:20:16.692878
# Source Brief: brief_00007.md
# Brief Index: 7

        
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
        "Controls: Use arrow keys to move. Hold space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of descending alien invaders in this retro top-down arcade shooter."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.MAX_STEPS = 10000
        self.TOTAL_WAVES = 5

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_BULLET = (100, 255, 200)
        self.COLOR_ALIEN_BULLET = (255, 255, 255)
        self.COLOR_EXPLOSION = (255, 150, 50)
        self.COLOR_TEXT = (220, 220, 255)
        self.ALIEN_WAVE_COLORS = [
            (255, 80, 80),   # Wave 1: Red
            (80, 150, 255),  # Wave 2: Blue
            (200, 80, 255),  # Wave 3: Purple
            (255, 200, 80),  # Wave 4: Yellow
            (255, 100, 200)  # Wave 5: Pink
        ]
        
        # Player state
        self.player_pos = [0, 0]
        self.player_lives = 0
        self.player_speed = 6
        self.player_fire_cooldown = 8 # frames
        self.player_fire_timer = 0
        self.player_hit_timer = 0
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.current_wave = 0
        self.wave_transition_timer = 0
        self.wave_transition_duration = 90 # 3 seconds
        
        # Entity lists
        self.player_bullets = []
        self.aliens = []
        self.alien_bullets = []
        self.particles = []

        # Alien properties
        self.alien_speed = 0
        self.alien_fire_rate = 0
        self.alien_move_dir = 1
        
        # Initialize state variables
        self.reset()
        
        # Validate implementation after full initialization
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 40]
        self.player_lives = 3
        self.player_fire_timer = 0
        self.player_hit_timer = 0
        
        self.player_bullets.clear()
        self.aliens.clear()
        self.alien_bullets.clear()
        self.particles.clear()
        
        self.current_wave = 0
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # --- Game Logic ---
        self.steps += 1
        reward += 0.01 # Small reward for surviving a frame

        # Handle wave transitions
        if self.wave_transition_timer > 0:
            self.wave_transition_timer -= 1
            if self.wave_transition_timer == 0:
                self._start_next_wave()
                if not self.win:
                    reward += 100 # Reward for starting next wave
        else:
            # Handle player actions
            self._handle_player_input(movement, space_held)

            # Update game entities
            reward += self._update_player_bullets()
            self._update_aliens()
            self._update_alien_bullets()
            
            # Check for collisions
            self._check_collisions()

            # Check for wave completion
            if not self.aliens:
                if self.current_wave == self.TOTAL_WAVES:
                    self.win = True
                    reward += 500 # Win game bonus
                else:
                    self.wave_transition_timer = self.wave_transition_duration

        # Update timers and effects
        if self.player_fire_timer > 0: self.player_fire_timer -= 1
        if self.player_hit_timer > 0: self.player_hit_timer -= 1
        self._update_particles()
        
        # Check for termination
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_input(self, movement, space_held):
        # Movement
        if movement == 1: self.player_pos[1] -= self.player_speed # Up
        if movement == 2: self.player_pos[1] += self.player_speed # Down
        if movement == 3: self.player_pos[0] -= self.player_speed # Left
        if movement == 4: self.player_pos[0] += self.player_speed # Right
        
        # Clamp player position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.SCREEN_WIDTH - 20)
        self.player_pos[1] = np.clip(self.player_pos[1], 20, self.SCREEN_HEIGHT - 20)

        # Firing
        if space_held and self.player_fire_timer == 0:
            # sfx: player_shoot.wav
            bullet_pos = [self.player_pos[0], self.player_pos[1] - 15]
            self.player_bullets.append(pygame.Rect(bullet_pos[0] - 2, bullet_pos[1] - 5, 4, 10))
            self.player_fire_timer = self.player_fire_cooldown

    def _update_player_bullets(self):
        reward = 0
        aliens_hit_indices = set()
        bullets_to_remove = []

        for i, bullet in enumerate(self.player_bullets):
            bullet.y -= 10 # Bullet speed
            if bullet.bottom < 0:
                bullets_to_remove.append(i)
                continue
            
            hit_index = bullet.collidelist(self.aliens)
            if hit_index != -1:
                aliens_hit_indices.add(hit_index)
                bullets_to_remove.append(i)
                reward += 10 # Reward for destroying an alien

        # Remove hit aliens and create explosions
        if aliens_hit_indices:
            # sfx: explosion.wav
            aliens_hit_indices = sorted(list(aliens_hit_indices), reverse=True)
            for index in aliens_hit_indices:
                alien_rect = self.aliens.pop(index)
                self._create_explosion(alien_rect.center, self.ALIEN_WAVE_COLORS[self.current_wave - 1])
                self.score += 10

        # Remove used/off-screen bullets
        for index in sorted(bullets_to_remove, reverse=True):
            del self.player_bullets[index]
            
        return reward

    def _update_aliens(self):
        move_down = False
        for alien in self.aliens:
            if (alien.right > self.SCREEN_WIDTH and self.alien_move_dir > 0) or \
               (alien.left < 0 and self.alien_move_dir < 0):
                self.alien_move_dir *= -1
                move_down = True
                break
        
        for alien in self.aliens:
            alien.x += self.alien_speed * self.alien_move_dir
            if move_down:
                alien.y += 15 # Descend amount
            
            # Alien firing
            if self.np_random.random() < self.alien_fire_rate:
                # sfx: alien_shoot.wav
                bullet_rect = pygame.Rect(alien.centerx - 1, alien.bottom, 3, 7)
                self.alien_bullets.append(bullet_rect)

    def _update_alien_bullets(self):
        bullets_to_remove = []
        for i, bullet in enumerate(self.alien_bullets):
            bullet.y += 6 # Bullet speed
            if bullet.top > self.SCREEN_HEIGHT:
                bullets_to_remove.append(i)
        
        for index in sorted(bullets_to_remove, reverse=True):
            del self.alien_bullets[index]

    def _check_collisions(self):
        # Player vs Alien Bullets
        player_rect = self._get_player_rect()
        hit_index = player_rect.collidelist(self.alien_bullets)
        if hit_index != -1 and self.player_hit_timer == 0:
            del self.alien_bullets[hit_index]
            self._handle_player_hit()

        # Player vs Aliens
        hit_index = player_rect.collidelist(self.aliens)
        if hit_index != -1 and self.player_hit_timer == 0:
            alien_rect = self.aliens.pop(hit_index)
            self._create_explosion(alien_rect.center, self.ALIEN_WAVE_COLORS[self.current_wave-1])
            self._handle_player_hit()

        # Aliens reaching bottom
        aliens_to_remove = []
        for i, alien in enumerate(self.aliens):
            if alien.bottom > self.SCREEN_HEIGHT:
                aliens_to_remove.append(i)
                if self.player_lives > 0:
                    self.player_lives -= 1
                    # sfx: life_lost.wav
        
        if aliens_to_remove:
            for index in sorted(aliens_to_remove, reverse=True):
                del self.aliens[index]


    def _handle_player_hit(self):
        # sfx: player_hit.wav
        self.player_lives -= 1
        self.player_hit_timer = 60 # 2 seconds of invincibility
        self._create_explosion(self.player_pos, self.COLOR_EXPLOSION, count=40)

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.TOTAL_WAVES:
            return

        self.player_bullets.clear()
        self.alien_bullets.clear()
        self.aliens.clear()
        self.alien_move_dir = 1
        
        # Scale difficulty
        self.alien_speed = 1 + self.current_wave * 0.2
        self.alien_fire_rate = 0.001 + self.current_wave * 0.001

        # Spawn aliens in a grid
        num_cols = 8
        num_rows = 3 + min(self.current_wave, 2)
        h_spacing = 60
        v_spacing = 40
        grid_width = (num_cols - 1) * h_spacing
        start_x = (self.SCREEN_WIDTH - grid_width) / 2
        start_y = 50

        for row in range(num_rows):
            for col in range(num_cols):
                x = start_x + col * h_spacing
                y = start_y + row * v_spacing
                self.aliens.append(pygame.Rect(x, y, 30, 20))

    def _check_termination(self):
        if self.player_lives <= 0:
            return True
        if self.win:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_particles()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render aliens
        if self.aliens:
            color = self.ALIEN_WAVE_COLORS[self.current_wave - 1]
            for alien in self.aliens:
                pygame.draw.rect(self.screen, color, alien, border_radius=3)

        # Render bullets
        for bullet in self.player_bullets:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_BULLET, bullet, border_radius=2)
        for bullet in self.alien_bullets:
            pygame.draw.rect(self.screen, self.COLOR_ALIEN_BULLET, bullet)

        # Render player
        if self.player_lives > 0:
            is_invincible = self.player_hit_timer > 0
            # Flicker when invincible
            if not is_invincible or (is_invincible and self.steps % 10 < 5):
                px, py = int(self.player_pos[0]), int(self.player_pos[1])
                points = [(px, py - 15), (px - 12, py + 10), (px + 12, py + 10)]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Wave
        wave_str = f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}"
        if self.wave_transition_timer > 0:
            wave_str = f"WAVE {self.current_wave} CLEARED"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        
        # Lives
        life_icon_rect = pygame.Rect(0, 0, 12, 15)
        for i in range(self.player_lives):
            px = self.SCREEN_WIDTH - 20 - (i * 25)
            py = self.SCREEN_HEIGHT - 20
            points = [(px, py - 8), (px - 6, py + 5), (px + 6, py + 5)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Game Over / Win Text
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _create_explosion(self, pos, color, count=20):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = 1 + self.np_random.random() * 4
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = 15 + self.np_random.integers(0, 15)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifetime, 'max_life': lifetime, 'color': color})

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0:
                particles_to_remove.append(i)
        
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = int(life_ratio * 5)
            if radius > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                # Fade out color
                current_color = tuple(int(c * life_ratio) for c in p['color'])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, current_color)

    def _get_player_rect(self):
        return pygame.Rect(self.player_pos[0] - 12, self.player_pos[1] - 15, 24, 25)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
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
    # It's a demonstration of how to use the environment
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Gymnasium Arcade Shooter")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # --- Human Controls ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0 # Not used in this game

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    terminated = False
                    total_reward = 0
                    print("--- Game Reset ---")

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Print Info ---
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

        # Control the frame rate
        env.clock.tick(env.FPS)
        
    env.close()